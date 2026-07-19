import asyncio
import aiohttp
import aiodns
import ssl
import pandas as pd
import ipaddress
import random
from collections import Counter
from datetime import datetime
from dateutil.parser import parse as parse_date

# Constants
TIMEOUT_SEC = 1.5
SEMAPHORE_LIMIT = 100
CYMRU_V4 = 'origin.asn.cymru.com'
CYMRU_V6 = 'origin6.asn.cymru.com'

INFRASTRUCTURE_SUFFIXES = (
    '.googleapis.com', 
    '.akamaihd.net',
    '.cloudfront.net',
    '.amazonaws.com',
    '.shopeemobile.com',
    '.fbcdn.net',
    '.googleusercontent.com',
    '.susercontent.com',
    '.gstatic.com',
    '.whatsapp.net',
    '.facebook.com',
    '.facebook.net',
    '.instagram.com',
    '.discord.gg',
    '.discordapp.com',
    '.twimg.com'
)

async def _timeout(coro, timeout=TIMEOUT_SEC):
    """Wrapper to strictly enforce timeouts and swallow errors."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        return None

async def _cymru_asn_lookup(ip_str, resolver):
    """Resolve ASN for a single IP via Team Cymru DNS TXT query."""
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.version == 4:
            # e.g., 1.2.3.4 -> 4.3.2.1.origin.asn.cymru.com
            reversed_ip = '.'.join(reversed(ip_str.split('.')))
            query = f"{reversed_ip}.{CYMRU_V4}"
        else:
            # e.g., 2001:db8::1 -> 1.0.0.0.0.0.0.0...8.b.d.0.1.0.0.2.origin6.asn.cymru.com
            query = ip.reverse_pointer.replace('ip6.arpa', CYMRU_V6)
            
        result = await resolver.query_dns(query, 'TXT')
        if result and result[0].text:
            # Format: "ASN | IP | BGP Prefix | Country | Registry | Date"
            # Return just the ASN part (e.g., "13335")
            return result[0].text.split('|')[0].strip()
    except Exception:
        pass
    return None

async def _cymru_asn(domain, resolver):
    """Resolve domain to IPs, sample up to 3, and calculate ASN & variance."""
    try:
        # Try A records first
        records = await resolver.query_dns(domain, 'A')
        ips = [r.host for r in records]
        ipv6_only = False
    except aiodns.error.DNSError:
        # Fallback to AAAA if A fails
        try:
            records = await resolver.query_dns(domain, 'AAAA')
            ips = [r.host for r in records]
            ipv6_only = True
        except aiodns.error.DNSError:
            return None, 0.0, False # NXDOMAIN

    if not ips:
        return None, 0.0, ipv6_only

    # Fast-Flux check: Sample up to 3 IPs
    sample_ips = random.sample(ips, min(len(ips), 3))
    
    tasks = [_cymru_asn_lookup(ip, resolver) for ip in sample_ips]
    asns = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out failed lookups
    valid_asns = [asn for asn in asns if isinstance(asn, str)]
    
    if not valid_asns:
        return None, 0.0, ipv6_only

    # Calculate mode for primary ASN
    asn_counts = Counter(valid_asns)
    primary_asn = asn_counts.most_common(1)[0][0]
    
    # Variance score: Unique ASNs / Sample size
    variance_score = len(set(valid_asns)) / len(sample_ips)
    
    return primary_asn, variance_score, ipv6_only

async def _tls_issuer(domain):
    """Extract TLS issuer Common Name by terminating handshake early."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        reader, writer = await asyncio.open_connection(domain, 443, ssl=ctx)
        cert = writer.get_extra_info('peercert')
        writer.close()
        await writer.wait_closed()
        
        if not cert:
            return None
            
        # Parse issuer tuple: ((('countryName', 'US'),), (('organizationName', 'Let\'s Encrypt'),), ...)
        for rdn in cert.get('issuer', []):
            if rdn and rdn[0][0] == 'commonName':
                return rdn[0][1]
    except Exception:
        pass
    return None

async def _domain_age_days(domain, session):
    """Query RDAP for domain registration date, fail fast on 429."""
    try:
        async with session.get(f"https://rdap.org/domain/{domain}") as resp:
            if resp.status != 200:
                return None
                
            data = await resp.json()
            for event in data.get('events', []):
                if event.get('eventAction') == 'registration':
                    event_date = parse_date(event.get('eventDate'))
                    # Naive datetime math for age in days
                    return (datetime.utcnow().astimezone() - event_date).days
    except Exception:
        pass
    return None

async def _extract_all(domain, sem, resolver, session):
    """Coordinate all probes for a single domain under a semaphore lock."""
    async with sem:
        asn_task = _timeout(_cymru_asn(domain, resolver))
        tls_task = _timeout(_tls_issuer(domain))
        age_task = _timeout(_domain_age_days(domain, session))
        
        asn_result, tls, age = await asyncio.gather(asn_task, tls_task, age_task)
        
        # Unpack ASN tuple if valid
        asn, variance, ipv6_only = asn_result if asn_result else (None, 0.0, False)
        
        # Calculate lexical infrastructure flag
        is_core_infra = True if domain.lower().endswith(INFRASTRUCTURE_SUFFIXES) else False
        
        return {
            "domain": domain,
            "asn": asn,
            "asn_variance_score": variance,
            "ipv6_only": ipv6_only,
            "tls_issuer": tls,
            "tls_timeout": tls is None,
            "domain_age_days": age,
            "is_core_infra": is_core_infra
        }

async def _probe_domains_async(domains):
    """Internal async orchestrator."""
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    resolver = aiodns.DNSResolver()
    
    async with aiohttp.ClientSession() as session:
        tasks = [_extract_all(d, sem, resolver, session) for d in domains]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    # Handle massive structural failures smoothly
    clean_results = []
    for d, r in zip(domains, results):
        if isinstance(r, dict):
            clean_results.append(r)
        else:
            # Fallback empty row if entirely crashed
            clean_results.append({
                "domain": d, "asn": None, "asn_variance_score": 0.0,
                "ipv6_only": False, "tls_issuer": None, 
                "tls_timeout": True, "domain_age_days": None,
                "is_core_infra": False
            })
            
    return pd.DataFrame(clean_results)

def probe_domains(domains):
    """
    Public synchronous API.
    Accepts a list of domains, returns a Pandas DataFrame of extracted network features.
    """
    if not domains:
        return pd.DataFrame()
        
    return asyncio.run(_probe_domains_async(domains))

if __name__ == '__main__':
    # Quick sanity test block
    test_domains = ["google.com", "anthropic.com", "definitely-not-real-tracker-999.xyz"]
    df = probe_domains(test_domains)
    print(df)
