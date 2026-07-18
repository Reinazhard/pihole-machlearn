import sqlite3
import pandas as pd
import onnxruntime as ort
import subprocess
import time
import os
import numpy as np

GRAVITY_DB = os.environ.get('GRAVITY_DB', '/etc/pihole/gravity.db')
FTL_DB = os.environ.get('FTL_DB', '/etc/pihole/pihole-FTL.db')
MODEL_FILE = '/app/data/model.onnx' if os.path.exists('/app/data/model.onnx') else os.path.join(os.path.dirname(__file__), 'model.onnx')
MAJESTIC_FILE = '/app/data/majestic.csv' if os.path.exists('/app/data') else os.path.join(os.path.dirname(__file__), 'majestic.csv')
TIME_WINDOW_SEC = 300 # 5 minutes
MAX_LEN = 100
CONFIDENCE_THRESHOLD = 0.99

# Selective TLD Infrastructure Rule
INFRASTRUCTURE_SUFFIXES = [
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
]

def load_top_safe_domains(limit=100000):
    if not os.path.exists(MAJESTIC_FILE):
        return set()
    try:
        # Load up to limit, but also ensure any subdomains in the list are treated properly.
        # However, for bypass, we only care about the exact domain matches from the list.
        # Majestic uses column 2 for the domain
        df = pd.read_csv(MAJESTIC_FILE, usecols=[2], names=['domain'], header=0, nrows=limit)
        
        # Explicitly hardcode ultra-critical core domains that we never want blocked 
        # just in case they drop out of majestic or fail to parse.
        core_whitelist = {
            'google.com', 'www.google.com', 'facebook.com', 'www.facebook.com', 
            'instagram.com', 'www.instagram.com', 'whatsapp.com', 'www.whatsapp.com',
            'apple.com', 'www.apple.com', 'microsoft.com', 'www.microsoft.com',
            'amazon.com', 'www.amazon.com', 'netflix.com', 'www.netflix.com',
            'youtube.com', 'www.youtube.com', 'twitter.com', 'www.twitter.com',
            'linkedin.com', 'www.linkedin.com', 'reddit.com', 'www.reddit.com'
        }
        
        loaded_domains = set(df['domain'].str.lower().tolist())
        return loaded_domains.union(core_whitelist)
    except Exception as e:
        print(f"Error loading majestic.csv: {e}")
        return set()

def get_recent_allowed_domains():
    if not os.path.exists(FTL_DB):
        print(f"Error: Database {FTL_DB} not found.")
        return []
    
    # Use standard connection, Docker volume mapping does not like sqlite URI modes.
    conn = sqlite3.connect(FTL_DB, timeout=20.0)
        
    conn.text_factory = lambda b: b.decode(errors='ignore')
    recent_timestamp = int(time.time()) - TIME_WINDOW_SEC
    
    # status 2 = forwarded, 3 = cached (both mean it was allowed)
    query = f"""
        SELECT DISTINCT domain FROM queries 
        WHERE status IN (2, 3) 
        AND timestamp >= {recent_timestamp}
        AND domain != ''
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['domain'].tolist()

def filter_existing_blocks(domains):
    if not domains:
        return []
        
    if not os.path.exists(GRAVITY_DB):
        return domains
    
    conn = sqlite3.connect(GRAVITY_DB, timeout=20.0)
        
    placeholders = ','.join('?' for _ in domains)
    
    # Check exact blocks (type 1) and exact allows (type 0/2/4) to skip re-evaluating.
    # By checking type 0 (Whitelist) and 2 (Regex Whitelist), we prevent the 
    # detector from ever re-adding a domain the user explicitly allowed in the Pi-hole UI.
    query = f"SELECT domain FROM domainlist WHERE domain IN ({placeholders}) AND type IN (0, 1, 2, 4)"
    cursor = conn.cursor()
    cursor.execute(query, domains)
    existing = set(row[0] for row in cursor.fetchall())
    conn.close()
    
    return [d for d in domains if d not in existing]

def check_and_apply_wildcard(domains):
    if not domains:
        return domains # Return remaining domains to be exact-blocked
        
    conn = sqlite3.connect(GRAVITY_DB, timeout=20.0)
        
    cursor = conn.cursor()
    
    # Extract root domains to check for DGA wildcard aggregation
    import tldextract
    root_counts = {}
    domain_to_root = {}
    
    for d in domains:
        ext = tldextract.extract(d)
        root = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        domain_to_root[d] = root
        
    # Check how many times we've manually blocked subdomains of these roots
    roots_to_wildcard = set()
    exact_domains_to_keep = []
    
    for d in domains:
        root = domain_to_root[d]
        if root in roots_to_wildcard:
            continue
            
        cursor.execute("SELECT COUNT(*) FROM domainlist WHERE type = 1 AND domain LIKE ?", (f"%.{root}",))
        count = cursor.fetchone()[0]
        
        # If we have caught 3+ subdomains of this root, upgrade to wildcard
        if count >= 3:
            roots_to_wildcard.add(root)
        else:
            exact_domains_to_keep.append(d)
            
    conn.close()
    
    # Apply Wildcard Blocks
    if roots_to_wildcard:
        print(f"Aggregating {len(roots_to_wildcard)} root domains into regex wildcards...")
        conn_w = sqlite3.connect(GRAVITY_DB, timeout=20.0)
        cursor_w = conn_w.cursor()
        timestamp = int(time.time())
        
        records = []
        for root in roots_to_wildcard:
            # Type 3 = Regex Blacklist
            # Avoid backslash in f-string expression part
            safe_root = root.replace('.', r'\.')
            regex_str = f"(\\.|^){safe_root}$"
            records.append((3, regex_str, 1, timestamp, timestamp, 'Added by ML Detector (Wildcard Aggregation)'))
            
        cursor_w.executemany("""
            INSERT OR IGNORE INTO domainlist 
            (type, domain, enabled, date_added, date_modified, comment) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, records)
        conn_w.commit()
        conn_w.close()
        
    return exact_domains_to_keep

def block_domains(domains):
    if not domains:
        return
        
    # 1. Handle DGA Wildcard Aggregation First
    domains = check_and_apply_wildcard(domains)
    
    if not domains:
        # All caught domains were absorbed by wildcards
        print("Reloading Pi-hole DNS lists...")
        subprocess.run(["docker", "exec", "pihole", "pihole", "reloaddns"], check=False)
        return
        
    print(f"Blocking {len(domains)} new exact ad/tracker domains...")
    conn = sqlite3.connect(GRAVITY_DB, timeout=20.0)
    cursor = conn.cursor()
    
    timestamp = int(time.time())
    # type 1 = exact blacklist
    records = [(1, domain, 1, timestamp, timestamp, 'Added by ML Detector (Char-CNN High Confidence)') for domain in domains]
    
    cursor.executemany("""
        INSERT OR IGNORE INTO domainlist 
        (type, domain, enabled, date_added, date_modified, comment) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, records)
    
    conn.commit()
    conn.close()
    
    print("Reloading Pi-hole DNS lists...")
    subprocess.run(["docker", "exec", "pihole", "pihole", "reloaddns"], check=False)

def encode_domains(domains):
    encoded = np.zeros((len(domains), MAX_LEN), dtype=np.int64)
    for i, d in enumerate(domains):
        d_bytes = d.encode('utf-8', 'ignore')[:MAX_LEN]
        for j, b in enumerate(d_bytes):
            encoded[i, j] = b
    return encoded

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    if not os.path.exists(MODEL_FILE):
        print("Model file not found. Please run train.py first.")
        return
    print("Fetching recent queries...")
    recent_domains = get_recent_allowed_domains()
    if not recent_domains:
        print("No recent domains found.")
        return

    print(f"Checking {len(recent_domains)} domains against existing blocklists...")
    new_domains = filter_existing_blocks(recent_domains)
    if not new_domains:
        print("All recent domains are already evaluated or blocked.")
        return

    print("Loading Top 10k Safe Domains bypass list...")
    top_10k_safe = load_top_safe_domains()

    # Filter out domains that are known false positives using the dual-filter logic
    safe_new_domains = []
    for d in new_domains:
        d_lower = d.lower()
        
        # 1. Dynamic Top 10k Bypass (Strict Exact Match)
        if d_lower in top_10k_safe:
            print(f"Bypass (Top 10k): {d}")
            continue
            
        # 2. Selective TLD Infrastructure Rule (Suffix Match)
        is_infra = False
        for suffix in INFRASTRUCTURE_SUFFIXES:
            if d_lower.endswith(suffix):
                print(f"Bypass (Infrastructure CDN): {d}")
                is_infra = True
                break
                
        if not is_infra:
            safe_new_domains.append(d)
    
    new_domains = safe_new_domains
    
    if not new_domains:
        print("All recent domains are either blocked or whitelisted via bypass lists.")
        return

    print("Loading ONNX model and predicting...")
    session = ort.InferenceSession(MODEL_FILE, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    X = encode_domains(new_domains)
    outputs = session.run(None, {input_name: X})[0]
    
    probabilities = softmax(outputs)
    ad_probs = probabilities[:, 1]
    
    detected_ads = []
    for domain, prob in zip(new_domains, ad_probs):
        if prob > CONFIDENCE_THRESHOLD:
            detected_ads.append(domain)
            print(f"ML Detected Ad/Tracker: {domain} (Confidence: {prob:.4f})")
    
    if detected_ads:
        block_domains(detected_ads)
    else:
        print("No new ad/tracker domains detected in this batch exceeding the confidence threshold.")

if __name__ == '__main__':
    main()
