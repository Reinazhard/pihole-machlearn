import math
import collections
import re
import tldextract

KEYWORDS = ['ad', 'track', 'metric', 'telemetry', 'analytics', 'pixel', 'beacon', 'log', 'stats']

def shannon_entropy(s):
    if not s:
        return 0
    entropy = 0
    for x in set(s):
        p_x = float(s.count(x)) / len(s)
        entropy += - p_x * math.log(p_x, 2)
    return entropy

def extract_features(domain):
    ext = tldextract.extract(domain)
    # Combine subdomain and domain for analysis, ignore suffix (TLD)
    core_domain = f"{ext.subdomain}.{ext.domain}" if ext.subdomain else ext.domain
    
    length = len(core_domain)
    subdomain_depth = core_domain.count('.')
    
    vowels = sum(1 for c in core_domain if c.lower() in 'aeiou')
    consonants = sum(1 for c in core_domain if c.lower() in 'bcdfghjklmnpqrstvwxyz')
    vowel_ratio = vowels / length if length > 0 else 0
    
    digits = sum(1 for c in core_domain if c.isdigit())
    digit_ratio = digits / length if length > 0 else 0
    
    entropy = shannon_entropy(core_domain)
    
    features = {
        'length': length,
        'subdomain_depth': subdomain_depth,
        'vowel_ratio': vowel_ratio,
        'digit_ratio': digit_ratio,
        'entropy': entropy
    }
    
    for kw in KEYWORDS:
        features[f'has_kw_{kw}'] = 1 if kw in core_domain.lower() else 0
        
    return features
