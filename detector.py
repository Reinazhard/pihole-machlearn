import sqlite3
import pandas as pd
import onnxruntime as ort
import subprocess
import time
import os
import numpy as np

GRAVITY_DB = os.environ.get('GRAVITY_DB', '/etc/pihole/gravity.db')
FTL_DB = os.environ.get('FTL_DB', '/etc/pihole/pihole-FTL.db')
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'model.onnx')
TIME_WINDOW_SEC = 300 # 5 minutes
MAX_LEN = 100
CONFIDENCE_THRESHOLD = 0.95

def get_recent_allowed_domains():
    conn = sqlite3.connect(FTL_DB)
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
    
    conn = sqlite3.connect(GRAVITY_DB)
    placeholders = ','.join('?' for _ in domains)
    query = f"SELECT domain FROM domainlist WHERE domain IN ({placeholders})"
    cursor = conn.cursor()
    cursor.execute(query, domains)
    existing = set(row[0] for row in cursor.fetchall())
    conn.close()
    
    return [d for d in domains if d not in existing]

def block_domains(domains):
    if not domains:
        return
        
    print(f"Blocking {len(domains)} new ad/tracker domains...")
    conn = sqlite3.connect(GRAVITY_DB)
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
    subprocess.run(["docker", "exec", "pihole", "pihole", "restartdns", "reload-lists"], check=False)

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
