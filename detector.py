import sqlite3
import pandas as pd
import joblib
import subprocess
import time
from features import extract_features
import os

GRAVITY_DB = os.environ.get('GRAVITY_DB', '/etc/pihole/gravity.db')
FTL_DB = os.environ.get('FTL_DB', '/etc/pihole/pihole-FTL.db')
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'model.pkl')
TIME_WINDOW_SEC = 300 # 5 minutes

def get_recent_allowed_domains():
    conn = sqlite3.connect(FTL_DB)
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
    records = [(1, domain, 1, timestamp, timestamp, 'Added by ML Detector') for domain in domains]
    
    cursor.executemany("""
        INSERT OR IGNORE INTO domainlist 
        (type, domain, enabled, date_added, date_modified, comment) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, records)
    
    conn.commit()
    conn.close()
    
    print("Reloading Pi-hole DNS lists...")
    # Adjust command based on your docker setup if necessary
    subprocess.run(["docker", "exec", "pihole", "pihole", "restartdns", "reload-lists"], check=False)

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

    print("Loading model and predicting...")
    clf = joblib.load(MODEL_FILE)
    
    # Extract features for new domains
    features_list = [extract_features(d) for d in new_domains]
    X = pd.DataFrame(features_list)
    
    predictions = clf.predict(X)
    
    # Identify which domains were classified as ads (label == 1)
    detected_ads = [domain for domain, pred in zip(new_domains, predictions) if pred == 1]
    
    if detected_ads:
        for ad in detected_ads:
            print(f"ML Detected Ad/Tracker: {ad}")
        block_domains(detected_ads)
    else:
        print("No new ad/tracker domains detected in this batch.")

if __name__ == '__main__':
    main()
