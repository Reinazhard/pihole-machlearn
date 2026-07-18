import sqlite3
import subprocess
import os

GRAVITY_DB = os.environ.get('GRAVITY_DB', '/etc/pihole/gravity.db')
ML_LIST_FILE = os.environ.get('ML_LIST_FILE', '/etc/pihole/ml-blocklist.txt')
COMMENT_FLAG = 'Added by ML Detector'

def main():
    if not os.path.exists(GRAVITY_DB):
        print("gravity.db not found.")
        return
        
    conn = sqlite3.connect(GRAVITY_DB)
    cursor = conn.cursor()
    
    # 1. Find all domains added by the ML detector
    cursor.execute(f"SELECT domain FROM domainlist WHERE comment LIKE '%{COMMENT_FLAG}%'")
    domains = [row[0] for row in cursor.fetchall()]
    
    if not domains:
        print("No new ML domains to sweep.")
        conn.close()
        return
        
    print(f"Sweeping {len(domains)} domains from domainlist to local adlist...")
    
    # 2. Append to the local ml-blocklist.txt
    existing_domains = set()
    if os.path.exists(ML_LIST_FILE):
        with open(ML_LIST_FILE, 'r') as f:
            existing_domains = set(line.strip() for line in f if line.strip())
            
    with open(ML_LIST_FILE, 'a') as f:
        for domain in domains:
            if domain not in existing_domains:
                f.write(f"{domain}\n")
                
    # 3. Remove them from the domainlist table
    cursor.execute(f"DELETE FROM domainlist WHERE comment LIKE '%{COMMENT_FLAG}%'")
    conn.commit()
    conn.close()
    
    # 4. Ensure the local list is registered in Pi-hole
    conn = sqlite3.connect(GRAVITY_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT address FROM adlist WHERE address = ?", (f"file://{ML_LIST_FILE}",))
    if not cursor.fetchone():
        print("Registering ml-blocklist.txt in Pi-hole adlists...")
        cursor.execute("""
            INSERT INTO adlist (address, enabled, comment)
            VALUES (?, 1, 'ML Detector Sweep List')
        """, (f"file://{ML_LIST_FILE}",))
        conn.commit()
    conn.close()
    
    # 5. Run gravity update to compile the list
    print("Running Pi-hole gravity update...")
    subprocess.run(["docker", "exec", "pihole", "pihole", "-g"], check=False)
    print("Sweep complete.")

if __name__ == '__main__':
    main()
