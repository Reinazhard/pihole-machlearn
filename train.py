import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from features import extract_features
import os
import sys

GRAVITY_DB = os.environ.get('GRAVITY_DB', '/etc/pihole/gravity.db')
FTL_DB = os.environ.get('FTL_DB', '/etc/pihole/pihole-FTL.db')

def fetch_data():
    if not os.path.exists(GRAVITY_DB) or not os.path.exists(FTL_DB):
        print(f"Error: Databases not found. Ensure {GRAVITY_DB} and {FTL_DB} exist.")
        sys.exit(1)

    print("Fetching ad domains from gravity.db...")
    # Fetch known ad domains (limit to 50k to balance dataset and speed up training)
    # Using ignore errors for utf8 decoding
    conn_grav = sqlite3.connect(GRAVITY_DB)
    conn_grav.text_factory = lambda b: b.decode(errors='ignore')
    df_ads = pd.read_sql_query("SELECT domain FROM gravity LIMIT 50000", conn_grav)
    conn_grav.close()
    df_ads['label'] = 1

    print("Fetching safe domains from pihole-FTL.db...")
    # Fetch safe domains (status 2 or 3 = forwarded/cached, i.e., allowed)
    # Using group by to get the most frequently queried allowed domains
    conn_ftl = sqlite3.connect(FTL_DB)
    conn_ftl.text_factory = lambda b: b.decode(errors='ignore')
    df_safe = pd.read_sql_query("""
        SELECT domain FROM queries 
        WHERE status IN (2, 3) AND domain != ''
        GROUP BY domain 
        ORDER BY count(domain) DESC 
        LIMIT 50000
    """, conn_ftl)
    conn_ftl.close()
    df_safe['label'] = 0

    return pd.concat([df_ads, df_safe], ignore_index=True)

def main():
    df = fetch_data()
    print(f"Total dataset size: {len(df)} domains")

    print("Extracting features (this may take a minute)...")
    # Apply feature extraction
    feature_dicts = df['domain'].apply(extract_features)
    X = pd.DataFrame(feature_dicts.tolist())
    y = df['label']

    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("\nModel Evaluation:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Ad/Tracker']))

    print("Saving model to model.pkl...")
    joblib.dump(clf, 'model.pkl')
    print("Done.")

if __name__ == '__main__':
    main()
