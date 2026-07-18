import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import sys
import requests

GRAVITY_DB = os.environ.get('GRAVITY_DB', '/etc/pihole/gravity.db')
FTL_DB = os.environ.get('FTL_DB', '/etc/pihole/pihole-FTL.db')
TRANCO_URL = "https://tranco-list.eu/download/J9NWX/1000000"

def fetch_data():
    if not os.path.exists(GRAVITY_DB):
        print(f"Error: Databases not found. Ensure {GRAVITY_DB} exists.")
        sys.exit(1)

    print("Fetching ad domains from gravity.db...")
    conn_grav = sqlite3.connect(GRAVITY_DB)
    conn_grav.text_factory = lambda b: b.decode(errors='ignore')
    df_ads = pd.read_sql_query("SELECT domain FROM gravity LIMIT 150000", conn_grav)
    conn_grav.close()
    df_ads['label'] = 1

    print("Downloading Majestic Million list for safe domains (to ensure enough data)...")
    if not os.path.exists("majestic.csv"):
        r = requests.get("http://downloads.majestic.com/majestic_million.csv")
        with open("majestic.csv", "wb") as f:
            f.write(r.content)
            
    df_safe = pd.read_csv("majestic.csv", usecols=[2], names=['domain'], header=0)
    df_safe = df_safe.head(150000) # Match the 150k ads for balance
    df_safe['label'] = 0
    df_safe = df_safe[['domain', 'label']]
    
    # Clean any NaN values
    df_ads = df_ads.dropna(subset=['domain'])
    df_safe = df_safe.dropna(subset=['domain'])
    
    print(f"Loaded {len(df_safe)} safe domains and {len(df_ads)} ad domains.")

    return pd.concat([df_ads, df_safe], ignore_index=True)

def main():
    df = fetch_data()
    print(f"Total dataset size: {len(df)} domains")

    print("Training TF-IDF + Logistic Regression model...")
    X = df['domain']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use character n-grams, extremely effective for finding domain patterns
    # Combine word-based and char-based n-grams for maximum pattern capture
    clf = make_pipeline(
        FeatureUnion([
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=100000)),
            ('word', TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b', max_features=50000))
        ]),
        LogisticRegression(max_iter=3000, class_weight='balanced', C=50.0, n_jobs=-1)
    )
    clf.fit(X_train, y_train)

    print("\nModel Evaluation:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Ad/Tracker']))

    print("Saving model to model.pkl...")
    joblib.dump(clf, 'model.pkl')
    print("Done.")

if __name__ == '__main__':
    main()
