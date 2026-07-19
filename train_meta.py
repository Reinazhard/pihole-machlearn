import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

OUTPUT_MATRIX = '/app/data/xgboost_training_matrix.csv' if os.path.exists('/app/data') else os.path.join(os.path.dirname(__file__), 'xgboost_training_matrix.csv')
META_MODEL = '/app/data/meta_model.json' if os.path.exists('/app/data') else os.path.join(os.path.dirname(__file__), 'meta_model.json')
ENCODER_FILE = '/app/data/categorical_encoder.pkl' if os.path.exists('/app/data') else os.path.join(os.path.dirname(__file__), 'categorical_encoder.pkl')

def main():
    print(f"Loading dataset from {OUTPUT_MATRIX}...")
    if not os.path.exists(OUTPUT_MATRIX):
        print(f"Error: Dataset {OUTPUT_MATRIX} not found. Please run Phase 1 first.")
        return

    df = pd.read_csv(OUTPUT_MATRIX)
    print(f"Loaded {len(df)} records.")

    # 1. Separate Features and Target
    X = df.drop(columns=['domain', 'label'])
    y = df['label']

    # 2. Categorical Encoding (Crucial)
    # We must explicitly handle NaNs in categorical columns and map unknown future ASNs to -1
    print("Preparing Categorical Encoders...")
    cat_cols = ['asn', 'tls_issuer']
    
    # Fill categorical missing values with 'MISSING' string
    X[cat_cols] = X[cat_cols].fillna('MISSING')
    
    # Initialize robust encoder
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Fit and Transform
    X[cat_cols] = encoder.fit_transform(X[cat_cols])
    
    # Save the encoder for production inference
    print(f"Saving categorical encoder to {ENCODER_FILE}...")
    joblib.dump(encoder, ENCODER_FILE)

    # 3. Handle Continuous Nulls (Sparsity-Aware Pass-Through)
    # Ensure domain_age_days and tls_timeout are proper numeric/NaN types
    X['domain_age_days'] = pd.to_numeric(X['domain_age_days'], errors='coerce')
    X['tls_timeout'] = pd.to_numeric(X['tls_timeout'], errors='coerce')
    
    # Log the shape of the data for sanity checking
    print("\nFeature Matrix Summary:")
    print(X.info())

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Training (XGBoost)
    print("\nTraining XGBoost Meta-Learner...")
    clf = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, # Kept shallow to prevent overfitting behavioral telemetry
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)

    print("\nMeta-Learner Evaluation:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Ad/Tracker']))

    # 6. Export Format (JSON)
    print(f"Saving XGBoost model to {META_MODEL}...")
    clf.save_model(META_MODEL)
    print("Phase 2: Meta-Learner Training Complete.")

if __name__ == '__main__':
    main()
