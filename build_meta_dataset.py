import os
import time
import pandas as pd
import numpy as np
import onnxruntime as ort
from network_prober import probe_domains
from train import fetch_data, encode_domains

MODEL_FILE = '/app/data/model.onnx' if os.path.exists('/app/data') else os.path.join(os.path.dirname(__file__), 'model.onnx')
OUTPUT_FILE = '/app/data/xgboost_training_matrix.csv' if os.path.exists('/app/data') else os.path.join(os.path.dirname(__file__), 'xgboost_training_matrix.csv')
CHUNK_SIZE = 500

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    print(f"Target dataset file: {OUTPUT_FILE}")
    
    # 1. Load or initialize checkpoint
    if os.path.exists(OUTPUT_FILE):
        print("Found existing checkpoint. Loading...")
        df_done = pd.read_csv(OUTPUT_FILE)
        processed_domains = set(df_done['domain'].tolist())
        print(f"Already processed {len(processed_domains)} domains.")
    else:
        print("No checkpoint found. Starting fresh.")
        processed_domains = set()
        # Initialize file with headers
        pd.DataFrame(columns=[
            'domain', 'label', 'cnn_prob', 'asn', 
            'asn_variance_score', 'ipv6_only', 'tls_issuer', 
            'tls_timeout', 'domain_age_days'
        ]).to_csv(OUTPUT_FILE, index=False)

    # 2. Fetch raw historical data
    print("Fetching raw datasets...")
    df_raw = fetch_data()
    
    # Stratified Sample: 10k Safe, 10k Ads
    df_safe = df_raw[df_raw['label'] == 0].sample(n=10000, random_state=42)
    df_ads = df_raw[df_raw['label'] == 1].sample(n=10000, random_state=42)
    df_target = pd.concat([df_safe, df_ads], ignore_index=True)
    
    # 3. Filter out already processed domains
    df_queue = df_target[~df_target['domain'].isin(processed_domains)].copy()
    print(f"Remaining domains in queue: {len(df_queue)}")
    
    if len(df_queue) == 0:
        print("Dataset generation complete!")
        return

    # 4. Generate CNN Probabilities
    print("Loading ONNX model for Char-CNN pre-scoring...")
    if not os.path.exists(MODEL_FILE):
        print(f"Error: ONNX model not found at {MODEL_FILE}")
        return
        
    session = ort.InferenceSession(MODEL_FILE, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Score in batches to avoid RAM spikes
    print("Generating cnn_prob for the queue...")
    cnn_probs = []
    batch_size = 5000
    queue_domains = df_queue['domain'].tolist()
    
    for i in range(0, len(queue_domains), batch_size):
        batch = queue_domains[i:i+batch_size]
        X_encoded = encode_domains(batch)
        outputs = session.run(None, {input_name: X_encoded})[0]
        probs = softmax(outputs)[:, 1] # Probability of Class 1 (Ad)
        cnn_probs.extend(probs)
        
    df_queue['cnn_prob'] = cnn_probs
    
    # 5. Stateful Chunking & Active Network Probing
    queue_records = df_queue.to_dict('records')
    total_chunks = (len(queue_records) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print(f"Starting active network probing across {total_chunks} chunks (Size: {CHUNK_SIZE})...")
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, len(queue_records))
        chunk = queue_records[start_idx:end_idx]
        
        chunk_domains = [row['domain'] for row in chunk]
        print(f"Probing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_domains)} domains)...")
        
        # Probe
        df_probed = probe_domains(chunk_domains)
        
        # Merge back labels and cnn_prob
        chunk_df = pd.DataFrame(chunk)
        df_merged = pd.merge(chunk_df[['domain', 'label', 'cnn_prob']], df_probed, on='domain', how='left')
        
        # Append to CSV
        df_merged.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        
        if chunk_idx < total_chunks - 1:
            print("Chunk complete. Sleeping 5s to bleed TCP sockets and respect upstream rate limits...")
            time.sleep(5)

    print("Phase 1: Dataset Generation Complete.")

if __name__ == '__main__':
    main()
