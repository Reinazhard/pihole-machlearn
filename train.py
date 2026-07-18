import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import onnx
import os
import sys
import requests

GRAVITY_DB = os.environ.get('GRAVITY_DB', '/etc/pihole/gravity.db')
FTL_DB = os.environ.get('FTL_DB', '/etc/pihole/pihole-FTL.db')
MAX_LEN = 100
BATCH_SIZE = 512

class CharCNN(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=32, num_classes=2):
        super(CharCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        
        self.fc1 = nn.Linear(128 * (MAX_LEN // 8), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embed(x) 
        x = x.transpose(1, 2) 
        
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.reshape(x.size(0), -1) 
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def encode_domains(domains):
    encoded = np.zeros((len(domains), MAX_LEN), dtype=np.int64)
    for i, d in enumerate(domains):
        d_bytes = d.encode('utf-8', 'ignore')[:MAX_LEN]
        for j, b in enumerate(d_bytes):
            encoded[i, j] = b
    return encoded

def fetch_data():
    if not os.path.exists(GRAVITY_DB):
        print(f"Error: Databases not found. Ensure {GRAVITY_DB} exists.")
        sys.exit(1)

    print("Fetching ad domains from gravity.db...")
    try:
        ro_uri = f"file:{GRAVITY_DB}?mode=ro"
        conn_grav = sqlite3.connect(ro_uri, uri=True, timeout=20.0)
    except sqlite3.OperationalError:
        conn_grav = sqlite3.connect(GRAVITY_DB, timeout=20.0)
        
    conn_grav.text_factory = lambda b: b.decode(errors='ignore')
    df_ads = pd.read_sql_query("SELECT domain FROM gravity LIMIT 150000", conn_grav)
    conn_grav.close()
    df_ads['label'] = 1

    print("Downloading Majestic Million list for safe domains (to ensure enough data)...")
    majestic_path = "/app/data/majestic.csv" if os.path.exists("/app/data") else "majestic.csv"
    if os.path.exists(majestic_path):
        print("Removing old majestic.csv to force fresh download...")
        os.remove(majestic_path)
        
    r = requests.get("http://downloads.majestic.com/majestic_million.csv")
    with open(majestic_path, "wb") as f:
        f.write(r.content)
            
    df_safe = pd.read_csv(majestic_path, usecols=[2], names=['domain'], header=0)
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

    print("Encoding domain characters...")
    X = encode_domains(df['domain'].tolist())
    y = df['label'].values

    # Train/Test Split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    
    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_test, y_test = torch.tensor(X[test_idx]), torch.tensor(y[test_idx])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = CharCNN().to(device)
    # Loss Function with class weights
    # We heavily penalize false positives (predicting Safe as Ad) by weighting the Safe class (0) higher.
    # Class 0 weight: 3.0, Class 1 weight: 1.0
    weights = torch.tensor([3.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training Char-CNN...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {100 * correct / total:.2f}%")

    print("Exporting model to ONNX...")
    model.eval()
    model.to('cpu')
    dummy_input = torch.zeros((1, MAX_LEN), dtype=torch.long)
    torch.onnx.export(
        model, 
        dummy_input, 
        "/app/data/model.onnx" if os.path.exists("/app/data") else "model.onnx", 
        export_params=True, 
        opset_version=18, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    data_path = "/app/data/model.onnx.data" if os.path.exists("/app/data") else "model.onnx.data"
    if os.path.exists(data_path):
        try:
            os.remove(data_path)
        except:
            pass
    print("Exported to model.onnx")

if __name__ == '__main__':
    main()
