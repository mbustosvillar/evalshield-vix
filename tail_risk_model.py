"""
tail_risk_model.py
==================
Lightweight Transformer-based Tail-Risk Predictor.
Uses a time-series Transformer architecture to predict 'is_crash_30d'.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# CONFIG
SEQ_LENGTH = 30  # Look back 30 trading days
INPUT_DIM = 8    # [spy, vix, vix3m, vvix, skew, tnx, tyx, term_structure]
HIDDEN_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 20

class TailRiskTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TailRiskTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LENGTH, hidden_dim))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, seq, features]
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        # We only take the last time step for prediction
        x = x[:, -1, :]
        return self.fc(x)

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    feature_cols = ["spy", "vix", "vix3m", "vvix", "skew", "tnx", "tyx", "vix_term_structure"]
    target_col = "is_crash_30d"
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    X, y = [], []
    for i in range(len(scaled_features) - SEQ_LENGTH):
        X.append(scaled_features[i : i + SEQ_LENGTH])
        y.append(df[target_col].iloc[i + SEQ_LENGTH])
    
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).view(-1, 1), scaler

def train_model(data_path, model_path="tail_risk_model.pth"):
    X, y, scaler = preprocess_data(data_path)
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    model = TailRiskTransformer(INPUT_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("[INFO] Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }, model_path)
    print(f"[SUCCESS] Model saved to {model_path}")

if __name__ == "__main__":
    import os
    if os.path.exists("historical_training_data.csv"):
        train_model("historical_training_data.csv")
    else:
        print("[ERROR] Training data not found. Run data_collector.py first.")
