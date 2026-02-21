import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import os

# Set random seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Define an enhanced MLP model with BatchNorm and extra hidden layers
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

# Load datasets
def load_data(train_path, test_path, feature_cols, label_col):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # One-hot encode Dataset column
    df_train = pd.get_dummies(df_train, columns=['Dataset'])
    df_test = pd.get_dummies(df_test, columns=['Dataset'])

    # Align columns
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

    # Extract features and labels
    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[label_col].values
    return X_train, y_train, X_test, y_test

# Preprocess, scale, and tensor conversion
def prepare_tensors(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    return X_train_t, y_train_t, X_test_t, y_test_t, scaler

# Training loop
def train_model(model, train_loader, test_loader, epochs=100, lr=1e-4, wd=1e-3, device=None):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    patience = 10
    trigger = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                labels = (preds>=0.5).float()
                correct += (labels==yb).sum().item()
                total += yb.size(0)
        val_loss /= len(test_loader.dataset)
        val_acc = correct/total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss<best_loss:
            best_loss=val_loss
            best_state=model.state_dict().copy()
            trigger=0
        else:
            trigger+=1
            if trigger>=patience:
                print("Early stopping\n")
                break
    if best_state: model.load_state_dict(best_state)
    return model

# Final evaluation
def evaluate(model, loader, device=None):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            bin_preds = (preds>=0.5).float().cpu().numpy()
            all_preds.extend(bin_preds)
            all_labels.extend(yb.cpu().numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds))
    print("Recall:", recall_score(all_labels, all_preds))
    print("F1:", f1_score(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))

# Main entry
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train MLP eviction model")
    parser.add_argument("--train_csv", type=str, default="../training_dataset.csv", help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, default="../test_dataset.csv", help="Path to test CSV")
    parser.add_argument("--model_dir", type=str, default="trained_model", help="Directory to save model artifacts")
    args = parser.parse_args()

    train_path = args.train_csv
    test_path  = args.test_csv
    model_dir = args.model_dir

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    os.makedirs(model_dir, exist_ok=True)

    # Define feature columns
    numeric_features = ['Rank','Batch_Size','Num_Total_Nodes','Num_Partition_Nodes',
            'Num_Remote_Nodes','buffer_size','Eviction_Interval_ID',
            'Num_Evicted_Nodes','Pre_Avg_Hitrate', 'Pre_Avg_T_rpc']

    categorical_features = ['Dataset']
    label_col = 'eviction_label_norm'

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
    ], remainder="drop")

    pipeline = Pipeline([
        ("pre", preprocessor),
    ])
    pipeline.fit(df_train)  # learns scaling & one-hot categories

    # transform into numpy arrays
    X_train = pipeline.transform(df_train)
    y_train = df_train[label_col].values
    X_test  = pipeline.transform(df_test)
    y_test  = df_test[label_col].values

    # save the pipeline
    joblib.dump(pipeline, os.path.join(model_dir, "eviction_preprocessor.joblib"))

    # train MLP on X_train / y_train as before
    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    te_ds = TensorDataset(torch.tensor(X_test,  dtype=torch.float32),
                          torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1))

    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(X_train.shape[1])
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model.to(device)

    model = train_model(model, tr_loader, te_loader, epochs=10, lr=1e-4, wd=1e-3, device=device)
    evaluate(model, te_loader, device=device)

    # save MLP weights
    torch.save(model.state_dict(), os.path.join(model_dir, "mlp_eviction.pth"))
    print(f"Saved MLP weights to {os.path.join(model_dir,'mlp.pth')}")
