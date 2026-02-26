import os
import argparse
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Set random seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Load and align train/test data
def load_data(train_path, test_path, feature_cols, label_col):
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # One-hot encode Dataset
    df_train = pd.get_dummies(df_train, columns=['Dataset'])
    df_test  = pd.get_dummies(df_test,  columns=['Dataset'])
    df_test  = df_test.reindex(columns=df_train.columns, fill_value=0)

    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values.astype(int)
    X_test  = df_test[feature_cols].values
    y_test  = df_test[label_col].values.astype(int)
    return X_train, y_train, X_test, y_test

# Build and fit preprocessing pipeline
def build_preprocessor(df_sample, numeric_features, categorical_features):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
    ], remainder="drop")
    pipe = Pipeline([("pre", preprocessor)])
    pipe.fit(df_sample)
    return pipe

# Train TabNet
def train_tabnet(X_train, y_train, X_valid, y_valid, params):
    clf = TabNetClassifier(device_name='cuda', **params)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=['valid'],
        eval_metric=['accuracy'],
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    return clf

# Evaluate
def evaluate(clf, X, y):
    preds = clf.predict(X)
    print("Accuracy:", accuracy_score(y, preds))
    print("Precision:", precision_score(y, preds))
    print("Recall:", recall_score(y, preds))
    print("F1:", f1_score(y, preds))
    print(classification_report(y, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TabNet eviction model")
    parser.add_argument("--train_csv", type=str, default="../training_dataset.csv", help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, default="../test_dataset.csv", help="Path to test CSV")
    parser.add_argument("--model_dir", type=str, default="trained_model", help="Directory to save model artifacts")
    args = parser.parse_args()

    train_path = args.train_csv
    test_path  = args.test_csv
    model_dir  = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # Define your feature sets
    numeric_features = [
        'Rank','Batch_Size','Num_Total_Nodes','Num_Partition_Nodes',
        'Num_Remote_Nodes','buffer_size','Eviction_Interval_ID',
        'Num_Evicted_Nodes','Pre_Avg_Hitrate',
        'Pre_Avg_T_rpc'
    ]
    categorical_features = ['Dataset']
    label_col = 'eviction_label_norm'

    # Read a sample to fit preprocessor
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # Build & save preprocessing pipeline
    pipeline = build_preprocessor(df_train, numeric_features, categorical_features)
    joblib.dump(pipeline, os.path.join(model_dir, "eviction_preprocessor.joblib"))
    print(f"Saved preprocessing pipeline to {os.path.join(model_dir, 'eviction_preprocessor.joblib')}")

    # Load, preprocess and split
    feature_cols = numeric_features + [
        col for col in pipeline.named_steps['pre']
                           .named_transformers_['cat']
                           .get_feature_names_out(categorical_features)
    ]
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, feature_cols, label_col)

    # Apply scaling & encoding
    X_train = pipeline.transform(df_train)
    X_test  = pipeline.transform(df_test)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # TabNet hyperparameters
    params = {
        'n_d': 64,
        'n_a': 64,
        'n_steps': 5,
        'gamma': 1.5,
        'optimizer_params': dict(lr=2e-2, weight_decay=1e-3),
        'mask_type': 'sparsemax'
    }

    # Train & evaluate
    clf = train_tabnet(X_train, y_train, X_test, y_test, params)
    print("Validation results:")
    evaluate(clf, X_test, y_test)

    # Save TabNet weights
    clf.save_model(os.path.join(model_dir, "tabnet_eviction")) # saves as .zip
    print(f"Saved TabNet model to {os.path.join(model_dir,'tabnet_eviction.pth')}")
