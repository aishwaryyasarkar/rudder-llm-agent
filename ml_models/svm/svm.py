import os
import argparse
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVM eviction model")
    parser.add_argument("--train_csv", type=str, default="../training_dataset.csv", help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, default="../test_dataset.csv", help="Path to test CSV")
    parser.add_argument("--model_dir", type=str, default="trained_model", help="Directory to save model artifacts")
    args = parser.parse_args()

    train_path = args.train_csv
    test_path  = args.test_csv
    model_dir  = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    numeric_features = [
        "Rank", "Batch_Size", "Num_Total_Nodes", "Num_Partition_Nodes",
        "Num_Remote_Nodes", "buffer_size", "Eviction_Interval_ID",
        "Num_Evicted_Nodes", "Pre_Avg_Hitrate", "Pre_Avg_T_rpc"
    ]
    categorical_features = ["Dataset"]
    label_col = "eviction_label_norm"

    preproc_path = os.path.join(model_dir, "eviction_preprocessor.joblib")

    if os.path.exists(preproc_path):
        print(f"Loading existing preprocessor from {preproc_path}")
        pipeline = joblib.load(preproc_path)
    else:
        print("Fitting new preprocessor (StandardScaler + OneHotEncoder)")
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
            ],
            remainder="drop",
        )
        pipeline = Pipeline([("pre", preprocessor)])
        pipeline.fit(df_train)
        joblib.dump(pipeline, preproc_path)
        print(f"Saved preprocessor to {preproc_path}")

    # Transform data
    X_train = pipeline.transform(df_train)
    X_test  = pipeline.transform(df_test)
    y_train = df_train[label_col].values.astype(int)
    y_test  = df_test[label_col].values.astype(int)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # SVM model
    clf = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
    )

    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    print("=== SVM (RBF) Results ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1       :", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(model_dir, "svm_eviction.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved SVM model to {model_path}")
