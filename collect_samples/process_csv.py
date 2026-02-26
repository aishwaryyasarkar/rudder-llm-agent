#!/usr/bin/env python3
import argparse
import glob
import os
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REQUIRED_METRIC_COLS = [
    "Pre_Avg_Hitrate",
    "Post_Avg_Hitrate",
    "Pre_Avg_Node_Freq",
    "Post_Avg_Node_Freq",
    "Pre_Avg_T_rpc",
    "Post_Avg_T_rpc",
]


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def _load_and_engineer(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    missing = [c for c in REQUIRED_METRIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_file}")

    # Make classifier scripts happy: they expect "Dataset" categorical feature.
    if "Dataset" not in df.columns:
        if "Graph_Name" in df.columns:
            df["Dataset"] = df["Graph_Name"]
        else:
            df["Dataset"] = "unknown"

    df["DeltaHitrate"] = df["Post_Avg_Hitrate"] - df["Pre_Avg_Hitrate"]
    df["DeltaFrequency"] = df["Post_Avg_Node_Freq"] - df["Pre_Avg_Node_Freq"]
    df["DeltaT_rpc"] = df["Post_Avg_T_rpc"] - df["Pre_Avg_T_rpc"]

    df["S"] = df["DeltaHitrate"] - df["DeltaFrequency"] - df["DeltaT_rpc"]
    df["eviction_label"] = (df["S"] > 0).astype(int)
    return df


def _find_csvs(base_dir: str) -> List[str]:
    files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Rudder tracker CSVs, compute labels, and split into train/test datasets."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Root directory to recursively search for per-rank tracker CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where merged/training/test CSVs are written.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--export_tensors",
        action="store_true",
        help="Also export standardized train/test tensors (.pt) and a scaler fit on train split.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    merged_out = os.path.join(args.output_dir, "merged_with_labels_normalized.csv")
    train_out = os.path.join(args.output_dir, "training_dataset.csv")
    test_out = os.path.join(args.output_dir, "test_dataset.csv")

    csv_files = _find_csvs(args.base_dir)
    # Ignore previously generated merged/train/test outputs when rerunning.
    csv_files = [
        f for f in csv_files if os.path.abspath(f) not in
        {os.path.abspath(merged_out), os.path.abspath(train_out), os.path.abspath(test_out)}
    ]

    if not csv_files:
        print(f"No CSV files found in {args.base_dir}")
        return

    dataframes = []
    for f in csv_files:
        try:
            dataframes.append(_load_and_engineer(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not dataframes:
        print("No valid CSV files to process.")
        return

    merged = pd.concat(dataframes, ignore_index=True)

    merged["DeltaHitrate_norm"] = _zscore(merged["DeltaHitrate"])
    merged["DeltaFrequency_norm"] = _zscore(merged["DeltaFrequency"])
    merged["DeltaT_rpc_norm"] = _zscore(merged["DeltaT_rpc"])
    merged["S_norm"] = (
        merged["DeltaHitrate_norm"] - merged["DeltaFrequency_norm"] - merged["DeltaT_rpc_norm"]
    )
    merged["eviction_label_norm"] = (merged["S_norm"] > 0).astype(int)

    merged.to_csv(merged_out, index=False)
    print(f"Merged dataset written to: {merged_out}")

    # Columns consumed by classifier training scripts (+ optional extra fields).
    training_cols = [
        "Dataset",
        "Rank",
        "Batch_Size",
        "Num_Total_Nodes",
        "Num_Partition_Nodes",
        "Num_Remote_Nodes",
        "Fan_Out",
        "buffer_size",
        "Eviction_Interval_ID",
        "Num_Evicted_Nodes",
        "Pre_Avg_Hitrate",
        "Pre_Avg_T_rpc",
        "Pre_Avg_Node_Freq",
        "eviction_label_norm",
    ]
    missing_training = [c for c in training_cols if c not in merged.columns]
    if missing_training:
        raise ValueError(
            f"Missing required columns for classifier datasets: {missing_training}"
        )

    dataset_df = merged[training_cols].copy()

    y = dataset_df["eviction_label_norm"]
    stratify = y if y.nunique() > 1 else None
    train_df, test_df = train_test_split(
        dataset_df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"Training dataset written to: {train_out} ({len(train_df)} rows)")
    print(f"Test dataset written to: {test_out} ({len(test_df)} rows)")

    if args.export_tensors:
        import torch

        feature_train = train_df.drop(columns=["eviction_label_norm"]).copy()
        feature_test = test_df.drop(columns=["eviction_label_norm"]).copy()
        y_train = train_df["eviction_label_norm"].values
        y_test = test_df["eviction_label_norm"].values

        # One-hot encode categorical dataset column and align test columns to train.
        feature_train = pd.get_dummies(feature_train, columns=["Dataset"])
        feature_test = pd.get_dummies(feature_test, columns=["Dataset"])
        feature_test = feature_test.reindex(columns=feature_train.columns, fill_value=0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(feature_train.values)
        X_test = scaler.transform(feature_test.values)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        tensor_out = os.path.join(args.output_dir, "classifier_train_test_tensors.pt")
        torch.save(
            {
                "X_train": X_train_tensor,
                "y_train": y_train_tensor,
                "X_test": X_test_tensor,
                "y_test": y_test_tensor,
                "feature_columns": list(feature_train.columns),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
            },
            tensor_out,
        )
        print(f"Tensor bundle written to: {tensor_out}")


if __name__ == "__main__":
    main()
