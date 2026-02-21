import os
import joblib
import torch
from classifier_models.mlp.mlp import MLPClassifier
import pandas as pd
from collections import OrderedDict
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


def compute_eviction_label(pre_metrics: dict, post_metrics: dict) -> int:
    """
    Given two successive metric dicts (before and after an eviction interval),
    compute the binary label: 1 if (delta hit rate − delta RPC latency) > 0, else 0.
    """
    # Calculate changes
    delta_hr = post_metrics['Pre_Avg_Hitrate'] - pre_metrics['Pre_Avg_Hitrate']
    delta_t  = post_metrics['Pre_Avg_T_rpc'] - pre_metrics['Pre_Avg_T_rpc']
    S = delta_hr - delta_t
    return int(S > 0)

def label_stream(metrics_stream: list[dict]) -> list[int]:
    """
    Takes a time-ordered list of metric dicts. Pairs each entry with its successor
    to generate a list of labels (length N-1).
    """
    labels = []
    for prev, curr in zip(metrics_stream, metrics_stream[1:]):
        labels.append(compute_eviction_label(prev, curr))
    return labels

class MLPEvictionClassifier:
    """
    Loads a trained MLP model to decide whether to evict a node based on the metrics
    """
    def __init__(self, 
            model_dir: str, 
            device: torch.device,
            dataset: str,
            rank: int,
            batch_size: int,
            num_total_nodes: int,
            num_partition_nodes: int,
            num_remote_nodes: int,
            buffer_size: int,
            enable_finetune: bool = False,
            finetune_interval: int = 50,
        ):
        self.device = device

        # Set static features
        self.static_features = {
            "Dataset": dataset,
            "Rank": rank,
            "Batch_Size": batch_size,
            "Num_Total_Nodes": num_total_nodes,
            "Num_Partition_Nodes": num_partition_nodes,
            "Num_Remote_Nodes": num_remote_nodes,
            "buffer_size": buffer_size
        }

        print(f"Static features set: {self.static_features}")

        # load preprocessing pipeline
        pipe_path = os.path.join(model_dir, "eviction_preprocessor.joblib")
        self.pipeline = joblib.load(pipe_path)

        # load MLP and weights
        ct = self.pipeline.named_steps['pre']
        in_dim = len(ct.get_feature_names_out())
        raw_state = torch.load(os.path.join(model_dir, "mlp_eviction.pth"), map_location=self.device)

        # remove "module." prefix from all keys
        new_state = OrderedDict()
        for k, v in raw_state.items():
            # if the key starts with "module.", strip that off
            name = k[len("module."):] if k.startswith("module.") else k
            new_state[name] = v

        # now load into a plain model
        self.model = MLPClassifier(in_dim).to(self.device)
        self.model.load_state_dict(new_state)
        self.model.eval()

        # Online fine-tuning setup
        self.enable_finetune = enable_finetune

        # Args FIXME: adjust later 
        self._ft_interval = finetune_interval  # fine-tune interval
        self._ft_epochs = 3 # epochs per online update
        self._ft_lr = 1e-4 # learning rate for head

        # Buffer and counter
        self._counter = 0
        self._X_buffer = []
        self._y_buffer = []
        self._prev_metrics = None  # previous metrics for labeling

        # Optimizer & loss for head fine-tuning
        # Freeze-all then unfreeze head in finetune call
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self._ft_lr)
        self._criterion = torch.nn.BCEWithLogitsLoss()
        # BatchNorm layers to keep in eval mode
        self._bn_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
     
    def decide_eviction(self, metrics: dict):
        _ = metrics.pop("epoch", None)
        _ = metrics.pop("step",  None)

        xt = self._make_feature_tensor(metrics)

        # forward pass
        with torch.no_grad():
            prob = torch.sigmoid(self.model(xt)).item()

        decision = "yes, evict" if prob >= 0.5 else "no, do not evict"
        
        print(f"MLP eviction probability: {prob:.4f}, decision: {decision}")
        if self.enable_finetune:
            print(f"MLP eviction fine-tuning enabled, counter: {self._counter}/{self._ft_interval}")
            # If we have a previous entry, compute label
            if self._prev_metrics is not None:
                label = compute_eviction_label(self._prev_metrics, metrics)
                self._X_buffer.append(xt)
                self._y_buffer.append(label)
                self._counter += 1
                # Periodic fine-tune
                if self._counter >= self._ft_interval:
                    self._finetune_head()
                    self._counter = 0
                    self._X_buffer.clear()
                    self._y_buffer.clear()
            # Update previous metrics for next call
            # Store only dynamic fields for labeling
            self._prev_metrics = {
                'Pre_Avg_Hitrate': metrics['Pre_Avg_Hitrate'],
                'Pre_Avg_T_rpc':  metrics['Pre_Avg_T_rpc']
            }
        return decision

    def _make_feature_tensor(self, metrics: dict) -> torch.Tensor:
        """
        Build and transform a single-example feature tensor from metrics dict.
        """
        full = {**self.static_features, **metrics}
        df = pd.DataFrame([full])
        X = self.pipeline.transform(df)
        xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        return xt

    def _finetune_head(self):
        """
        Fine-tune only the last linear layer (head) on buffered examples.
        """
        print("Fine-tuning MLP head on buffered examples...")

        # Freeze all except head
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze the head 
        head = self.model.net[-2]
        for p in head.parameters():
            p.requires_grad = True

        # Prepare DataLoader on buffer
        X = torch.cat(self._X_buffer, dim=0)
        y = torch.tensor(self._y_buffer, dtype=torch.float32, device=self.device)
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=len(ds), shuffle=True)

        # Make optimizer focus on head only (safer/cleaner)
        self._optimizer = torch.optim.Adam(head.parameters(), lr=self._ft_lr)
        self.model.eval()

        with torch.enable_grad():
            for _ in range(self._ft_epochs):
                for xb, yb in loader:
                    logits = self.model(xb).squeeze(-1)
                    loss = self._criterion(logits, yb)
                    self._optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self._optimizer.step()
        self.model.eval()

class TabNetEvictionClassifier:
    """
    TabNet-based eviction agent with periodic online head fine-tuning using tensors.
    """
    def __init__(self,
                 model_dir: str,
                 device: torch.device,
                 dataset: str,
                 rank: int,
                 batch_size: int,
                 num_total_nodes: int,
                 num_partition_nodes: int,
                 num_remote_nodes: int,
                 buffer_size: int,
                 enable_finetune: bool = False,
                 finetune_interval: int = 50
                ):
        self.device = device

        # static features
        self.static_features = {
            "Dataset": dataset,
            "Rank": rank,
            "Batch_Size": batch_size,
            "Num_Total_Nodes": num_total_nodes,
            "Num_Partition_Nodes": num_partition_nodes,
            "Num_Remote_Nodes": num_remote_nodes,
            "buffer_size": buffer_size
        }

        # load preprocessing pipeline
        pipe_path = os.path.join(model_dir, "eviction_preprocessor.joblib")
        self.pipeline = joblib.load(pipe_path)

        # initialize TabNet model
        params = {'n_d': 64, 'n_a': 64, 'n_steps': 5, 'gamma': 1.5, 'mask_type': 'sparsemax', 'device_name': str(self.device)}
        self.model = TabNetClassifier(**params)
        self.model.load_model(os.path.join(model_dir, "tabnet_eviction.zip"))
        self.model.network.to(self.device)
        self.model.network.eval()

        # online fine-tuning config
        self.enable_finetune = enable_finetune
        self._ft_interval = finetune_interval  # fine-tune interval
        self._ft_epochs = 3   # epochs for head fine-tune

        # buffers and counter
        self._counter = 0
        self._X_buffer = []  # list of torch.Tensor
        self._y_buffer = []  # list of ints
        self._prev_metrics = None

    def _make_feature_tensor(self, metrics: dict) -> torch.Tensor:
        """
        Build and transform a single-example feature tensor from metrics dict.
        """
        row = {**self.static_features, **metrics}
        df  = pd.DataFrame([row])
        X   = self.pipeline.transform(df)
        xt  = torch.tensor(X, dtype=torch.float32, device=self.device)
        return xt

    def _finetune_head(self):
        """Fine-tune the TabNet head on buffered tensor examples."""
        # assemble batched numpy for fit
        # X_np = torch.stack(self._X_buffer).cpu().numpy()
        # y_np = np.array(self._y_buffer)
        X_np = torch.cat([t.detach().cpu() for t in self._X_buffer], dim=0).numpy().astype(np.float32)  # (N, D)
        y_np = np.asarray(self._y_buffer, dtype=np.int64).ravel()  # (N,)
        self.model.fit(
            X_train=X_np,
            y_train=y_np,
            max_epochs=self._ft_epochs,
            patience=0,
            batch_size=len(y_np)
        )

    def decide_eviction(self, metrics: dict) -> str:
        """
        Perform inference and optionally online head fine-tuning.
        Returns "yes, evict" or "no, do not evict".
        """
        # clean metrics
        metrics.pop("epoch", None)
        metrics.pop("step",  None)

        # feature tensor
        xt = self._make_feature_tensor(metrics)

        # inference via TabNet
        with torch.no_grad():
            X_np = xt.cpu().numpy()
            proba_arr = self.model.predict_proba(X_np)

        # extract positive-class probability
        if proba_arr.ndim > 1 and proba_arr.shape[1] > 1:
            prob = proba_arr[0, 1]
        else:
            prob = proba_arr.flatten()[0]
        decision = "yes, evict" if prob >= 0.5 else "no, do not evict"

        # online head fine-tuning
        if self.enable_finetune:
            print(f"TabNet eviction fine-tuning enabled, counter: {self._counter}/{self._ft_interval}")
            if self._prev_metrics is not None:
                label = compute_eviction_label(self._prev_metrics, metrics)
                self._X_buffer.append(xt)
                self._y_buffer.append(label)
                self._counter += 1
                if self._counter >= self._ft_interval:
                    self._finetune_head()
                    self._counter = 0
                    self._X_buffer.clear()
                    self._y_buffer.clear()

            # update previous metrics
            self._prev_metrics = {
                'Pre_Avg_Hitrate': metrics['Pre_Avg_Hitrate'],
                'Pre_Avg_T_rpc':    metrics['Pre_Avg_T_rpc']
            }
        return decision

class LogisticRegressionEvictionClassifier:
    """
    Eviction agent backed by a trained Logistic Regression model.
    Uses the same static features & preprocessing pipeline as the MLP/TabNet agents.
    """
    def __init__(self,
                 model_dir: str,
                 device: torch.device,
                 dataset: str,
                 rank: int,
                 batch_size: int,
                 num_total_nodes: int,
                 num_partition_nodes: int,
                 num_remote_nodes: int,
                 buffer_size: int,
                 enable_finetune: bool = False,
                 finetune_interval: int = 50):
        self.device = device  # not really used, but kept for API compatibility

        # Static features (identical pattern to MLP / TabNet)
        self.static_features = {
            "Dataset": dataset,
            "Rank": rank,
            "Batch_Size": batch_size,
            "Num_Total_Nodes": num_total_nodes,
            "Num_Partition_Nodes": num_partition_nodes,
            "Num_Remote_Nodes": num_remote_nodes,
            "buffer_size": buffer_size,
        }

        # Load preprocessing pipeline
        pipe_path = os.path.join(model_dir, "eviction_preprocessor.joblib")
        self.pipeline = joblib.load(pipe_path)

        # Load trained Logistic Regression model
        model_path = os.path.join(model_dir, "lr_eviction.joblib")
        self.model = joblib.load(model_path)
        self.model.n_jobs = -1  # just to be safe

        # Fine-tuning flags (unused, but kept to match constructor signature)
        self.enable_finetune = enable_finetune
        self._ft_interval = finetune_interval

    def decide_eviction(self, metrics: dict) -> str:
        # Drop bookkeeping fields if present
        metrics.pop("epoch", None)
        metrics.pop("step",  None)

        # Merge static + dynamic features
        full = {**self.static_features, **metrics}
        df = pd.DataFrame([full])

        # Transform with the same pipeline used at training
        X = self.pipeline.transform(df)

        # Logistic Regression probability for positive class
        proba = self.model.predict_proba(X)[0, 1]
        decision = "yes, evict" if proba >= 0.5 else "no, do not evict"

        print(f"LR eviction probability: {proba:.4f}, decision: {decision}")
        return decision

class RandomForestEvictionClassifier:
    """
    Eviction agent backed by a trained Random Forest classifier.
    """
    def __init__(self,
                 model_dir: str,
                 device: torch.device,
                 dataset: str,
                 rank: int,
                 batch_size: int,
                 num_total_nodes: int,
                 num_partition_nodes: int,
                 num_remote_nodes: int,
                 buffer_size: int,
                 enable_finetune: bool = False,
                 finetune_interval: int = 50):
        self.device = device

        self.static_features = {
            "Dataset": dataset,
            "Rank": rank,
            "Batch_Size": batch_size,
            "Num_Total_Nodes": num_total_nodes,
            "Num_Partition_Nodes": num_partition_nodes,
            "Num_Remote_Nodes": num_remote_nodes,
            "buffer_size": buffer_size,
        }

        pipe_path = os.path.join(model_dir, "eviction_preprocessor.joblib")
        self.pipeline = joblib.load(pipe_path)

        model_path = os.path.join(model_dir, "rf_eviction.joblib")
        self.model = joblib.load(model_path)

        for est in getattr(self.model, "estimators_", []):
            if isinstance(est, DecisionTreeClassifier) and not hasattr(est, "monotonic_cst"):
                est.monotonic_cst = None
                
        self.enable_finetune = enable_finetune
        self._ft_interval = finetune_interval

    def decide_eviction(self, metrics: dict) -> str:
        metrics.pop("epoch", None)
        metrics.pop("step",  None)

        full = {**self.static_features, **metrics}
        df = pd.DataFrame([full])
        X = self.pipeline.transform(df)

        proba = self.model.predict_proba(X)[0, 1]
        decision = "yes, evict" if proba >= 0.5 else "no, do not evict"

        print(f"RF eviction probability: {proba:.4f}, decision: {decision}")
        return decision

class SVMEvictionClassifier:
    """
    Eviction agent backed by a trained linear SVM (LinearSVC).
    Note: LinearSVC does not expose calibrated probabilities; we use the class label directly.
    """
    def __init__(self,
                 model_dir: str,
                 device: torch.device,
                 dataset: str,
                 rank: int,
                 batch_size: int,
                 num_total_nodes: int,
                 num_partition_nodes: int,
                 num_remote_nodes: int,
                 buffer_size: int,
                 enable_finetune: bool = False,
                 finetune_interval: int = 50):
        self.device = device

        self.static_features = {
            "Dataset": dataset,
            "Rank": rank,
            "Batch_Size": batch_size,
            "Num_Total_Nodes": num_total_nodes,
            "Num_Partition_Nodes": num_partition_nodes,
            "Num_Remote_Nodes": num_remote_nodes,
            "buffer_size": buffer_size,
        }

        pipe_path = os.path.join(model_dir, "eviction_preprocessor.joblib")
        self.pipeline = joblib.load(pipe_path)

        model_path = os.path.join(model_dir, "svm_eviction.joblib")
        self.model = joblib.load(model_path)

        self.enable_finetune = enable_finetune
        self._ft_interval = finetune_interval

    def decide_eviction(self, metrics: dict) -> str:
        metrics.pop("epoch", None)
        metrics.pop("step",  None)

        full = {**self.static_features, **metrics}
        df = pd.DataFrame([full])
        X = self.pipeline.transform(df)

        # LinearSVC returns class labels directly
        label = int(self.model.predict(X)[0])
        decision = "yes, evict" if label == 1 else "no, do not evict"

        print(f"LinearSVM eviction label: {label}, decision: {decision}")
        return decision

class XGBoostEvictionClassifier:
    """
    Eviction agent backed by a trained XGBoost classifier.
    """
    def __init__(self,
                 model_dir: str,
                 device: torch.device,
                 dataset: str,
                 rank: int,
                 batch_size: int,
                 num_total_nodes: int,
                 num_partition_nodes: int,
                 num_remote_nodes: int,
                 buffer_size: int,
                 enable_finetune: bool = False,
                 finetune_interval: int = 50):
        self.device = device

        self.static_features = {
            "Dataset": dataset,
            "Rank": rank,
            "Batch_Size": batch_size,
            "Num_Total_Nodes": num_total_nodes,
            "Num_Partition_Nodes": num_partition_nodes,
            "Num_Remote_Nodes": num_remote_nodes,
            "buffer_size": buffer_size,
        }

        pipe_path = os.path.join(model_dir, "eviction_preprocessor.joblib")
        self.pipeline = joblib.load(pipe_path)

        # XGBoost was saved via `clf.save_model("xgb_eviction.json")`
        model_path = os.path.join(model_dir, "xgb_eviction.json")
        self.model = XGBClassifier()
        self.model.load_model(model_path)

        self.enable_finetune = enable_finetune
        self._ft_interval = finetune_interval

    def decide_eviction(self, metrics: dict) -> str:
        metrics.pop("epoch", None)
        metrics.pop("step",  None)

        full = {**self.static_features, **metrics}
        df = pd.DataFrame([full])
        X = self.pipeline.transform(df)

        proba = self.model.predict_proba(X)[0, 1]
        decision = "yes, evict" if proba >= 0.5 else "no, do not evict"

        print(f"XGBoost eviction probability: {proba:.4f}, decision: {decision}")
        return decision
