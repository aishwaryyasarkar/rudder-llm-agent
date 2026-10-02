"""Save and restore distributed GNN training checkpoints."""
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


FORMAT_VERSION = 2
RUNTIME_FORMAT_VERSION = 2


def _module(model):
    return model.module if hasattr(model, "module") else model


def _atomic_torch_save(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _collective_error(error, device, operation):
    failed = torch.tensor(int(error is not None), device=device)
    torch.distributed.all_reduce(failed)
    if failed.item():
        raise RuntimeError(f"{operation} failed: {error or 'see another rank log'}") from error


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)),
                  numpy_state[2], numpy_state[3], numpy_state[4]),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_model_checkpoint(model, optimizer, path, epoch, model_config,
                          validation_metric=None, best_metric=float("-inf"),
                          save_last=True, runtime_state_saved=False,
                          state_rank=None):
    """Save scheduled latest state and any newly improved best model."""
    module = _module(model)
    device = next(module.parameters()).device
    improved = (validation_metric is not None and math.isfinite(validation_metric)
                and validation_metric > best_metric)
    next_best = validation_metric if improved else best_metric

    if state_rank is None:
        state_rank = torch.distributed.get_rank()
    rng_entries = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(
        rng_entries, {"state_rank": int(state_rank), "state": _rng_state()}
    )
    rng_states = [entry["state"] for entry in rng_entries]
    rng_states_by_rank = {
        entry["state_rank"]: entry["state"] for entry in rng_entries
    }
    error = None
    if torch.distributed.get_rank() == 0:
        try:
            artifact = {
                "format_version": FORMAT_VERSION,
                "epoch": epoch,
                "world_size": torch.distributed.get_world_size(),
                "model_config": model_config,
                "validation_metric": validation_metric,
                "best_validation_metric": next_best,
                "model_state_dict": _cpu_copy(module.state_dict()),
                "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
                "rng_states": rng_states,
                "rng_states_by_rudder_rank": rng_states_by_rank,
            }
            path = Path(path)
            if save_last:
                latest = dict(artifact, checkpoint_kind="last",
                              rudder_state_saved=runtime_state_saved)
                _atomic_torch_save(latest, path)
                print(f"Latest model checkpoint saved to: {path.resolve()}")
            if improved:
                best_path = path.with_name("model.best")
                best = dict(artifact, checkpoint_kind="best", rudder_state_saved=False)
                _atomic_torch_save(best, best_path)
                print(f"Best model checkpoint saved to: {best_path.resolve()}")
        except Exception as exc:
            error = exc
    _collective_error(error, device, "Checkpoint save")
    return next_best


def load_training_checkpoint(model, optimizer, path, model_config, device,
                             state_rank=None):
    """Restore model, optimizer, epoch, best metric, and this rank's RNG."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint {path} is not a resumable version-{FORMAT_VERSION} checkpoint"
        )
    if artifact.get("checkpoint_kind") != "last":
        raise ValueError("Training can only resume from model.last, not model.best")
    if artifact.get("model_config") != model_config:
        raise ValueError("Checkpoint model configuration does not match this training run")
    model.load_state_dict(artifact["model_state_dict"])
    optimizer.load_state_dict(artifact["optimizer_state_dict"])
    _optimizer_to_device(optimizer, device)

    rank = torch.distributed.get_rank()
    saved_world_size = artifact.get("world_size")
    if saved_world_size == torch.distributed.get_world_size():
        states_by_rank = artifact.get("rng_states_by_rudder_rank")
        if states_by_rank is not None and state_rank is not None:
            _restore_rng_state(states_by_rank[int(state_rank)])
        else:
            _restore_rng_state(artifact["rng_states"][rank])
            if rank == 0 and state_rank is not None:
                print(
                    "Warning: this checkpoint predates DGL-rank RNG mapping; "
                    "restored RNG by PyTorch rank"
                )
    elif rank == 0:
        print("Warning: world size changed; checkpoint RNG state was not restored")
    return artifact


def runtime_checkpoint_path(model_checkpoint_path, state_rank):
    return Path(model_checkpoint_path).with_name(f"runtime.rank-{state_rank:05d}.last")


def save_runtime_checkpoint(prefetcher, pending_minibatch, model_checkpoint_path,
                            epoch, device):
    """Save each rank's Rudder state and its already-prefetched minibatch."""
    state_rank = int(prefetcher.rank)
    destination = runtime_checkpoint_path(model_checkpoint_path, state_rank)
    error = None
    try:
        _atomic_torch_save({
            "format_version": RUNTIME_FORMAT_VERSION,
            "epoch": epoch,
            "rank": state_rank,
            "distributed_rank": torch.distributed.get_rank(),
            "world_size": torch.distributed.get_world_size(),
            "runtime_state": prefetcher.runtime_state_dict(),
            "pending_minibatch": pending_minibatch,
        }, destination)
        print(f"DGL rank {state_rank} Rudder checkpoint saved to: {destination.resolve()}")
    except Exception as exc:
        error = exc
    _collective_error(error, device, "Rudder checkpoint save")


def load_runtime_checkpoint(prefetcher, model_checkpoint_path, expected_epoch, device):
    """Restore this rank's Rudder state from the latest sidecar."""
    state_rank = int(prefetcher.rank)
    source = runtime_checkpoint_path(model_checkpoint_path, state_rank)
    error = None
    pending_minibatch = None
    try:
        print(f"DGL rank {state_rank} loading Rudder checkpoint: {source.resolve()}", flush=True)
        artifact = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        error = exc
    _collective_error(error, device, "Rudder checkpoint read")

    error = None
    try:
        needs_mapping = torch.tensor(
            int(artifact.get("rank") != state_rank), device=device
        )
        torch.distributed.all_reduce(
            needs_mapping, op=torch.distributed.ReduceOp.MAX
        )
        if needs_mapping.item():
            # All trainers exchange only the small rank/path descriptors. This
            # migrates old checkpoints without every trainer loading every
            # large runtime sidecar to discover its saved DGL rank.
            descriptors = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(descriptors, {
                "rank": artifact.get("rank"), "path": str(source)
            })
            matches = [item for item in descriptors if item["rank"] == state_rank]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one runtime checkpoint for DGL rank {state_rank}, "
                    f"found {len(matches)}"
                )
            matched_source = Path(matches[0]["path"])
            if matched_source != source:
                source = matched_source
                artifact = torch.load(source, map_location="cpu", weights_only=True)
        if artifact.get("format_version") != RUNTIME_FORMAT_VERSION:
            raise ValueError(f"Unsupported Rudder checkpoint format in {source}")
        if artifact.get("epoch") != expected_epoch:
            raise ValueError(f"Rudder checkpoint epoch does not match model checkpoint: {source}")
        if (artifact.get("rank") != state_rank
                or artifact.get("world_size") != torch.distributed.get_world_size()):
            raise ValueError(f"Rudder checkpoint DGL rank or world size does not match: {source}")
        print(f"DGL rank {state_rank} restoring Rudder buffer state", flush=True)
        prefetcher.load_runtime_state_dict(artifact["runtime_state"])
        pending_minibatch = artifact.get("pending_minibatch")
        if pending_minibatch is None:
            raise ValueError(f"Rudder checkpoint has no pending minibatch: {source}")
        print(f"DGL rank {state_rank} Rudder checkpoint restored from: {source.resolve()}")
    except Exception as exc:
        error = exc
    _collective_error(error, device, "Rudder checkpoint restore")
    return pending_minibatch


def checkpoint_validation_score(predictions, labels, node_ids, multilabel, device, batch_size):
    """Aggregate validation accuracy or micro-F1 for best-model selection."""
    counts = torch.zeros(4 if multilabel else 2, dtype=torch.float64, device=device)
    for offset in range(0, len(node_ids), batch_size):
        ids = node_ids[offset:offset + batch_size]
        scores, target = predictions[ids], labels[ids]
        if multilabel:
            positive = scores.sigmoid() > 0.5
            target = target > 0
            values = [(positive & target).sum().item(),
                      (positive & ~target).sum().item(),
                      (~positive & target).sum().item(), len(ids)]
        else:
            values = [(scores.argmax(1) == target.long()).sum().item(), len(ids)]
        counts += torch.tensor(values, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(counts)
    if counts[-1].item() == 0:
        return float("nan")
    if multilabel:
        tp, fp, fn, _ = counts.tolist()
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 0.0
    return (counts[0] / counts[1]).item()
