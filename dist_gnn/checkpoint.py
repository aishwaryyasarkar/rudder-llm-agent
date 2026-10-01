"""Save trained GNN weights without changing training or evaluation behavior."""
import os
import math
from pathlib import Path
import tempfile

import torch


def save_checkpoint(model, path, epoch, model_config, validation_metric=None,
                    best_metric=float("-inf")):
    """All workers call; global rank zero atomically replaces the checkpoint."""
    module = model.module if hasattr(model, 'module') else model
    device = next(module.parameters()).device
    improved = (validation_metric is not None and math.isfinite(validation_metric)
                and validation_metric > best_metric)
    next_best = validation_metric if improved else best_metric
    error = None
    if torch.distributed.get_rank() == 0:
        temporary = None
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            artifact = {
                'format_version': 1,
                'epoch': epoch,
                'model_config': model_config,
                'validation_metric': validation_metric,
                'best_validation_metric': next_best,
                'model_state_dict': {
                    key: value.detach().cpu() for key, value in module.state_dict().items()
                },
            }
            destinations = [path]
            if improved:
                destinations.append(path.with_name('model.best'))
            for destination in destinations:
                fd, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', dir=path.parent)
                os.close(fd)
                torch.save(artifact, temporary)
                os.replace(temporary, destination)
                temporary = None
                if destination.name == 'model.best':
                    print(f'Best model checkpoint saved to: {destination.resolve()}')
        except Exception as exc:
            error = exc
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
    # Use each worker's model device; this supports CPU/Gloo and GPU/NCCL.
    failed = torch.tensor(int(error is not None), device=device)
    torch.distributed.all_reduce(failed)
    if failed.item():
        raise RuntimeError(f'Checkpoint save failed: {error or "see global rank zero log"}') from error
    return next_best


def checkpoint_validation_score(predictions, labels, node_ids, multilabel, device, batch_size):
    """Aggregate validation counts solely for checkpoint selection.

    Existing evaluation return values and reporting remain unchanged.
    """
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
        return float('nan')
    if multilabel:
        tp, fp, fn, _ = counts.tolist()
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 0.0
    return (counts[0] / counts[1]).item()
