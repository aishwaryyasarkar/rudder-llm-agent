"""Portable GNN artifacts. Checkpoints contain tensors and plain metadata only."""
import os
from pathlib import Path
import tempfile

import torch

FORMAT_VERSION = 1


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_model(config):
    from models.graphsage import DistSAGE
    from models.gat import GAT
    common = (config['in_feats'], config['num_hidden'], config['n_classes'],
              config['num_layers'])
    if config['model'] == 'sage':
        return DistSAGE(*common, torch.nn.functional.relu, config['dropout'])
    if config['model'] == 'gat':
        return GAT(*common, config['num_heads'], torch.nn.functional.relu,
                   multilabel=config['is_multilabel'])
    raise ValueError(f"Unsupported model: {config['model']}")


def load_checkpoint(path, in_feats=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint format version')
    config = checkpoint['model_config']
    if in_feats is not None and config['in_feats'] != in_feats:
        raise ValueError(f"Feature dimension mismatch: checkpoint expects {config['in_feats']}, graph has {in_feats}")
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model, checkpoint


def save_training_checkpoint(model, config, graph_metadata, directory, epoch,
                             validation_metric, best_metric, device):
    """All ranks call; only global rank zero writes. Propagate write failures."""
    improved = validation_metric is not None and validation_metric > best_metric
    error = [None]
    if torch.distributed.get_rank() == 0:
        try:
            module = model.module if hasattr(model, 'module') else model
            artifact = {
                'format_version': FORMAT_VERSION,
                'model_state_dict': {k: v.detach().cpu() for k, v in module.state_dict().items()},
                'model_config': config,
                'graph_metadata': graph_metadata,
                'epoch': epoch,
                'validation_metric': validation_metric,
                'best_validation_metric': validation_metric if improved else best_metric,
            }
            atomic_save(artifact, Path(directory) / 'last.pt')
            if improved:
                atomic_save(artifact, Path(directory) / 'best.pt')
        except Exception as exc:
            error[0] = f'Checkpoint write failed: {exc}'
    torch.distributed.broadcast_object_list(error, src=0, device=device)
    if error[0] is not None:
        raise RuntimeError(error[0])
    return validation_metric if improved else best_metric


def metric_counts(scores, labels, multilabel):
    """Sufficient statistics, so global F1 is not an average of rank F1s."""
    if multilabel:
        prediction, target = scores.sigmoid() > 0.5, labels > 0
        return torch.tensor([(prediction & target).sum().item(),
                             (prediction & ~target).sum().item(),
                             (~prediction & target).sum().item()], dtype=torch.float64)
    valid = torch.isfinite(labels) & (labels >= 0)
    return torch.tensor([(scores.argmax(1)[valid] == labels[valid]).sum().item(),
                         valid.sum().item()], dtype=torch.float64)


def metric_from_counts(counts, multilabel):
    if multilabel:
        tp, fp, fn = counts.tolist()
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 0.0
    correct, total = counts.tolist()
    return correct / total if total else float('nan')
