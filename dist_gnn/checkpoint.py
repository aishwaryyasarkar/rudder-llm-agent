"""Save trained GNN weights without changing training or evaluation behavior."""
import os
from pathlib import Path
import tempfile

import torch


def save_checkpoint(model, path, epoch, model_config):
    """All workers call; global rank zero atomically replaces the checkpoint."""
    module = model.module if hasattr(model, 'module') else model
    device = next(module.parameters()).device
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
                'model_state_dict': {
                    key: value.detach().cpu() for key, value in module.state_dict().items()
                },
            }
            fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
            os.close(fd)
            torch.save(artifact, temporary)
            os.replace(temporary, path)
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
