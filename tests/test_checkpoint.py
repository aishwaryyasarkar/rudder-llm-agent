"""Run with python -m unittest discover -s tests -v."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dist_gnn.checkpoint import save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_save_reload_and_replace(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.all_reduce'):
            model = torch.nn.Linear(3, 2)
            inputs = torch.randn(5, 3)
            path = Path(directory)/'checkpoints/last.pt'
            config = {'in_feats': 3, 'n_classes': 2}
            save_checkpoint(model, path, 1, config)
            with torch.no_grad():
                model.weight.add_(1)
            save_checkpoint(model, path, 2, config)
            artifact = torch.load(path, map_location='cpu', weights_only=True)
            restored = torch.nn.Linear(3, 2)
            restored.load_state_dict(artifact['model_state_dict'])
            torch.testing.assert_close(restored(inputs), model(inputs))
            self.assertEqual(artifact['epoch'], 2)
            self.assertEqual(artifact['model_config'], config)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_nonzero_rank_does_not_write(self):
        with patch('torch.distributed.get_rank', return_value=1), \
             patch('torch.distributed.all_reduce'), patch('torch.save') as save:
            save_checkpoint(torch.nn.Linear(3, 2), '/unused/last.pt', 1, {})
            save.assert_not_called()

    def test_failed_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.all_reduce'):
            model = torch.nn.Linear(3, 2)
            path = Path(directory)/'last.pt'
            save_checkpoint(model, path, 1, {})
            with patch('torch.save', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(RuntimeError, 'disk full'):
                    save_checkpoint(model, path, 2, {})
            self.assertEqual(torch.load(path, weights_only=True)['epoch'], 1)
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
