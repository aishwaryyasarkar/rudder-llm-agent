"""Run with python -m unittest discover -s tests -v (DGL cases skip if absent)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dist_gnn.checkpoint import (atomic_save, build_model, load_checkpoint,
                                 metric_counts, metric_from_counts, save_training_checkpoint)
from dist_gnn.inference import predict, run_inference


def config(model='sage', multilabel=False):
    return dict(model=model, in_feats=3, n_classes=2, num_hidden=4, num_layers=3,
                num_heads=2, dropout=0.2, is_multilabel=multilabel, class_ids=None)


class CheckpointTests(unittest.TestCase):
    def test_atomic_save_and_fresh_process_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            model = torch.nn.Linear(3, 2)
            x = torch.randn(5, 3)
            atomic_save({'weights': model.state_dict(), 'x': x, 'expected': model(x).detach()}, path)
            result = subprocess.run([sys.executable, '-c', '''
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
m = torch.nn.Linear(3, 2)
m.load_state_dict(c['weights'])
torch.testing.assert_close(m(c['x']), c['expected'])
''', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with patch('dist_gnn.checkpoint.torch.save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    atomic_save({}, path)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertIn('weights', torch.load(path, weights_only=True))

    def test_last_best_and_rank_zero(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.broadcast_object_list'):
            model = torch.nn.Linear(3, 2)
            best = float('-inf')
            for epoch, metric in enumerate([None, 0.8, 0.7, None], 1):
                best = save_training_checkpoint(model, config(), {'graph_name': 'tiny'},
                                                directory, epoch, metric, best, torch.device('cpu'))
            self.assertEqual(best, 0.8)
            self.assertEqual(torch.load(Path(directory)/'best.pt', weights_only=True)['epoch'], 2)
            self.assertEqual(torch.load(Path(directory)/'last.pt', weights_only=True)['epoch'], 4)
            with patch('torch.distributed.get_rank', return_value=1), \
                 patch('dist_gnn.checkpoint.atomic_save') as save:
                save_training_checkpoint(model, config(), {}, directory, 5, 0.9, best, torch.device('cpu'))
                save.assert_not_called()
            with patch('dist_gnn.checkpoint.atomic_save', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(RuntimeError, 'disk full'):
                    save_training_checkpoint(model, config(), {}, directory, 5, 0.9, best, torch.device('cpu'))

    def test_reject_incompatible_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.pt'
            atomic_save({'format_version': 99}, path)
            with self.assertRaisesRegex(ValueError, 'version'):
                load_checkpoint(path)
            atomic_save({'format_version': 1, 'model_config': config()}, path)
            with self.assertRaisesRegex(ValueError, 'dimension mismatch'):
                load_checkpoint(path, in_feats=8)

    def test_global_metrics_weight_by_counts(self):
        # Uneven ranks: 1/1 and 0/3 must produce 1/4, not 1/2.
        a = metric_counts(torch.tensor([[3., 0.]]), torch.tensor([0]), False)
        b = metric_counts(torch.tensor([[0., 3.]] * 3), torch.tensor([0, 0, 0]), False)
        self.assertEqual(metric_from_counts(a + b, False), 0.25)
        ignored = metric_counts(torch.tensor([[0., 3.]]), torch.tensor([-1]), False)
        self.assertEqual(ignored.tolist(), [0, 0])
        a = metric_counts(torch.tensor([[3., -3.]]), torch.tensor([[1, 0]]), True)
        b = metric_counts(torch.tensor([[3., 3.]]), torch.tensor([[0, 1]]), True)
        self.assertEqual(metric_from_counts(a + b, True), 0.8)

    def test_predictions_and_class_mapping(self):
        c = config(); c['class_ids'] = [10, 99]
        pred, prob = predict(torch.tensor([[0., 3.]]), c)
        self.assertEqual(pred.tolist(), [99])
        torch.testing.assert_close(prob.sum(1), torch.ones(1))
        c = config('gat')
        _, prob = predict(torch.tensor([[0., 3.]]).log_softmax(1), c)
        torch.testing.assert_close(prob.sum(1), torch.ones(1))
        c = config('gat', True)
        pred, _ = predict(torch.tensor([[3., -3.]]), c)
        self.assertEqual(pred.tolist(), [[1, 0]])

    def test_inference_exports_unlabeled_graph_in_chunks(self):
        class Model(torch.nn.Module):
            def inference(self, graph, features, *args):
                assert not self.training and not torch.is_grad_enabled()
                return torch.tensor([[3., 0.], [0., 3.], [3., 0.]])
        for original in (False, True):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as directory:
                graph = SimpleNamespace(ndata={'features': torch.zeros(3, 3)}, num_nodes=lambda: 3)
                if original:
                    graph.ndata['original_node_id'] = torch.tensor([30, 10, 20])
                args = SimpleNamespace(checkpoint_path='test.pt', graph_name='tiny',
                                       output_dir=str(Path(directory)/'predictions'), batch_size_eval=2,
                                       save_scores=True, prediction_threshold=0.5)
                artifact = {'model_config': config(), 'graph_metadata': {'graph_name': 'tiny'}, 'epoch': 2}
                with patch('dist_gnn.inference.load_checkpoint', return_value=(Model(), artifact)), \
                     patch('torch.distributed.get_rank', return_value=0), \
                     patch('torch.distributed.get_world_size', return_value=1), \
                     patch('torch.distributed.broadcast_object_list'), patch('torch.distributed.barrier'), \
                     patch('torch.distributed.all_reduce'):
                    run_inference(args, graph, torch.device('cpu'))
                    with self.assertRaisesRegex(RuntimeError, 'directory'):
                        run_inference(args, graph, torch.device('cpu'))
                output = Path(args.output_dir)
                manifest = json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['node_id_space'], 'original' if original else 'partition')
                shards = [torch.load(p, weights_only=True) for p in sorted(output.glob('*.pt'))]
                self.assertEqual(len(shards), 2)
                self.assertEqual(torch.cat([s['node_ids'] for s in shards]).tolist(),
                                 [30, 10, 20] if original else [0, 1, 2])
                self.assertEqual(torch.cat([s['predictions'] for s in shards]).tolist(), [0, 1, 0])


@unittest.skipUnless(importlib.util.find_spec('dgl'), 'DGL is not installed')
class GNNRoundTripTests(unittest.TestCase):
    def test_both_models_round_trip_after_training_step(self):
        import dgl
        for name in ('sage', 'gat'):
            for multilabel in (False, True):
                with self.subTest(model=name, multilabel=multilabel), tempfile.TemporaryDirectory() as directory:
                    c = config(name, multilabel)
                    graph = dgl.add_self_loop(dgl.graph(([0, 1, 2], [1, 2, 0])))
                    block = dgl.to_block(graph, torch.arange(3))
                    blocks = [block] * c['num_layers']
                    x = torch.randn(3, 3)
                    model = build_model(c)
                    optimizer = torch.optim.Adam(model.parameters())
                    model(blocks, x).square().mean().backward()
                    optimizer.step()
                    model.eval()
                    expected = model(blocks, x).detach()
                    path = Path(directory)/'gnn.pt'
                    atomic_save({'format_version': 1, 'model_config': c,
                                 'model_state_dict': model.state_dict()}, path)
                    restored, _ = load_checkpoint(path, in_feats=3)
                    restored.eval()
                    torch.testing.assert_close(restored(blocks, x), expected)


if __name__ == '__main__':
    unittest.main()
