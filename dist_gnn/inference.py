"""Distributed batch inference without training or decision-agent dependencies."""
import json
from pathlib import Path

import torch

from dist_gnn.checkpoint import atomic_save, load_checkpoint


def predict(scores, config, threshold=0.5):
    if config['is_multilabel']:
        probabilities = scores.sigmoid()
        return (probabilities > threshold).to(torch.int64), probabilities
    probabilities = scores.exp() if config['model'] == 'gat' else scores.softmax(1)
    prediction = scores.argmax(1)
    mapping = config.get('class_ids')
    if mapping is not None:
        prediction = torch.tensor(mapping, device=prediction.device)[prediction]
    return prediction, probabilities


def check_rank_errors(error, device):
    """Keep peers out of graph barriers if a rank cannot load or write files."""
    failed = torch.tensor(int(error is not None), device=device)
    torch.distributed.all_reduce(failed)
    if failed.item():
        raise RuntimeError(error or "Inference failed on another rank; see its log")


def run_inference(args, graph, device):
    error = None
    try:
        model, artifact = load_checkpoint(args.checkpoint_path, graph.ndata['features'].shape[1])
        config = artifact['model_config']
        expected_graph = artifact['graph_metadata']['graph_name']
        if expected_graph != args.graph_name:
            raise ValueError(f'This checkpoint is for graph {expected_graph}, not {args.graph_name}')
        if artifact['graph_metadata'].get('num_nodes', graph.num_nodes()) != graph.num_nodes():
            raise ValueError('Graph node count differs from the training graph')
        model.to(device).eval()
    except Exception as exc:
        error = f'Checkpoint loading failed: {exc}'
    check_rank_errors(error, device)
    output = Path(args.output_dir)
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    # A fresh directory prevents old shards being mistaken for current predictions.
    error = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            error[0] = f'Cannot create prediction directory: {exc}'
    torch.distributed.broadcast_object_list(error, src=0, device=device)
    if error[0]:
        raise RuntimeError(error[0])
    with torch.no_grad():
        if config['model'] == 'sage':
            scores = model.inference(graph, graph.ndata['features'], args.batch_size_eval, device)
        else:
            scores = model.inference(graph, graph.ndata['features'], config['num_heads'], device,
                                     args.batch_size_eval)
        error = None
        try:
            # Disjoint ranges avoid padded/repeated node IDs and bound export memory.
            start, end = graph.num_nodes() * rank // world, graph.num_nodes() * (rank + 1) // world
            original_ids = 'original_node_id' in graph.ndata
            files = []
            for index, offset in enumerate(range(start, end, args.batch_size_eval)):
                ids = torch.arange(offset, min(offset + args.batch_size_eval, end))
                predictions, probabilities = predict(scores[ids], config, args.prediction_threshold)
                shard = {'node_ids': graph.ndata['original_node_id'][ids] if original_ids else ids,
                         'partition_node_ids': ids, 'predictions': predictions}
                if args.save_scores:
                    shard['probabilities'] = probabilities
                filename = f'rank-{rank:05d}-{index:05d}.pt'
                atomic_save(shard, output / filename)
                files.append(filename)
            (output / f'rank-{rank:05d}.json').write_text(json.dumps({'files': files, 'num_nodes': end - start}, indent=2))
        except Exception as exc:
            error = f'Prediction export failed: {exc}'
        check_rank_errors(error, device)
    torch.distributed.barrier()
    error = None
    try:
        if rank == 0:
            manifest = {'format_version': 1, 'num_nodes': graph.num_nodes(), 'world_size': world,
                        'node_id_space': 'original' if original_ids else 'partition',
                        'checkpoint_path': str(args.checkpoint_path), 'checkpoint_epoch': artifact['epoch'],
                        'model_config': config, 'save_scores': args.save_scores,
                        'prediction_threshold': args.prediction_threshold,
                        'rank_manifests': [f'rank-{r:05d}.json' for r in range(world)]}
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            print(f'Predictions saved to {output}; node IDs use {manifest["node_id_space"]} numbering.')
    except Exception as exc:
        error = f'Manifest write failed: {exc}'
    check_rank_errors(error, device)
    torch.distributed.barrier()
