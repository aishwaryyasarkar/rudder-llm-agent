import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import socket
import dgl
import dgl.distributed
import numpy as np
import torch as th
import utils
from trainer import Trainer
import datetime
from agents import start_ollama
import signal
import traceback

# Global variables for use in the signal handler
global_ollama_proc = None
global_ollama_port = None
CLASSIFIER_MODELS = {"mlp", "tabnet", "lr", "rf", "xgb", "svm"}

def cleanup_and_exit():
    """Ensure the Ollama server is stopped before exiting."""
    global global_ollama_proc, global_ollama_port
    if global_ollama_proc is not None:
        print("\nStopping Ollama server...")
        start_ollama.stop_ollama_server(global_ollama_proc, global_ollama_port)
        print("Ollama server stopped.")
    # sys.exit(1)

def signal_handler(sig, frame):
    """Handle SIGINT and SIGTERM to clean up the Ollama server."""
    print(f"\nReceived termination signal ({sig}). Cleaning up...")
    cleanup_and_exit()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main(args):
    """
    Main function.
    """
    host_name = socket.gethostname()
    print(f"{host_name}: Initializing DistDGL.")
    dgl.distributed.initialize(args.ip_config)
    print(f"{host_name}: Initializing PyTorch process group.")
    th.distributed.init_process_group(backend=args.backend, timeout=datetime.timedelta(seconds=5400))
    local_rank = args.local_rank
    # get pytorch's local rank
    print(f"Local rank: {args.local_rank}")
    utils.set_numa_affinity(local_rank)
    print(f"CPU affinity of process {os.getpid()} rank {local_rank}: {os.sched_getaffinity(0)}")
    print(f"{host_name}: Initializing DistGraph.")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print("Graph Obj g.ndata:", g.ndata)

    # Split train/val/test IDs for each trainer.
    pb = g.get_partition_book()
    print(f"Partition book metadata of {host_name}", pb.metadata())
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=True
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"], pb, force_even=True
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"], pb, force_even=True
        )
    local_nid = pb.partid2nids(pb.partid).detach().numpy() # get local node ids
    num_train_local = len(np.intersect1d(train_nid.numpy(), local_nid)) 
    num_val_local = len(np.intersect1d(val_nid.numpy(), local_nid))
    num_test_local = len(np.intersect1d(test_nid.numpy(), local_nid))
    print(
        f"part {g.rank()}, train: {len(train_nid)} (local: {num_train_local}), "
        f"val: {len(val_nid)} (local: {num_val_local}), "
        f"test: {len(test_nid)} (local: {num_test_local})"
    )

    del local_nid
    if args.num_gpus == 0:
        device = th.device("cpu")
        print("Using CPU.")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
        print(f"Using GPU {dev_id}.")
    n_classes = args.n_classes
    if n_classes == 0:
        if args.graph_name == "yelp":
            # Multi-label: labels shape [N, C] => use C
            n_classes = int(g.ndata["labels"].shape[1])
            print(f"Graph {args.graph_name} has {n_classes} classes.")
        else:
            labels = g.ndata["labels"][np.arange(g.num_nodes())]
            if args.graph_name == "orkut" or args.graph_name == "friendster":
                # Filter out -1 labels (nodes without a community)
                valid_labels = labels[labels >= 0]
                # Get a sorted list of the unique valid labels
                unique_labels = th.unique(valid_labels).tolist()
                # Create a mapping: old community label --> new contiguous label (starting at 0)
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
                # Remap the labels while leaving -1 untouched (so they can be ignored in loss calculation)
                new_labels = labels.clone()
                for old, new in label_mapping.items():
                    new_labels[labels == old] = new
                g.ndata["labels"] = new_labels
                # Update n_classes to match the number of valid communities (remapped labels)
                n_classes = len(label_mapping)
            else:
                n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Pack data.
    in_feats = g.ndata["features"].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g

    logdir = args.summary_filepath.replace(".txt", "")
    os.makedirs(logdir, exist_ok=True) 

    if args.decision_model not in CLASSIFIER_MODELS:
        """ Ollama Server """
        ollama_port = 11434 + local_rank
        global_ollama_port = ollama_port

        # Kill any old Ollama server processes on the specified port
        start_ollama.kill_ollama_servers(ollama_port)

        # Start Ollama server
        ollama_proc = start_ollama.start_ollama_server(
            ollama_port,
            local_rank,
            logdir,
            g.rank(),
            ollama_bin=args.ollama_bin,
            ollama_models_dir=args.ollama_models_dir,
        )

        # Create an error flag tensor (0 means success, 1 means failure).
        error_flag = th.tensor(0)
        if ollama_proc is None:
            print("Ollama server failed to start. Exiting.")
            error_flag.fill_(1)
        # Broadcast the error flag to all processes.
        error_flag = error_flag.to(device)
        th.distributed.broadcast(error_flag, src=0)

        # If any rank encountered an error, perform cleanup and exit.
        if error_flag.item() == 1:
            print("Ollama server failed to start on one or more ranks. Exiting on all processes.")
            cleanup_and_exit()
            sys.exit(1)
            
        global_ollama_proc = ollama_proc
    else:
        ollama_port = None
        global_ollama_port = None
        print("Using classifier decision model, no Ollama server started.")

    try:
        trainer = Trainer(args, device, data, utils.get_halos(g), ollama_port, local_rank, logdir)
        print(f"Rank {g.rank()} Trainer and Prefetcher Initialized.")
        
        # Train and evaluate
        (epoch_time, test_acc, forward_time, backward_time, update_time, sample_time, eval_time, 
        hit_rate, miss_rate, alpha, period, threshold, absolute_total_time,
        prefetch_time) = trainer.run()

        print(
        f"Summary of node classification(GraphSAGE): GraphName "
        f"{args.graph_name} | TrainEpochTime(mean) {epoch_time:.4f} "
        f"| TestAccuracy {test_acc:.4f} | ForwardTime {forward_time:.4f}"
        f"| BackwardTime {backward_time:.4f} | UpdateTime {update_time:.4f}"
        f" | SampleTime {sample_time:.4f} | EvalTime {eval_time:.4f}"
        )

        # calculate the mean epoch time accross all processes
        epoch_time_tensor = utils.calculate_mean(epoch_time, device)
        forward_time_tensor = utils.calculate_mean(forward_time, device)
        backward_time_tensor = utils.calculate_mean(backward_time, device)
        update_time_tensor = utils.calculate_mean(update_time, device)
        sample_time_tensor = utils.calculate_mean(sample_time, device)
        eval_time_tensor = utils.calculate_mean(eval_time, device)
        test_acc_tensor = utils.calculate_mean(test_acc, device) 
        total_epoch_time_tensor = utils.sum(absolute_total_time['epoch_time'], device)

        # Write individual rank's total epoch time to args.summary_filepath
        with open(args.summary_filepath, "a") as f:
            f.write(
                "\n"
                f"Rank {g.rank()} | TotalEpochTime {absolute_total_time['epoch_time']:.4f}s"
                f"| HitRate {hit_rate:.4f} | MissRate {miss_rate:.4f}"
                f"| ForwardTime {absolute_total_time['forward_time']:.4f}s"
                f"| BackwardTime {absolute_total_time['backward_time']:.4f}s"
                f"| UpdateTime {absolute_total_time['update_time']:.4f}s"
                f"| FirstMinibatchSampleTime {absolute_total_time['first_minibatch_sample_time']:.4f}s"
                f"| SampleTime {absolute_total_time['sample_time']:.4f}s"
                f"| WaitForThreadTime {absolute_total_time['wait_for_thread_time']:.4f}s"
                f"| EvalTime {absolute_total_time['eval_time']:.4f}s"
                f"| EpochTime80Percent {absolute_total_time['epoch_time_80_percent']:.4f}s"
                f"| PrefetchComputeTime {prefetch_time['prefetch_compute_time']:.4f}s"
                f"| AgentDecisionWaitTime {prefetch_time['agent_decision_wait_time']:.4f}s"
                f"| EvictionTime {prefetch_time['eviction_time']:.4f}s"
                f"| RPCTime {prefetch_time['rpc_time']:.4f}s"
                "\n"
            )
        # print the final summary
        if th.distributed.get_rank() == 0:
            print("Average training time across processes: {:.4f} seconds".format(epoch_time_tensor))
            print("Average forward time across processes: {:.4f} seconds".format(forward_time_tensor))
            print("Average backward time across processes: {:.4f} seconds".format(backward_time_tensor))
            print("Average update time across processes: {:.4f} seconds".format(update_time_tensor))
            print("Average sample time across processes: {:.4f} seconds".format(sample_time_tensor))
            print("Average eval time across processes: {:.4f} seconds".format(eval_time_tensor))
            print("Average test accuracy across processes: {:.4f}".format(test_acc_tensor))
            
            # write the summary to a args.summary_filepath
            with open(args.summary_filepath, "a") as f:
                f.write(f"alpha: {alpha}, period: {period}, threshold: {threshold}")
                f.write(
                    "\n"
                    "\n"
                    f"Summary of node classification({args.model}): GraphName, prefetch_fraction: {args.prefetch_fraction}, "
                    f"{args.graph_name} | TrainEpochTime(mean) {epoch_time_tensor:.4f} | TotalEpochTime {total_epoch_time_tensor:.4f}"
                    f"| TestAccuracy {test_acc_tensor:.4f} | ForwardTime {forward_time_tensor:.4f}"
                    f"| BackwardTime {backward_time_tensor:.4f} | UpdateTime {update_time_tensor:.4f}"
                    f"| SampleTime+Data_Copy {sample_time_tensor:.4f} | EvalTime {eval_time_tensor:.4f}"
                    "\n"
                )
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()  # Print full stack trace
        cleanup_and_exit()
    finally:
        # Ensure Ollama server is always stopped at the end
        cleanup_and_exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed GraphSAGE.")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--n_classes", type=int, default=0, help="the number of classes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="the number of GPU device. Use 0 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local-rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
        "of batches to be the same.",
    )
    parser.add_argument(
        "--summary_filepath", type=str, help="path to save summary file"
    )
    parser.add_argument(
        "--prefetch_fraction", type=float, default=0.5, help="prefetch fraction"
    )
    parser.add_argument(
        "--prefetcher_init", type=str, default="degree", help="prefetcher init"
    )
    parser.add_argument(
        "--use_memory_efficient_prefetcher",
        type=utils.str2bool,
        default=None,
        help="Override prefetcher choice. If unset, defaults to True for ogbn-papers100M and False otherwise.",
    )
    parser.add_argument(
        "--eviction_period", type=int, default=25, help="eviction period"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="alpha"
    )
    parser.add_argument(
        "--num_numba_threads", type=int, default=1, help="number of numba threads"
    )
    parser.add_argument(
        "--num_trainer_threads", type=int, default=1, help="number of trainer threads"
    )
    parser.add_argument(
        "--hit_rate_flag", type=utils.str2bool, default=False, help="Enable or disable hit rate flag. Accepts: True or False"
    )
    parser.add_argument(
        "--model", type=str, default="sage", help="Model to use for training. Accepts: graphsage or gat"
    )
    parser.add_argument(
        "--decision_model", type=str, default="gemma", help="Decision model for eviction. Use an LLM model name (agent) or one of: mlp/tabnet/lr/rf/xgb/svm (classifier)"
    )
    parser.add_argument(
        "--ml_model_dir", type=str, default="classifier_models", help="Directory for the non-LLM model"
    )
    parser.add_argument(
        "--ollama_bin", type=str, default=None, help="Path to ollama executable (defaults to OLLAMA_BIN env var or 'ollama')"
    )
    parser.add_argument(
        "--ollama_models_dir", type=str, default=None, help="Path to Ollama models directory (defaults to OLLAMA_MODELS env var)"
    )
    parser.add_argument(
        "--enable_finetune", type=utils.str2bool, default=False, help="Enable or disable finetuning of classifier models. Accepts: True or False"
    )

    parser.add_argument("--finetune_interval", type=int, nargs="?", const=50, default=None, help="Interval for finetuning; if provided without a value, defaults to 50.")

    parser.add_argument("--eviction", type=utils.str2bool, default=True, help="Enable or disable eviction. Accepts: True or False")
    parser.add_argument("--num_heads", type=int, default=0, help="Number of attention heads")
    args = parser.parse_args()


    if args.enable_finetune and args.decision_model not in CLASSIFIER_MODELS:
        raise ValueError("Finetuning is only supported for classifier models: mlp/tabnet/lr/rf/xgb/svm.")

    if args.model == "gat":
        assert args.num_heads > 0, "Number of attention heads must be greater than 0"
    
    if args.model not in ["sage", "gat"]:
        raise ValueError("Model not supported. Please choose either graphsage or gat.")
    
    if args.decision_model is None:
        raise ValueError("Decision model must be specified.")
    main(args)
