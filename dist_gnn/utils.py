import torch as th
import dgl
import os

def get_halos(g): 
    remote_lnid = th.nonzero(g.local_partition.ndata["inner_node"] == False).squeeze()
    print(f"part {g.rank()} Number of remote nodes: {len(remote_lnid)}")
    return g.local_partition.ndata[dgl.NID][remote_lnid].detach().numpy() 

def get_local_nodes(g):
    local_lnid = th.nonzero(g.local_partition.ndata["inner_node"] == True).squeeze()
    print(f"part {g.rank()} Number of local nodes: {len(local_lnid)}")
    return g.local_partition.ndata[dgl.NID][local_lnid].detach().numpy()
    
def calculate_mean(param, device):
    param_tensor = th.tensor(param).to(device)
    th.distributed.all_reduce(param_tensor, op=th.distributed.ReduceOp.SUM)
    param_tensor /= th.distributed.get_world_size()
    return param_tensor

def sum(param, device):
    param_tensor = th.tensor(param).to(device)
    th.distributed.all_reduce(param_tensor, op=th.distributed.ReduceOp.SUM)
    return param_tensor

def percentage(part, whole):
    return round(100 * float(part)/float(whole), 2)

def compute_acc(pred, labels):
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def set_numa_affinity(rank):
    numa_bindings = {
        0: set(range(0, 32)).union(range(128, 160)),  # Ranks 0 covers NUMA nodes 0 and 1
        1: set(range(32, 64)).union(range(160, 192)), # Ranks 1 covers NUMA nodes 2 and 3
        2: set(range(64, 96)).union(range(192, 224)), # Ranks 2 covers NUMA nodes 4 and 5
        3: set(range(96, 128)).union(range(224, 256)),# Ranks 3 covers NUMA nodes 6 and 7
    }
    
    if rank in numa_bindings:
        cpu_ids = numa_bindings[rank]
        os.sched_setaffinity(0, cpu_ids)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
