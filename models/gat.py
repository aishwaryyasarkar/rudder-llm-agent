import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import cProfile
import sys, os
import math

class GAT(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, num_heads, activation
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GATConv(
                (in_feats, in_feats),
                n_hidden,
                num_heads=num_heads,
                activation=activation,
            )
        )
        for i in range(1, n_layers - 1):
            self.layers.append(
                dglnn.GATConv(
                    (n_hidden * num_heads, n_hidden * num_heads),
                    n_hidden,
                    num_heads=num_heads,
                    activation=activation,
                )
            )
        self.layers.append(
            dglnn.GATConv(
                (n_hidden * num_heads, n_hidden * num_heads),
                n_classes,
                num_heads=num_heads,
                activation=None,
            )
        )

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h.log_softmax(dim=-1)

    def inference(self, g, x, num_heads, device, batch_size):
        """
        Distributed inference with the GAT model on full neighbors.

        Parameters
        ----------
        g : DistGraph
            The distributed input graph.
        x : DistTensor
            The distributed node feature data of the input graph.
        num_heads : int
            Number of attention heads.
        device : torch.device
            The device tensors will be moved to.

        Returns
        -------
        DistTensor
            The distributed inference results.
        """
        for l, layer in enumerate(self.layers):
            # Initialize a DistTensor to save inference results.
            if l < self.n_layers - 1:
                out_dim = self.n_hidden * num_heads
            else:
                out_dim = self.n_classes

            y = dgl.distributed.DistTensor(
                (g.num_nodes(), out_dim),
                th.float32,
                'y_inference' + str(l),  # Unique name for DistTensor
                persistent=True,
            )
            print(f"|V|={g.num_nodes()}, inference batch size: {batch_size}")
            # Split nodes for distributed processing.
            nodes = dgl.distributed.node_split(
                np.arange(g.num_nodes()),
                g.get_partition_book(),
                force_even=True,
            )

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                if l < self.n_layers - 1:
                    h = layer(block, (h, h_dst)).flatten(1)
                else:
                    # The final layer
                    h = layer(block, (h, h_dst))
                    h = h.mean(1)
                    h = h.log_softmax(dim=-1)

                # Copy back to CPU as DistTensor requires data reside on CPU.
                y[output_nodes] = h.cpu()

            x = y
            # Synchronize trainers
            g.barrier()

        # Return the result on CPU, move to device if necessary outside this function
        return x