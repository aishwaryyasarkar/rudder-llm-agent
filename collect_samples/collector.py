import csv
import os
import multiprocessing
 
class TrainingSampleCollector:
    def __init__(self, eviction_interval, csv_file, rank, graph_name, batch_size, num_total_nodes, num_partition_nodes, num_remote_nodes, fan_out, buffer_size, log):
        """
        Parameters:
          eviction_interval (int): Number of minibatches per phase (pre and post) over which to compute averages.
                                   In Cycle 1, pre-phase gets 'eviction_interval' minibatches and post-phase gets another 
                                   'eviction_interval' minibatches. For subsequent cycles, pre-data is inherited from the 
                                   previous cycle's post-data and new minibatches are recorded as post-phase.
          csv_file (str): Path to the CSV file where aggregated data will be saved.
          rank (int): The rank (or node) for which metrics are being recorded.
          graph_name (str): Graph name.
          batch_size (int): Batch size.
          num_total_nodes (int): Number of nodes in the current partition.
          num_partition_nodes (int): Number of nodes in the current partition.
          num_remote_nodes (int): Number of remote nodes.
          fan_out (int): Fan out.
          buffer_size (int/float): The (unchanging) buffer size for this training run.
        """
        self.eviction_interval = eviction_interval
        self.csv_file = csv_file
        self.rank = rank
        self.graph_name = graph_name
        self.batch_size = batch_size
        self.num_total_nodes = num_total_nodes
        self.num_partition_nodes = num_partition_nodes
        self.num_remote_nodes = num_remote_nodes
        self.fan_out = fan_out
        self.buffer_size = buffer_size
        self.interval_id = 0

        # For Cycle 1, we need to collect both pre and post data.
        # In subsequent cycles, pre_data is inherited from the previous cycle's post_data.
        self.pre_data = []   # List of dicts for pre-eviction metrics.
        self.post_data = []  # List of dicts for post-eviction metrics.
        # For Cycle 1, use a counter to distinguish pre vs. post.
        self.current_cycle_count = 0


        self._initialize_csv()
        if log:
            self.log_init_parameters()

    def log_init_parameters(self):
        """Prints out all initialization parameter values."""
        print("EvictionRoundRecorder Initialized with:")
        print(f"  eviction_interval: {self.eviction_interval}")
        print(f"  csv_file: {self.csv_file}")
        print(f"  rank: {self.rank}")
        print(f"  graph_name: {self.graph_name}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  num_total_nodes: {self.num_total_nodes}")
        print(f"  num_partition_nodes: {self.num_partition_nodes}")
        print(f"  num_remote_nodes: {self.num_remote_nodes}")
        print(f"  fan_out: {self.fan_out}")
        print(f"  buffer_size: {self.buffer_size}")


    def _initialize_csv(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as csvfile:
                fieldnames = [
                    # Static fields
                    "Rank", "Graph_Name", "Batch_Size", "Num_Total_Nodes", "Num_Partition_Nodes", "Num_Remote_Nodes", "Fan_Out", "buffer_size",
                    "Eviction_Interval_ID", "Num_Evicted_Nodes",
                    # Aggregated pre-eviction metrics
                    "Pre_Avg_Hitrate", "Pre_Avg_T_rpc", "Pre_Avg_Node_Freq",
                    # Aggregated post-eviction metrics
                    "Post_Avg_Hitrate", "Post_Avg_T_rpc", "Post_Avg_Node_Freq",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def record(self, hitrate, num_evicted_nodes, T_rpc, pre_candidate_freq, post_candidate_freq):
        """
        Records a minibatch’s metrics, distinguishing between pre/post eviction.
        
        Ensures:
        - `pre_candidate_freq` (next eviction candidates) is recorded properly.
        - `post_candidate_freq` tracks how often evicted nodes are refetched.
        """
        minibatch_record = {
            "hitrate": hitrate,
            "num_evicted_nodes": num_evicted_nodes,
            "T_rpc": T_rpc,
            # "T_ddp": T_ddp,
            "Pre_candidate_freq": pre_candidate_freq,  # Frequency of next eviction candidates
            "Post_candidate_freq": post_candidate_freq  # Frequency of evicted nodes being sampled
        }

        # print(f"Recording minibatch {minibatch_record}")

        if self.interval_id == 0:
            # First eviction cycle (collect both pre and post)
            if self.current_cycle_count < self.eviction_interval:
                self.pre_data.append(minibatch_record)
            else:
                self.post_data.append(minibatch_record)

                #Fix: While recording post-phase, track the new `pre_candidate_freq`
                if len(self.post_data) == 1:
                    self.next_pre_candidate_freq = pre_candidate_freq

            self.current_cycle_count += 1

            if self.current_cycle_count >= 2 * self.eviction_interval:
                self.finalize_interval()
                self.current_cycle_count = 0
        else:
            # All new minibatches after the first eviction are for post-phase
            self.post_data.append(minibatch_record)

            # Track new `pre_candidate_freq` for the next eviction
            if len(self.post_data) == 1:
                self.next_pre_candidate_freq = pre_candidate_freq

            if len(self.post_data) >= self.eviction_interval:
                self.finalize_interval()


    def finalize_interval(self):
        """
        Finalize the current eviction event:
        - Aggregate metrics from the pre and post phases.
        - For candidate-specific metrics (num_evicted_nodes, Pre_candidate_freq, Post_candidate_freq),
            compute them solely for this eviction event.
        - For general metrics (hitrate, T_rpc, T_ddp), chain the post-phase values to be used as the pre-phase for the next event.
        - Reset candidate-specific counters and candidate set.
        """
        # Ensure we have enough pre-eviction data.
        if len(self.pre_data) < self.eviction_interval:
            print("Insufficient pre-eviction data; cannot finalize interval.")
            return

        # --- Aggregate Pre-eviction Metrics ---
        pre_avg_hitrate = sum(d["hitrate"] for d in self.pre_data) / len(self.pre_data)
        # # Use the maximum number of evicted nodes observed in pre_data (candidate-specific)
        # num_evicted_nodes = max(d["num_evicted_nodes"] for d in self.pre_data)
        pre_avg_T_rpc = sum(d["T_rpc"] for d in self.pre_data) / len(self.pre_data)
        # pre_avg_T_ddp = sum(d["T_ddp"] for d in self.pre_data) / len(self.pre_data)
        total_pre_candidate_freq = sum(d.get("Pre_candidate_freq", 0) for d in self.pre_data)
        pre_avg_candidate_freq = total_pre_candidate_freq / len(self.pre_data)
        
        # --- Aggregate Post-eviction Metrics ---
        if len(self.post_data) > 0:
            num_evicted_nodes = max(d["num_evicted_nodes"] for d in self.post_data)
            post_avg_hitrate = sum(d["hitrate"] for d in self.post_data) / len(self.post_data)
            post_avg_T_rpc = sum(d["T_rpc"] for d in self.post_data) / len(self.post_data)
            # post_avg_T_ddp = sum(d["T_ddp"] for d in self.post_data) / len(self.post_data)
            total_post_candidate_freq = sum(d.get("Post_candidate_freq", 0) for d in self.post_data)
            post_avg_candidate_freq = total_post_candidate_freq / len(self.post_data)
        else:
            num_evicted_nodes = 0
            # post_avg_hitrate = post_avg_T_rpc = post_avg_T_ddp = post_avg_candidate_freq = float('nan')
            post_avg_hitrate = post_avg_T_rpc = post_avg_candidate_freq = float('nan')

        if hasattr(self, "next_pre_candidate_freq"):
            pre_avg_candidate_freq = self.next_pre_candidate_freq
        

        # --- Create Aggregated Record for This Eviction Event ---
        record = {
            "Rank": self.rank,
            "Graph_Name": self.graph_name,
            "Batch_Size": self.batch_size,
            "Num_Total_Nodes": self.num_total_nodes,
            "Num_Partition_Nodes": self.num_partition_nodes,
            "Num_Remote_Nodes": self.num_remote_nodes,
            "Fan_Out": self.fan_out,
            "buffer_size": self.buffer_size,
            "Eviction_Interval_ID": self.interval_id,
            "Num_Evicted_Nodes": num_evicted_nodes,
            "Pre_Avg_Hitrate": pre_avg_hitrate,
            "Pre_Avg_T_rpc": pre_avg_T_rpc,
            # "Pre_Avg_T_ddp": pre_avg_T_ddp,
            "Pre_Avg_Node_Freq": pre_avg_candidate_freq,
            "Post_Avg_Hitrate": post_avg_hitrate,
            "Post_Avg_T_rpc": post_avg_T_rpc,
            # "Post_Avg_T_ddp": post_avg_T_ddp,
            "Post_Avg_Node_Freq": post_avg_candidate_freq,
        }
        
        self.save_record(record)
        self.interval_id += 1

        # --- Chain Only General Metrics for Next Event ---
        # Create a new pre_data list by copying only the general metrics from each post_data record.
        new_pre_data = []
        for rec in self.post_data:
            new_rec = {
                "hitrate": rec["hitrate"],
                "T_rpc": rec["T_rpc"],
                # "T_ddp": rec["T_ddp"],
                # Candidate-specific fields are NOT carried over.
            }
            new_pre_data.append(new_rec)
        self.pre_data = new_pre_data
        self.post_data = []


    def save_record(self, record):
        """Append the aggregated record to the CSV file."""
        with open(self.csv_file, mode='a', newline='') as csvfile:
            fieldnames = [
                "Rank", "Graph_Name", "Batch_Size", "Num_Total_Nodes", "Num_Partition_Nodes", "Num_Remote_Nodes", "Fan_Out", "buffer_size",
                "Eviction_Interval_ID", "Num_Evicted_Nodes",
                "Pre_Avg_Hitrate", "Pre_Avg_T_rpc", "Pre_Avg_Node_Freq",
                "Post_Avg_Hitrate", "Post_Avg_T_rpc", "Post_Avg_Node_Freq"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(record)
