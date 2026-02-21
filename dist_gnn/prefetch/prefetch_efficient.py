import numpy as np
import torch as th
import time
from concurrent.futures import ThreadPoolExecutor
import numba
from .lookup import lookup, update_normal_scores, update_normal_score_of_evicted_nodes
from agents.local_agents import SharedStateStore, MetricsCollectionAgent, ContextAnalysisAgent, DecisionMakingLLMAgent
from agents.non_llm_agents import MLPEvictionAgent, TabNetEvictionAgent, LogisticRegressionEvictionAgent, RandomForestEvictionAgent, XGBoostEvictionAgent, SVMEvictionAgent
import queue
import threading

class MemoryEfficientPrefetcher:
    """
    This is the memory efficient version of the prefetcher. It uses numba for lookup and score update.
    The normal score is O(len(halo_nodes)) and eviction score is O(len(buffer)) space.
    """
    def __init__(self, graph, halo_nodes, train_nid, device, args, metadata, ollama_port, local_rank, logdir):
        print(f"Initializing Memory Efficient Prefetcher with fraction {args.prefetch_fraction} and eviction period {args.eviction_period}")
        # Graph and Data Parameters
        self.args = args
        self.graph = graph
        self.train_nid = train_nid
        self.halo_nodes_rank = np.array(list(halo_nodes))
        self.sort_halo_nodes()
        self.device = device
        self.rank = self.graph.rank()
        self.num_layers = args.num_layers

        # Prefetch Buffer Parameters
        self.fraction = args.prefetch_fraction
        self.buffer_length = 0
        self.prefetch_ids = np.zeros(self.buffer_length, dtype=np.int32)
        self.prefetch_features = th.zeros(self.buffer_length, self.graph.ndata["features"].shape[1])
        self.eviction_score = None  # O(len(buffer)) space initialized in bulk_prefetch
        self.normal_score = np.zeros(self.halo_nodes_rank.shape[0], dtype=np.float32)
        self.prefetcher_init = args.prefetcher_init

        # Sparse Tensor for Feature Storage
        self.fetched_features = th.sparse_coo_tensor(
            (self.graph.number_of_nodes(), self.graph.ndata["features"].shape[1]), dtype=th.float32)
        
        # Eviction Parameters
        self.alpha = args.alpha
        self.decay = np.float32(1 - self.alpha)
        self.period = args.eviction_period
        self.threshold = round(self.calculate_threshold(), 3)
        self.evict = False
        self.eviction_cutoff = args.eviction_cutoff
        self.num_evicted_nodes = 0
        self.evicted_candidates = set()

        
        # Performance Tracking Parameters
        self.counter = 0
        self.rpc_time = 0
        self.prefetch_compute_time = 0
        self.lookup_time = 0
        self.evict_time = 0
        self.update_score_time = 0
        self.agent_decision_wait_time = 0

        # Flag and State Parameters
        self.prefetch_indices_map = None
        self.sorted = False
        self.hit = 0
        self.miss = 0
        self.hit_rate_flag = args.hit_rate_flag

        # Parallelization Parameters
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.num_numba_threads = args.num_numba_threads

        # Initializing Buffer; This is where buffer length is set
        if args.prefetcher_init == "degree":
            print("Degree based prefetch")
            self.degree_based_prefetch()
        elif args.prefetcher_init == "empty":
            print("Empty buffer")
            self.empty_buffer()
        elif args.prefetcher_init == "random":
            print("Random prefetch")
            self.random_prefetch()
        else:
            ValueError("Invalid prefetcher initialization method")
        
        # Agent Parameters
        self.window_size = self.period 
        self.history_size = 5

        # Graph Metadata for Context Analysis
        self.metadata = metadata
        self.metadata["buffer_size"] = self.buffer_length

        # Initialize Model
        if self.args.agent_model == "mlp":
            self.use_llm = False
            self.decision_agent = MLPEvictionAgent(
                model_dir=self.args.ml_model_dir,
                device=self.device,
                dataset=self.metadata["dataset"],
                rank=self.rank,
                batch_size=self.metadata["minibatch_size"],
                num_total_nodes=self.metadata["total_nodes"], 
                num_partition_nodes=self.graph.local_partition.number_of_nodes(),
                num_remote_nodes=self.metadata["num_remote_nodes"],
                buffer_size=self.metadata["buffer_size"]
            )
        elif self.args.agent_model == "tabnet":
            self.use_llm = False
            self.decision_agent = TabNetEvictionAgent(
                model_dir=self.args.ml_model_dir,
                device=self.device,
                dataset=self.metadata["dataset"],
                rank=self.rank,
                batch_size=self.metadata["minibatch_size"],
                num_total_nodes=self.metadata["total_nodes"], 
                num_partition_nodes=self.graph.local_partition.number_of_nodes(),
                num_remote_nodes=self.metadata["num_remote_nodes"],
                buffer_size=self.metadata["buffer_size"]
            )
        elif self.args.agent_model == "lr":
            self.use_llm = False
            self.decision_agent = LogisticRegressionEvictionAgent(
                model_dir=self.args.ml_model_dir,
                device=self.device,
                dataset=self.metadata["dataset"],
                rank=self.rank,
                batch_size=self.metadata["minibatch_size"],
                num_total_nodes=self.metadata["total_nodes"], 
                num_partition_nodes=self.graph.local_partition.number_of_nodes(),
                num_remote_nodes=self.metadata["num_remote_nodes"],
                buffer_size=self.metadata["buffer_size"],
                enable_finetune=self.args.enable_finetune,
                finetune_interval=self.args.finetune_interval,
            )
        elif self.args.agent_model == "rf":
            self.use_llm = False
            self.decision_agent = RandomForestEvictionAgent(
                model_dir=self.args.ml_model_dir,
                device=self.device,
                dataset=self.metadata["dataset"],
                rank=self.rank,
                batch_size=self.metadata["minibatch_size"],
                num_total_nodes=self.metadata["total_nodes"], 
                num_partition_nodes=self.graph.local_partition.number_of_nodes(),
                num_remote_nodes=self.metadata["num_remote_nodes"],
                buffer_size=self.metadata["buffer_size"],
                enable_finetune=self.args.enable_finetune,
                finetune_interval=self.args.finetune_interval,
            )
        elif self.args.agent_model == "xgb":
            self.use_llm = False
            self.decision_agent = XGBoostEvictionAgent(
                model_dir=self.args.ml_model_dir,
                device=self.device,
                dataset=self.metadata["dataset"],
                rank=self.rank,
                batch_size=self.metadata["minibatch_size"],
                num_total_nodes=self.metadata["total_nodes"], 
                num_partition_nodes=self.graph.local_partition.number_of_nodes(),
                num_remote_nodes=self.metadata["num_remote_nodes"],
                buffer_size=self.metadata["buffer_size"],
                enable_finetune=self.args.enable_finetune,
                finetune_interval=self.args.finetune_interval,
            )
        elif self.args.agent_model == "svm":
            self.use_llm = False
            self.decision_agent = SVMEvictionAgent(
                model_dir=self.args.ml_model_dir,
                device=self.device,
                dataset=self.metadata["dataset"],
                rank=self.rank,
                batch_size=self.metadata["minibatch_size"],
                num_total_nodes=self.metadata["total_nodes"], 
                num_partition_nodes=self.graph.local_partition.number_of_nodes(),
                num_remote_nodes=self.metadata["num_remote_nodes"],
                buffer_size=self.metadata["buffer_size"],
                enable_finetune=self.args.enable_finetune,
                finetune_interval=self.args.finetune_interval,
            )
        else:
            self.use_llm = True
            if self.args.agent_model=="qwen-1.5b":
                self.agent_model_name = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:F16"
            elif self.args.agent_model=="smollm2-360M":
                self.agent_model_name = "hf.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF"
            elif self.args.agent_model=="smollm2-1.7B":
                self.agent_model_name = "hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF"
            elif self.args.agent_model=="qwen2.5-math-1.5B":
                self.agent_model_name = "hf.co/bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF:F16"
            else:
                self.agent_model_name = self.args.agent_model
            
            # SharedState and Agents
            self.shared_state_store = SharedStateStore(self.history_size)
            self.metrics_agent = MetricsCollectionAgent(self.shared_state_store, self.window_size)
            self.context_agent = ContextAnalysisAgent(metadata, self.history_size)
            self.decision_agent = DecisionMakingLLMAgent(
                self.shared_state_store,
                model_name=self.agent_model_name,
                local_port=ollama_port,
                rank=local_rank,
                context_agent=self.context_agent,
                metadata=self.metadata
            )

            # Agent Helpers
            self.decision_agent.warmup() # Warmup the LLM agent to avoid latency in the first decision
        self.decision = None
        self.eviction_decision_requests = queue.Queue()
        self.eviction_decision_responses = queue.Queue()
        self.donotevict_counter = 0  # Counter for "do not evict" decisions
        self.disable_eviction = False  # Flag to disable eviction if needed

        # Thread Safety for Agents
        # Threading lock; this should be initialized before the worker thread starts
        self.pause_lock = threading.Lock()
        self.pause_cond = threading.Condition(self.pause_lock)
        self.pause_worker = False  # When True, worker waits

        # Start the eviction decision worker thread
        if not self.use_llm:
            self.worker_thread = threading.Thread(target=self.ml_worker, daemon=True)
        else:
            self.worker_thread = threading.Thread(target=self.eviction_decision_worker, daemon=True)
        self.worker_thread.start()

        # Logging
        self.llm_file = open(f"{logdir}/{self.args.agent_model}_{self.rank}.txt", "a")

    def set_pause_worker(self, should_pause: bool):
        """Pause or unpause the eviction worker thread."""
        with self.pause_lock:
            self.pause_worker = should_pause
            if not should_pause:
                # Signal the worker thread to resume
                self.pause_cond.notify_all()

    def empty_buffer(self):
        self.buffer_length = int(len(self.halo_nodes_rank) * self.fraction) # Use a sentinel value (e.g., self.graph.number_of_nodes()) that is not a valid node id.
        sentinel_base = self.graph.number_of_nodes() # Create sentinel IDs outside the graph range
        self.prefetch_ids = np.arange(
            sentinel_base, 
            sentinel_base + self.buffer_length, 
            dtype=np.int32
        )
        self.prefetch_features = th.zeros(self.buffer_length, self.graph.ndata["features"].shape[1])
        self.eviction_score = np.ones(self.buffer_length, dtype=np.float32)

    def sort_halo_nodes(self):
        # Sort the halo nodes rank
        self.halo_nodes_rank = np.sort(self.halo_nodes_rank)

    def random_prefetch(self):
        # Randomly select a subset of halo nodes to prefetch
        selected_nodes = np.random.choice(self.halo_nodes_rank, int(len(self.halo_nodes_rank) * self.fraction),
                                          replace=False)
        self.buffer_length = len(selected_nodes)
        self.bulk_prefetch(selected_nodes)

    def degree_based_prefetch(self):
        halo_nodes_tensor = th.tensor(self.halo_nodes_rank)
        # Get the top fraction of nodes by degree
        self.buffer_length = int(len(self.halo_nodes_rank) * self.fraction)
        _, top_indices = th.topk(self.graph.in_degrees(halo_nodes_tensor), self.buffer_length)
        self.bulk_prefetch(halo_nodes_tensor[top_indices])

    def bulk_prefetch(self, nodes):
        # if nodes in numpy array, copy directly
        if isinstance(nodes, np.ndarray):
            self.prefetch_ids = nodes
        else:
            self.prefetch_ids = nodes.numpy()
        self.prefetch_features = self.graph.ndata["features"][nodes]
        self.eviction_score = np.ones(self.buffer_length, dtype=np.float32)
        self.tag_prefetched_nodes_in_normal_score()

    def tag_prefetched_nodes_in_normal_score(self):
        # all prefetch_ids are in halo_nodes_rank
        if self.prefetcher_init == "empty":
            in_bounds_mask = (self.prefetch_ids >= self.halo_nodes_rank[0]) & \
                 (self.prefetch_ids <= self.halo_nodes_rank[-1])
            valid_prefetch_ids = self.prefetch_ids[in_bounds_mask]

            indices = np.searchsorted(self.halo_nodes_rank, valid_prefetch_ids)
            self.normal_score[indices] = -1
        else:
            self.normal_score[np.searchsorted(self.halo_nodes_rank, self.prefetch_ids)] = -1

    def sort_prefetch(self):
        sort_idx = np.argsort(self.prefetch_ids)
        self.prefetch_ids = self.prefetch_ids[sort_idx]
        self.prefetch_features = self.prefetch_features[sort_idx]
        self.eviction_score = self.eviction_score[sort_idx]
        # turn on the sorted flag
        self.sorted = True

    def calculate_threshold(self):
        # calculate the threshold for eviction
        return 1 * (1 - self.alpha) ** self.period

    def calculate_hit_rate(self):
        total = self.hit + self.miss
        if total == 0:
            return 0  # or another default value you'd like to use when there's no data
        return round(self.hit / total * 100)

    def calculate_miss_rate(self):
        total = self.hit + self.miss
        if total == 0:
            return 0  # or another default value you'd like to use when there's no data
        return round(self.miss / total * 100)

    def update_score(self, missed_nodes_in_minibatch):
        numba.set_num_threads(self.num_numba_threads - 1) # leave one thread for the main thread
        update_score_start = time.time()
        self.normal_score = update_normal_scores(self.halo_nodes_rank, missed_nodes_in_minibatch, self.normal_score)
        update_score_end = time.time()
        return update_score_end - update_score_start

    def prefetch(self, input_nodes_array, batch_inputs):
        start_prefetch_compute = time.time()
        self.counter += 1

        # create a mapping from prefetch_ids to prefetch_features indices
        sort_start = time.time()
        if self.sorted is False:
            self.sort_prefetch()
        sort_end = time.time()

        numba.set_num_threads(self.num_numba_threads)
        lookup_start = time.time()
        hit_indices, missed_minibatch_idx, feature_indices, self.eviction_score = lookup(input_nodes_array,
                                                                                         self.prefetch_ids,
                                                                                         self.buffer_length,
                                                                                         self.eviction_score,
                                                                                         self.decay)
        lookup_end = time.time()

        copy_features_start = time.time()
        batch_inputs[hit_indices] = self.prefetch_features[feature_indices]
        copy_features_end = time.time()

        start_rpc = time.time()
        batch_inputs[missed_minibatch_idx] = self.rpc(input_nodes_array[missed_minibatch_idx])
        end_rpc = time.time()

        if self.hit_rate_flag: 
            count_hit_miss_start = time.time()
            self.hit += len(hit_indices)
            self.miss += np.count_nonzero(
                np.in1d(input_nodes_array[missed_minibatch_idx], self.halo_nodes_rank, kind='table'))
            count_hit_miss_end = time.time()

        end_prefetch_compute = time.time()
        total_rpc_time = (end_rpc - start_rpc)
        self.prefetch_compute_time += (end_prefetch_compute - start_prefetch_compute) - total_rpc_time
        self.rpc_time += total_rpc_time
        total_time = end_prefetch_compute - start_prefetch_compute
        return batch_inputs, total_rpc_time

    def track_per_minibatch(self, input_nodes_array, hit_indices, minibatchid):
        self.num_remote_nodes_sampled = 0
        self.num_remote_nodes_found = 0
        self.num_remote_nodes_sampled += np.count_nonzero(np.in1d(input_nodes_array, self.halo_nodes_rank, kind='table'))
        self.num_remote_nodes_found = len(hit_indices)

    def prefetch_with_eviction(self, input_nodes_array, batch_inputs, epoch, step):
        start_prefetch_compute = time.time()
        self.counter += 1

        # create a mapping from prefetch_ids to prefetch_features indices
        sort_start = time.time()
        if self.sorted is False:
            self.sort_prefetch()
        sort_end = time.time()

        if self.device == "cpu":
            numba.set_num_threads(self.num_numba_threads)
        else:
            numba.set_num_threads(30) #TODO: shouldn't be hard coded; add this to script arguments
            
        lookup_start = time.time()
        hit_indices, missed_minibatch_idx, hit_in_buffer, self.eviction_score = lookup(input_nodes_array,
                                                                                       self.prefetch_ids,
                                                                                       self.buffer_length,
                                                                                       self.eviction_score, self.decay)
        lookup_end = time.time()

        copy_features_start = time.time()
        batch_inputs[hit_indices] = self.prefetch_features[hit_in_buffer]
        copy_features_end = time.time()

        if self.hit_rate_flag:
            count_hit_miss_start = time.time()
            self.hit += len(hit_indices)
            self.miss += np.count_nonzero(
                np.in1d(input_nodes_array[missed_minibatch_idx], self.halo_nodes_rank, kind='table'))
            count_hit_miss_end = time.time()

        self.prefetch_compute_time += time.time() - start_prefetch_compute
        total_rpc = evict_start = evict_end = merge_rpc_start = merge_rpc_end = start_evict_rpc = end_evict_rpc = start_none_evict_rpc = end_none_evict_rpc = not_used_in_buffer_start = not_used_in_buffer_end = 0
        future = None

        # Non-blocking: Check if the decision agent has made a decision
        try:
            decision = self.eviction_decision_responses.get_nowait()
            if "yes, evict" in decision.lower() or "yes" in decision.lower():
                self.donotevict_counter = 0
                # Worker is already paused, so we can safely clear the queues
                # clear all queues
                with self.eviction_decision_requests.mutex:
                    self.eviction_decision_requests.queue.clear()
                with self.eviction_decision_responses.mutex:
                    self.eviction_decision_responses.queue.clear()
                self.evict = True
                self.decision = decision
            elif "no, do not evict" in decision.lower() or "no" in decision.lower():
                self.donotevict_counter += 1
                self.evict = False
                self.decision = decision
                # clear all queues
                with self.eviction_decision_requests.mutex:
                    self.eviction_decision_requests.queue.clear()
                with self.eviction_decision_responses.mutex:
                    self.eviction_decision_responses.queue.clear()
                # Record a pending eviction event for a no-evict decision.
                if self.use_llm:
                    self.context_agent.store_pending_eviction(
                        pre_eviction_summary=self.shared_state_store.aggregated_metrics,
                        num_evicted_nodes=0,  # No nodes evicted.
                        minibatch_id=self.counter,
                        status="decision_no_evict",
                        reason="You decided not to evict."
                    )
                self.set_pause_worker(False)  # Unpause the worker thread
            else:
                print(f"Rank {self.rank} Epoch {epoch} Step {step} | Invalid decision: {decision}")
                self.evict = False
                self.decision = None
                # clear all queues
                with self.eviction_decision_requests.mutex:
                    self.eviction_decision_requests.queue.clear()
                with self.eviction_decision_responses.mutex:
                    self.eviction_decision_responses.queue.clear()
                # with self.pause_lock:
                #     self.eviction_decision_requests.queue.clear()
                #     self.eviction_decision_responses.queue.clear()
                # Record a pending eviction event for a no-evict decision.
                if self.use_llm:
                    # Store a pending eviction record with an invalid decision reason.
                    self.context_agent.store_pending_eviction(
                        pre_eviction_summary=self.shared_state_store.aggregated_metrics,
                        num_evicted_nodes=0,  # No nodes evicted.
                        minibatch_id=self.counter,
                        status="invalid_decision",
                        reason="You did not respond with either 'yes, evict' or 'no, do not evict'"
                    )
                self.set_pause_worker(False)  # Unpause the worker thread
        except queue.Empty:
            # No decision available yet, carry on with the old value
            pass

        # if self.donotevict_counter >= 5: # FIXME: you can use self.cutoff instead of 4 later
        #     self.disable_eviction = True
        #     print(f"Rank {self.rank}: Eviction disabled after {self.donotevict_counter} 'do not evict' decisions.")
        #     with self.pause_lock:
        #         self.pause_worker = True # Pause the worker thread
        #     start_normal_rpc = time.time()
        #     batch_inputs[missed_minibatch_idx] = self.rpc(input_nodes_array[missed_minibatch_idx])
        #     total_rpc += time.time() - start_normal_rpc
        #     self.rpc_time += total_rpc
        #     return batch_inputs, total_rpc
        if self.evict:
            evict_start = time.time()
            eviction_candidates_idx, replace_candidates, final_slots = self.replace_eviction_candidates()
            evict_end = time.time()
            if eviction_candidates_idx is not None and final_slots > 0:
                self.evicted_candidates = set(self.prefetch_ids[eviction_candidates_idx])
                self.num_evicted_nodes = len(self.evicted_candidates)
                merge_rpc_start = time.time()
                # Convert to tensor and concatenate
                indices_to_update = th.cat(
                    (th.from_numpy(replace_candidates), th.from_numpy(input_nodes_array[missed_minibatch_idx])))
                # Fetch features
                start_evict_rpc = time.time()
                self.fetched_features = self.rpc(indices_to_update)
                end_evict_rpc = time.time()
                total_rpc += end_evict_rpc - start_evict_rpc
                # Split fetched features back into two groups
                self.prefetch_features[eviction_candidates_idx] = self.fetched_features[:final_slots]
                batch_inputs[missed_minibatch_idx] = self.fetched_features[final_slots:]
                # turn off the sorted flag
                self.sorted = False
                merge_rpc_end = time.time()
                # print(f"Rank {self.rank}: Minibatch {self.counter} | Evicted candidates")
                # self.evict = False
                if self.use_llm:
                    # Store a pending eviction record with a reason.
                    self.context_agent.store_pending_eviction(
                        pre_eviction_summary=self.shared_state_store.aggregated_metrics,
                        num_evicted_nodes=self.num_evicted_nodes,
                        minibatch_id=self.counter,
                        status="triggered",
                        reason=self.decision if self.decision is not None else "Eviction triggered."
                    )
            else:
                future = self.executor.submit(self.update_score, input_nodes_array[missed_minibatch_idx])
                no_evict_rpc_start = time.time()  # when no eviction candidates
                batch_inputs[missed_minibatch_idx] = self.rpc(input_nodes_array[missed_minibatch_idx])
                total_rpc += time.time() - no_evict_rpc_start
                # No eviction candidates found: store a pending eviction record (skipped) with a reason.
                if self.use_llm:
                    # Store a pending eviction record with a reason.
                    self.context_agent.store_pending_eviction(
                        pre_eviction_summary=self.shared_state_store.aggregated_metrics,
                        num_evicted_nodes=0,
                        minibatch_id=self.counter,
                        status="skipped",
                        reason="No eviction candidates found."
                    )
            #  Unpause the worker thread to allow it to process the next stat
            self.set_pause_worker(False)  # Unpause the worker thread
        else:
            future = self.executor.submit(self.update_score, input_nodes_array[missed_minibatch_idx])
            start_normal_rpc = time.time()
            batch_inputs[missed_minibatch_idx] = self.rpc(input_nodes_array[missed_minibatch_idx])
            total_rpc += time.time() - start_normal_rpc

        self.track_per_minibatch(input_nodes_array, hit_indices, self.counter)
        if not self.evict: # if eviction happened, we do not want to send this to the eviction decision worker; this is stale data
            # Requests for eviction decision
            if self.use_llm:
                # Prepare the stats for eviction decision
                stat = {
                    "epoch": int(epoch),
                    "step": int(step),
                    "minibatch_id": self.counter,
                    "remaining_minibatches": self.metadata["total_minibatches"] - self.counter,
                    "hitrate": int(self.calculate_hit_rate()),
                    "comm_volume": self.num_remote_nodes_sampled - self.num_remote_nodes_found
                }
                print(f"Rank {self.rank}, Epoch {epoch}, Step {step}, Minibatch {self.counter} | Stats: {stat} | Requesting Eviction Decision | Communication Volume: {self.num_remote_nodes_sampled - self.num_remote_nodes_found}")
            else:
                stat = {
                    "epoch": int(epoch),
                    "step": int(step),
                    "Eviction_Interval_ID": self.counter,
                    "Num_Evicted_Nodes": self.num_evicted_nodes,
                    "Pre_Avg_Hitrate": self.calculate_hit_rate(),
                    "Pre_Avg_T_rpc": total_rpc
                }
                print(f"Rank {self.rank}, Epoch {epoch}, Step {step}, Minibatch {self.counter} | Stats: {stat} | Requesting Eviction Decision | Communication Volume: {self.num_remote_nodes_sampled - self.num_remote_nodes_found}")
            self.eviction_decision_requests.put(stat)
        else:
            print(f"Rank {self.rank}, Epoch {epoch}, Step {step}, Minibatch {self.counter} | Nodes Evicted: {self.num_evicted_nodes} | Communication Volume: {self.num_remote_nodes_sampled - self.num_remote_nodes_found + self.num_evicted_nodes}")
        
        # if self.evict is true which means just evicted turn it off for the next minibatch
        self.evict = False
        # wait for update_score_thread to finish
        if future is not None:
            wait_start = time.time()
            update_score_time = future.result()
            self.update_score_time += update_score_time
            wait_end = time.time()
            wait_time = wait_end - wait_start
        else:
            wait_time = 0
            update_score_time = 0

        eviction_time = (evict_end - evict_start)
        self.rpc_time += total_rpc
        self.evict_time += eviction_time + (merge_rpc_end - merge_rpc_start) - (end_evict_rpc - start_evict_rpc)
        return batch_inputs, total_rpc

    def ml_worker(self):
        while True:
            # A) Check if we are paused
            with self.pause_lock:
                while self.pause_worker:
                    self.pause_cond.wait()  # Block until unpaused
            # block until a window arrives
            # print(f"Rank {self.rank}: Worker is waiting for stats...")
            stats = self.eviction_decision_requests.get()
            # print(f"Rank {self.rank}: Worker received stats: {stats}")
            # flatten 
            start = time.time()
            decision = self.decision_agent.decide_eviction(stats.copy())
            response_time = time.time() - start
            with self.pause_lock:
                self.pause_worker = True  # Pause the worker thread after processing the stats
            self.eviction_decision_responses.put(decision)
            # Log the decision
            log_entry = ("############################\n\n"
                f"Rank: {self.rank}, Minibatch_ID: {stats['Eviction_Interval_ID']}, "
                f"Full message sent: <user>\n{stats}\n<user>\n"
                f"Decision for minibatches {stats['Eviction_Interval_ID']} (Response Time {round(response_time,2)}s): \n<agent>\n{decision}\n<agent>\n")
            # Write to the log file
            print(f"Rank {self.rank}: Decision made: {decision}, writing to log file...")
            self.llm_file.write(log_entry)
            self.llm_file.flush() 

    def eviction_decision_worker(self):
        # print(f"Rank {self.rank}: Worker thread started!")
        while True:
            # A) Check if we are paused
            with self.pause_lock:
                while self.pause_worker:
                    self.pause_cond.wait()  # Block until unpaused
            # print(f"Rank {self.rank}: Worker is unpaused and waiting for stats...")
            # B) We are unpaused, so fetch the next stats
            stats = self.eviction_decision_requests.get()  # blocking
            summary = self.metrics_agent.add_metric(stats)
            # print(f"Rank {self.rank}: Worker received stats: {stats}")

            if summary is not None:
                # This is your original call:
                decision, msg, response_time = self.decision_agent.decide_eviction(stats["minibatch_id"])
                # c) Post the decision result back
                self.eviction_decision_responses.put(decision)
                epoch = stats["epoch"]
                step = stats["step"]
                log_entry = ("############################\n\n"
                            f"Rank: {self.rank}, Epoch: {epoch}, Step: {step}\n"
                            f"Full message sent: <user>\n{msg}\n<user>\n"
                            f"Decision for minibatches {summary['minibatch_id']} (Response Time {round(response_time,2)}s): \n<agent>\n{decision}\n<agent>\n")
                
                self.llm_file.write(log_entry)
                self.llm_file.flush()

                # Now self-pause immediately so no more stats are consumed
                with self.pause_lock:
                    self.pause_worker = True

    def rpc(self, node_idx):
        features = self.graph.ndata["features"][node_idx]
        return features

    def find_eviction_candidates(self, desired_slots=None):
        # select the nodes with eviction score < threshold and return their indices and how many of them are there
        below_threshold_mask = self.eviction_score < self.threshold
        eviction_candidates_idx = np.nonzero(below_threshold_mask)[0]
        slots = np.count_nonzero(below_threshold_mask)

        if slots == 0:
            return None, None, 0  # No eviction candidates
        # If a specific desired_slots is given and is less than the current slots, use it instead.
        if desired_slots is not None and desired_slots < slots:
            slots = desired_slots
            eviction_candidates_idx = eviction_candidates_idx[:slots]  # TODO:does this need to be sorted?

        return eviction_candidates_idx, self.eviction_score[eviction_candidates_idx], slots

    def find_replace_candidates(self):
        # Sort these scores in descending order and get their indices.
        sorted_indices_within_halo = np.argsort(-self.normal_score)
        # Filter out the indices corresponding to scores of 0.
        valid_indices = sorted_indices_within_halo[self.normal_score[sorted_indices_within_halo] > 0]
        return valid_indices, self.normal_score[valid_indices]

    def replace_eviction_candidates(self):
        eviction_candidates_idx, eviction_score, max_eviction_slots = self.find_eviction_candidates()

        # If no eviction candidates, exit early.
        if max_eviction_slots == 0:
            return None, None, 0

        replace_candidates_idx, normal_score = self.find_replace_candidates()
        final_slots = min(max_eviction_slots, len(replace_candidates_idx))
        replace_candidates_idx = replace_candidates_idx[:final_slots]
        eviction_candidates_idx = eviction_candidates_idx[:final_slots]
        replace_candidates = self.halo_nodes_rank[replace_candidates_idx]
        eviction_candidates = self.prefetch_ids[eviction_candidates_idx]

        self.prefetch_ids[eviction_candidates_idx] = replace_candidates

        numba.set_num_threads(self.num_numba_threads - 1)  # leave one thread for the main thread
        self.normal_score = update_normal_score_of_evicted_nodes(self.halo_nodes_rank, eviction_candidates,
                                                                 self.normal_score,
                                                                 self.eviction_score[eviction_candidates_idx])

        self.tag_prefetched_nodes_in_normal_score()

        # copy normal scores of the replaced nodes to the eviction scores as they are the new prefetch_ids
        self.eviction_score[eviction_candidates_idx] = normal_score[:final_slots]
        return eviction_candidates_idx, self.halo_nodes_rank[replace_candidates_idx[:final_slots]], final_slots

    def close(self):
        self.executor.shutdown()
