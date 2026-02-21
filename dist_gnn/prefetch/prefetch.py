import numpy as np
import torch as th
import time
from concurrent.futures import ThreadPoolExecutor
import numba
from .lookup import lookup, update_normal_scores, update_normal_score_of_evicted_nodes
from agents.local_agents import SharedStateStore, MetricsCollectionAgent, ContextAnalysisAgent, DecisionMakingLLMAgent
from agents.non_llm_classifiers import MLPEvictionClassifier, TabNetEvictionClassifier, LogisticRegressionEvictionClassifier, RandomForestEvictionClassifier, XGBoostEvictionClassifier, SVMEvictionClassifier
import queue
import threading

class PrefetchBuffer:
    """
    Shared prefetcher implementation.
    `memory_efficient=False` uses a full-graph normal_score array.
    `memory_efficient=True` uses a halo-only normal_score array.
    """
    def __init__(self, graph, halo_nodes, train_nid, device, args, metadata, ollama_port, local_rank, logdir, memory_efficient=False):
        """Initialize prefetch state, decision backend, workers, and logging."""
        self.args = args
        self.memory_efficient = memory_efficient
        print(
            f"Initializing {'Memory Efficient Prefetcher' if self.memory_efficient else 'PrefetchBuffer'} "
            f"with fraction {args.prefetch_fraction} period {args.eviction_period} alpha {args.alpha}"
        )
        self._setup_graph_context(graph, halo_nodes, train_nid, device)
        self._setup_buffer_state()
        self._setup_eviction_policy_params()
        self._setup_metrics_state()
        self._setup_eviction_backend(metadata, ollama_port, local_rank)
        self._setup_worker()
        self._setup_decision_logging(logdir)
        self._start_workers()

    def _setup_graph_context(self, graph, halo_nodes, train_nid, device):
        """Initialize graph- and partition-related fields."""
        self.graph = graph
        self.train_nid = train_nid
        self.halo_nodes_rank = np.array(list(halo_nodes))
        self.sort_halo_nodes()
        self.device = device
        self.rank = self.graph.rank()
        self.num_layers = self.args.num_layers

    def _setup_buffer_state(self):
        """Initialize prefetch buffer tensors, score arrays, and buffer policy."""
        self.fraction = self.args.prefetch_fraction
        self.buffer_length = 0  # Will be set later
        self.prefetch_ids = np.zeros(self.buffer_length, dtype=np.int32)
        self.prefetch_features = th.zeros(self.buffer_length, self.graph.ndata["features"].shape[1])
        self.eviction_score = None  # O(len(buffer)) space initialized in bulk_prefetch
        if self.memory_efficient:
            self.normal_score = np.zeros(self.halo_nodes_rank.shape[0], dtype=np.float32)
        else:
            self.normal_score = np.zeros(self.graph.number_of_nodes(), dtype=np.float32)
        self.prefetcher_init = self.args.prefetcher_init
        self.fetched_features = th.sparse_coo_tensor(
            (self.graph.number_of_nodes(), self.graph.ndata["features"].shape[1]), dtype=th.float32
        )
        self.num_numba_threads = self.args.num_numba_threads
        self.executor = ThreadPoolExecutor(max_workers=1)

        if self.args.prefetcher_init == "degree":
            print("Degree based prefetch")
            self.degree_based_prefetch()
        elif self.args.prefetcher_init == "empty":
            print("Empty buffer")
            self.empty_buffer()
        elif self.args.prefetcher_init == "random":
            print("Random prefetch")
            self.random_prefetch()
        else:
            ValueError("Invalid prefetcher initialization method")

    def _setup_eviction_policy_params(self):
        """Initialize eviction policy hyperparameters and state flags."""
        self.alpha = self.args.alpha
        self.decay = np.float32(1 - self.alpha)
        self.period = self.args.eviction_period
        self.threshold = round(self.calculate_threshold(), 3)
        self.evict = False
        self.eviction_cutoff = self.args.eviction_cutoff
        self.num_evicted_nodes = 0
        self.evicted_candidates = set()
        self.donotevict_counter = 0  # Counter for "do not evict" decisions
        self.disable_eviction = False  # Flag to disable eviction if needed

    def _setup_metrics_state(self):
        """Initialize runtime counters and metric accumulators."""
        self.counter = 0
        self.rpc_time = 0
        self.prefetch_compute_time = 0
        self.lookup_time = 0
        self.evict_time = 0
        self.update_score_time = 0
        self.agent_decision_wait_time = 0

        self.prefetch_indices_map = None
        self.sorted = False  # Indicates if `prefetch_ids` are sorted
        self.hit = 0
        self.miss = 0
        self.hit_rate_flag = self.args.hit_rate_flag

        self.window_size = self.period
        self.history_size = 5

    def _setup_eviction_backend(self, metadata, ollama_port, local_rank):
        """Select and initialize classifier or LLM decision backend."""
        self.metadata = metadata
        self.metadata["buffer_size"] = self.buffer_length
        classifiers = {
            "mlp": MLPEvictionClassifier,
            "tabnet": TabNetEvictionClassifier,
            "lr": LogisticRegressionEvictionClassifier,
            "rf": RandomForestEvictionClassifier,
            "xgb": XGBoostEvictionClassifier,
            "svm": SVMEvictionClassifier,
        }
        if self.args.decision_model in classifiers:
            classifier_kwargs = self._classifier_kwargs()
            self._create_classifier(classifiers[self.args.decision_model], classifier_kwargs)
        else:
            self._create_llm_agent(ollama_port, local_rank)

    def _classifier_kwargs(self):
        """Build common kwargs for classifier backend constructors."""
        return {
            "model_dir": self.args.ml_model_dir,
            "device": self.device,
            "dataset": self.metadata["dataset"],
            "rank": self.rank,
            "batch_size": self.metadata["minibatch_size"],
            "num_total_nodes": self.metadata["total_nodes"],
            "num_partition_nodes": self.graph.local_partition.number_of_nodes(),
            "num_remote_nodes": self.metadata["num_remote_nodes"],
            "buffer_size": self.metadata["buffer_size"],
            "enable_finetune": self.args.enable_finetune,
            "finetune_interval": self.args.finetune_interval,
        }

    def _create_classifier(self, classifier_cls, classifier_kwargs):
        """Construct the selected classifier backend."""
        self.use_classifier = True
        self.decision_classifier = classifier_cls(**classifier_kwargs)

    def _create_llm_agent(self, ollama_port, local_rank):
        """Construct the LLM decision backend and warm it up."""
        self.use_classifier = False
        self.agent_model_name = self._resolve_llm_model_name()
        self.shared_state_store = SharedStateStore(self.history_size)
        self.metrics_agent = MetricsCollectionAgent(self.shared_state_store, self.window_size)
        self.context_agent = ContextAnalysisAgent(self.metadata, self.history_size)
        self.decision_agent = DecisionMakingLLMAgent(
            self.shared_state_store,
            model_name=self.agent_model_name,
            local_port=ollama_port,
            rank=local_rank,
            context_agent=self.context_agent,
            metadata=self.metadata
        )
        self.decision_agent.warmup()  # Warmup the LLM agent to avoid latency in the first decision

    def _resolve_llm_model_name(self):
        """Resolve configured LLM alias to runtime model identifier."""
        alias_map = {
            "qwen-1.5b": "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:F16",
            "smollm2-360M": "hf.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
            "smollm2-1.7B": "hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
            "qwen2.5-math-1.5B": "hf.co/bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF:F16",
            "granite3.1-moe:3b": "granite3.1-moe:3b-instruct-fp16",
            "granite3.1-moe:1b": "granite3.1-moe:1b-instruct-fp16",
            "mixtral:8x7b": "mixtral:8x7b-instruct-v0.1-q3_K_L",
            "llama4": "hf.co/bartowski/meta-llama_Llama-4-Scout-17B-16E-Instruct-GGUF:Q2_K",
            "mixtral:8x22b": "mixtral:8x22b-instruct-v0.1-q2_K",
        }
        return alias_map.get(self.args.decision_model, self.args.decision_model)

    def _setup_worker(self):
        """Initialize worker queues and synchronization primitives."""
        self.decision = None
        self.eviction_decision_requests = queue.Queue()
        self.eviction_decision_responses = queue.Queue()
        self.pause_lock = threading.Lock()
        self.pause_cond = threading.Condition(self.pause_lock)
        self.pause_worker = False  # When True, worker waits

    def _setup_decision_logging(self, logdir):
        """Open the per-rank decision log file."""
        self.decision_log_file = open(f"{logdir}/{self.args.decision_model}_{self.rank}.txt", "a")

    def _start_workers(self):
        """Start background decision worker thread for current backend."""
        if self.use_classifier:
            self.worker_thread = threading.Thread(target=self.ml_worker, daemon=True)
        else:
            self.worker_thread = threading.Thread(target=self.eviction_decision_worker, daemon=True)
        self.worker_thread.start()
    
    def set_pause_worker(self, should_pause: bool):
        """Pause or unpause the eviction worker thread."""
        with self.pause_lock:
            self.pause_worker = should_pause
            if not should_pause:
                # Signal the worker thread to resume
                self.pause_cond.notify_all()

    def empty_buffer(self):
        """Initialize an empty/sentinel prefetch buffer."""
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
        """Sort halo node IDs for deterministic lookup operations."""
        self.halo_nodes_rank = np.sort(self.halo_nodes_rank)

    def random_prefetch(self):
        """Initialize the buffer with a random subset of halo nodes."""
        # Randomly select a subset of halo nodes to prefetch
        selected_nodes = np.random.choice(self.halo_nodes_rank, int(len(self.halo_nodes_rank) * self.fraction),
                                          replace=False)
        self.buffer_length = len(selected_nodes)
        self.bulk_prefetch(selected_nodes)

    def degree_based_prefetch(self):
        """Initialize the buffer with highest-degree halo nodes."""
        halo_nodes_tensor = th.tensor(self.halo_nodes_rank)
        # Get the top fraction of nodes by degree
        self.buffer_length = int(len(self.halo_nodes_rank) * self.fraction)
        _, top_indices = th.topk(self.graph.in_degrees(halo_nodes_tensor), self.buffer_length)
        self.bulk_prefetch(halo_nodes_tensor[top_indices])

    def bulk_prefetch(self, nodes):
        """Populate buffer IDs/features in bulk from selected nodes."""
        # if nodes in numpy array, copy directly
        if isinstance(nodes, np.ndarray):
            self.prefetch_ids = nodes
        else:
            self.prefetch_ids = nodes.numpy()
        self.prefetch_features = self.graph.ndata["features"][nodes]
        self.eviction_score = np.ones(self.buffer_length, dtype=np.float32)
        self.tag_prefetched_nodes_in_normal_score()

    def tag_prefetched_nodes_in_normal_score(self):
        """Mark currently prefetched nodes in normal-score storage."""
        if self.memory_efficient:
            if self.prefetcher_init == "empty":
                if self.halo_nodes_rank.size == 0:
                    return
                in_bounds_mask = (
                    (self.prefetch_ids >= self.halo_nodes_rank[0]) &
                    (self.prefetch_ids <= self.halo_nodes_rank[-1])
                )
                valid_prefetch_ids = self.prefetch_ids[in_bounds_mask]
                indices = np.searchsorted(self.halo_nodes_rank, valid_prefetch_ids)
                self.normal_score[indices] = -1
            else:
                self.normal_score[np.searchsorted(self.halo_nodes_rank, self.prefetch_ids)] = -1
        else:
            if self.prefetcher_init == "empty": # If the prefetcher is empty, we don't need to tag the nodes
                valid_mask = (self.prefetch_ids >= 0) & (self.prefetch_ids < len(self.normal_score))
                valid_ids = self.prefetch_ids[valid_mask]
                self.normal_score[valid_ids] = -1
            else:
                self.normal_score[self.prefetch_ids] = -1

    def sort_prefetch(self):
        """Sort prefetch IDs and align feature/score arrays accordingly."""
        sort_idx = np.argsort(self.prefetch_ids)
        self.prefetch_ids = self.prefetch_ids[sort_idx]
        self.prefetch_features = self.prefetch_features[sort_idx]
        self.eviction_score = self.eviction_score[sort_idx]
        # turn on the sorted flag
        self.sorted = True

    def calculate_threshold(self):
        """Compute eviction threshold from alpha and eviction period."""
        # calculate the threshold for eviction
        return 1 * (1 - self.alpha) ** self.period

    def calculate_hit_rate(self):
        """Return current buffer hit rate percentage."""
        total = self.hit + self.miss
        if total == 0:
            return 0  
        return round(self.hit / total * 100)

    def calculate_miss_rate(self):
        """Return current buffer miss rate percentage."""
        total = self.hit + self.miss
        if total == 0:
            return 0 
        return round(self.miss / total * 100)

    def update_score(self, missed_minibatch_nodes):
        """Update normal scores for missed nodes in the current minibatch."""
        update_score_start = time.time()
        if self.memory_efficient:
            numba.set_num_threads(self.num_numba_threads - 1) # leave one thread for the main thread
            self.normal_score = update_normal_scores(self.halo_nodes_rank, missed_minibatch_nodes, self.normal_score)
        else:
            mask = np.nonzero(np.in1d(missed_minibatch_nodes, self.halo_nodes_rank, kind='table'))[0]
            self.normal_score[missed_minibatch_nodes[mask]] += 1
        update_score_end = time.time()
        return update_score_end - update_score_start

    def prefetch(self, input_nodes_array, batch_inputs):
        """Serve minibatch from trainer without eviction."""
        start_prefetch_compute = time.time()
        self.counter += 1

        # Sort the prefetch_ids
        sort_start = time.time()
        if self.sorted is False:
            self.sort_prefetch()
        sort_end = time.time()

        # Set number of threads for numba
        numba.set_num_threads(self.num_numba_threads)
        lookup_start = time.time()
        
        # Lookup in the prefetch buffer
        hit_indices, missed_minibatch_idx, feature_indices, self.eviction_score = lookup(input_nodes_array,
                                                                                         self.prefetch_ids,
                                                                                         self.buffer_length,
                                                                                         self.eviction_score,
                                                                                         self.decay)
        lookup_end = time.time()

        copy_features_start = time.time()
        # Copy the features from the prefetch buffer
        batch_inputs[hit_indices] = self.prefetch_features[feature_indices]
        copy_features_end = time.time()

        start_rpc = time.time()
        # RPC for missed minibatch nodes (halo + local nodes)
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
        """Track sampled/found remote-node counts for one minibatch."""
        self.num_remote_nodes_sampled = 0
        self.num_remote_nodes_found = 0
        self.num_remote_nodes_sampled += np.count_nonzero(np.in1d(input_nodes_array, self.halo_nodes_rank, kind='table'))
        self.num_remote_nodes_found = len(hit_indices)

    def prefetch_with_eviction(self, input_nodes_array, batch_inputs, epoch, step):
        """Serve minibatch from trainer with eviction and backend decision loop."""
        start_prefetch_compute = time.time()
        self.counter += 1

        # create a mapping from prefetch_ids to prefetch_features indices
        sort_start = time.time()
        if self.sorted is False:
            self.sort_prefetch()
        sort_end = time.time()

        if self.memory_efficient and self.device != "cpu":
            numba.set_num_threads(30) # TODO: remove hardcoded value in a future cleanup
        else:
            numba.set_num_threads(self.num_numba_threads)
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
                if not self.use_classifier:
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

                # Record a pending eviction event for a no-evict decision.
                if not self.use_classifier:
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
                # Store a pending eviction record with a reason.
                if not self.use_classifier:
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
                if not self.use_classifier:
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
            if not self.use_classifier:
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
        """Background worker that gets classifier decisions from queued stats."""
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
            decision = self.decision_classifier.decide_eviction(stats.copy())
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
            self.decision_log_file.write(log_entry)
            self.decision_log_file.flush() 
            

    def eviction_decision_worker(self):
        """Background worker that queries LLM decisions on aggregated windows."""
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
                
                self.decision_log_file.write(log_entry)
                self.decision_log_file.flush()

                # Now self-pause immediately so no more stats are consumed
                with self.pause_lock:
                    self.pause_worker = True

    def rpc(self, node_idx):
        """Fetch node features from distributed graph storage."""
        features = self.graph.ndata["features"][node_idx]
        return features

    def find_eviction_candidates(self, desired_slots=None):
        """Find buffer slots whose eviction score is below threshold."""
        # select the nodes with eviction score < threshold and return their indices and how many of them are there
        below_threshold_mask = self.eviction_score < self.threshold
        eviction_candidates_idx = np.nonzero(below_threshold_mask)[0]
        slots = np.count_nonzero(below_threshold_mask)

        if slots == 0:
            # print(f"Minibatch {self.counter} | No eviction candidates fell below threshold of {self.threshold}")
            return None, None, 0  # No eviction candidates
        # If a specific desired_slots is given and is less than the current slots, use it instead.
        if desired_slots is not None and desired_slots < slots:
            slots = desired_slots
            eviction_candidates_idx = eviction_candidates_idx[:slots]  # TODO:does this need to be sorted?

        return eviction_candidates_idx, self.eviction_score[eviction_candidates_idx], slots

    def find_replace_candidates(self):
        """Find candidate nodes to insert into buffer based on normal scores."""
        if self.memory_efficient:
            sorted_indices_within_halo = np.argsort(-self.normal_score)
            valid_indices = sorted_indices_within_halo[self.normal_score[sorted_indices_within_halo] > 0]
            return valid_indices, self.normal_score[valid_indices]
        # Step 1: Get the scores of the halo nodes.
        halo_scores = self.normal_score[self.halo_nodes_rank]
        # Step 2: Sort these scores in descending order and get their indices.
        sorted_indices_within_halo = np.argsort(-halo_scores)
        # Filter out the indices corresponding to scores of 0.
        valid_indices = sorted_indices_within_halo[halo_scores[sorted_indices_within_halo] > 0]
        # Map these indices back to the original self.normal_score array.
        replace_candidates = self.halo_nodes_rank[valid_indices]
        return replace_candidates, self.normal_score[replace_candidates]

    def replace_eviction_candidates(self):
        """Replace selected eviction slots with scored replacement candidates."""
        eviction_candidates_idx, eviction_score, max_eviction_slots = self.find_eviction_candidates()

        # If no eviction candidates, exit early.
        if max_eviction_slots == 0:
            return None, None, 0

        if self.memory_efficient:
            replace_candidates_idx, normal_score = self.find_replace_candidates()
            final_slots = min(max_eviction_slots, len(replace_candidates_idx))
            replace_candidates_idx = replace_candidates_idx[:final_slots]
            eviction_candidates_idx = eviction_candidates_idx[:final_slots]
            replace_candidates = self.halo_nodes_rank[replace_candidates_idx]
            eviction_candidates = self.prefetch_ids[eviction_candidates_idx]
            self.prefetch_ids[eviction_candidates_idx] = replace_candidates

            numba.set_num_threads(self.num_numba_threads - 1)  # leave one thread for the main thread
            self.normal_score = update_normal_score_of_evicted_nodes(
                self.halo_nodes_rank,
                eviction_candidates,
                self.normal_score,
                self.eviction_score[eviction_candidates_idx],
            )
            self.tag_prefetched_nodes_in_normal_score()
            # copy normal scores of the replaced nodes to the eviction scores as they are the new prefetch_ids
            self.eviction_score[eviction_candidates_idx] = normal_score[:final_slots]
            return eviction_candidates_idx, self.halo_nodes_rank[replace_candidates_idx[:final_slots]], final_slots

        replace_candidates, normal_score = self.find_replace_candidates()
        final_slots = min(max_eviction_slots, len(replace_candidates))
        # truncate to final_slots
        replace_candidates = replace_candidates[:final_slots]
        eviction_candidates_idx = eviction_candidates_idx[:final_slots]
        eviction_candidates = self.prefetch_ids[eviction_candidates_idx]
        eviction_score = eviction_score[:final_slots]
        normal_score = normal_score[:final_slots]

        self.prefetch_ids[eviction_candidates_idx] = replace_candidates
        if self.prefetcher_init == "empty":
            valid_mask = eviction_candidates < self.graph.number_of_nodes()
            self.normal_score[eviction_candidates[valid_mask]] = eviction_score[valid_mask]
        else:
            self.normal_score[eviction_candidates] = eviction_score
        self.tag_prefetched_nodes_in_normal_score()

        # copy normal scores of the replaced nodes to the eviction scores as they are the new prefetch_ids
        self.eviction_score[eviction_candidates_idx] = normal_score
        return eviction_candidates_idx, replace_candidates, final_slots

    def close(self):
        """Release thread-pool resources used by this prefetcher."""
        if hasattr(self, "decision_log_file") and self.decision_log_file:
            self.decision_log_file.close()
        self.executor.shutdown()
