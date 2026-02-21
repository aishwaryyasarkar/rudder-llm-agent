import time
import random
from ollama import Client
import requests

#############################################
# Shared State Store
#############################################
class SharedStateStore:
    def __init__(self, history_size=5):
        self.aggregated_metrics = None  # Holds the aggregated summary
        self.history = []               # List to hold completed eviction records
        self.history_size = history_size  # Size of the eviction history

    def update_aggregated_metrics(self, summary):
        self.aggregated_metrics = summary

    def add_history(self, event):
        self.history.append(event)

    def get_history_summary(self):
        # Return the last 5 eviction events (if any)
        return self.history[-self.history_size:] if len(self.history) >= self.history_size else self.history

#############################################
# Metrics Collection Agent
#############################################
class MetricsCollectionAgent:
    def __init__(self, state_store, window_size):
        self.state_store = state_store
        self.window_size = window_size
        self.buffer = []  # Buffer to store individual minibatch records

    def add_metric(self, metric):
        self.buffer.append(metric)
        if len(self.buffer) >= self.window_size:
            summary = self._aggregate_metrics(self.buffer)
            self.reset_buffer()  # Clear the buffer after aggregation
            self.state_store.update_aggregated_metrics(summary)
            return summary
        return None

    def _aggregate_metrics(self, records):
        total_records = len(records)
        aggregated = {}
        # if only 1 record, return it as is
        # if total_records == 1:
        #     return records[0]
        # Use minibatch IDs to represent time.
        for key in records[0].keys():
            if key == "minibatch_id":
                aggregated["minibatch_id"] = [r["minibatch_id"] for r in records]
            elif key == "hitrate":
                avg = sum(r["hitrate"] for r in records) / total_records
                # trend = records[-1]["hitrate"] - records[0]["hitrate"]
                aggregated["hitrate"] = round(avg, 2)
                # aggregated["hitrate_trend"] = round(trend, 2)
            else:
                try:
                    avg = sum(r[key] for r in records) / total_records
                    aggregated[key] = round(avg, 2) if isinstance(avg, float) else avg
                except TypeError:
                    aggregated[key] = records[0][key]
        # Use the last minibatch id as a time proxy.
        aggregated["last_minibatch_id"] = records[-1]["minibatch_id"]
        return aggregated

    def reset_buffer(self):
        """Reset the buffer to clear any stored metrics."""
        self.buffer = []

#############################################
# Context Analysis Agent
#############################################
class ContextAnalysisAgent:
    """
    This agent stores information about eviction events.
    When an eviction decision is made (whether successful or skipped), it stores a record.
    Later, when a new aggregated summary becomes available, it can compute the effect of that eviction
    by comparing current metrics with the pre-eviction metrics.
    """
    def __init__(self, metadata, history_size=5):
        self.pending_eviction = None  # Stores pending event: dict with {"summary": pre_eviction_summary, "minibatch_id": X, "status": "triggered" or "skipped"}
        self.history_size = history_size
        self.eviction_history = []    # List of completed eviction event records
        self.stabilized_count = 0
        self.metadata = metadata

    def store_pending_eviction(self, pre_eviction_summary, num_evicted_nodes, minibatch_id, status="triggered", reason=None):
        """Store pending eviction event with its status. If no candidates were found, status is 'skipped' and reason is provided."""
        self.pending_eviction = {
            "minibatch_id": minibatch_id,
            "pre_summary": pre_eviction_summary,
            "num_evicted_nodes": num_evicted_nodes,
            "status": status,
            "reason": reason  # Only for skipped events
        }

    def update_eviction_effect(self, current_summary):
        """
        If there is a pending eviction event (either triggered or explicitly skipped),
        compute its impact by comparing the current aggregated summary (post-decision)
        to the stored pre-decision summary.
        Then, store a formatted record in history.
        """
        if self.pending_eviction is None:
            return  # Nothing pending to update

        event = self.pending_eviction
        pre_summary = event["pre_summary"]
        results = {}
        # Compute differences for key metrics.
        for key in ["hitrate", "comm_volume", "num_remote_nodes_sampled"]:
            pre_val = pre_summary.get(key)
            curr_val = current_summary.get(key)
            if pre_val is None or curr_val is None:
                continue
            # In some cases pre_val might be zero
            if pre_val == 0:
                results[key] = "N/A"
            else:
                diff = round(curr_val - pre_val, 2)
                pct_change = round((diff / pre_val) * 100, 2)
                # Prepend '+' if positive change; negative values already include '-'
                sign_diff = f"+{diff}" if diff > 0 else f"{diff}"
                sign_pct = f"+{pct_change}%" if pct_change > 0 else f"{pct_change}%"
                results[key] = f"{sign_diff} ({sign_pct})"

        if event["status"] == "triggered":
            record = (f"Minibatch {event['minibatch_id']} | You decided *eviction* | "
                    f"Action taken by System: {event['num_evicted_nodes']} nodes evicted which is "
                    f"{event['num_evicted_nodes'] / self.metadata['buffer_size'] * 100:.2f}% of cache | "
                    f"Impact: Hitrate changed by: {results.get('hitrate', 'N/A')} percent points | "
                    f"comm_volume changed by: {results.get('comm_volume', 'N/A')} | "
                    f"Rate of sampling changed by: {results.get('num_remote_nodes_sampled', 'N/A')}.")
        elif event["status"] == "skipped":
            record = (f"Minibatch {event['minibatch_id']} | You decided *eviction* | "
                    f"Action taken by System: Eviction skipped due to: {event['reason']}.")
        elif event["status"] == "decision_no_evict":
            record = (f"Minibatch {event['minibatch_id']} | You decided *no eviction* | "
                    f"Action taken by System: Eviction skipped | "
                    f"Since then - Hitrate changed by: {results.get('hitrate', 'N/A')} percent points| "
                    f"comm_volume changed by: {results.get('comm_volume', 'N/A')} |"
                    f"Rate of sampling changed by: {results.get('num_remote_nodes_sampled', 'N/A')}")
        elif event["status"] == "invalid_decision":
            record = (f"Minibatch {event['minibatch_id']} | {event['reason']} in a valid JSON with a 'decision' field. | "
                    f"Action taken by System: Eviction skipped | "
                    f"Since then - Hitrate changed by: {results.get('hitrate', 'N/A')} percent points| "
                    f"comm_volume changed by: {results.get('comm_volume', 'N/A')} |"
                    f"Rate of sampling changed by: {results.get('num_remote_nodes_sampled', 'N/A')}")
        else:
            record = "Unknown status"

        self.eviction_history.append(record)
        if len(self.eviction_history) > self.history_size:
            self.eviction_history.pop(0)
        self.clear_pending()


    def clear_pending(self):
        self.pending_eviction = None

    def get_formatted_history(self):
        """Return a formatted string of the eviction history records."""
        if not self.eviction_history:
            return "No past evictions."
        return "\n".join(self.eviction_history)

#############################################
# Decision-making LLM Agent
#############################################
class DecisionMakingLLMAgent:
    def __init__(self, state_store, model_name, local_port, rank, context_agent, metadata):
        self.state_store = state_store
        self.model_name = model_name
        self.context_agent = context_agent
        print(f"Ollama Client in rank {rank} connecting to local Ollama server on port {local_port}...")
        self.client = Client(host=f"http://127.0.0.1:{local_port}")
        # Build static context with graph metadata.
        self.static_context = f"""
You are an LLM agent responsible for deciding whether to **evict** caches in a distributed GNN training pipeline.

**Context**  
- We have a large graph partitioned across machines. To reduce remote node fetches we will use a cache to store remote nodes.
- **Eviction** will remove unused (currently least‐sampled nodes) from the cache and will fetch new ones to fill the cache. 
- The cache starts empty, so early evictions may be necessary. Once filled, the cache always holds a fixed number of nodes (buffer_size). 
- **Evictions are computationally expensive**, so only do them if beneficial.

### **Current Graph & Training Statistics**
- **Total Nodes:** {metadata['total_nodes']}
- **Total Edges:** {metadata['total_edges']}
- **Partition:** {metadata['current_partition']}
- **Minibatch Size:** {metadata['minibatch_size']}
- **Total Minibatches:** {metadata['total_minibatches']}
- **Remote Nodes in Partition:** {metadata['num_remote_nodes']}
- **Cache Capacity:** {metadata['buffer_size']}

### **Key Performance Metrics**
- **Hit Rate (HR):** Percentage of remote nodes sampled that were found in cache (higher is better).
- **Communication Volume (comm_volume):** Volume of remote nodes communication (lower is better).
- **Last Eviction Effect:** Evaluates the impact of past evictions on HR and comm_volume:
    - If HR increased and comm_volume decreased, the eviction was beneficial.
    - If HR & comm_volume did not increase/decrease or skipped due to no candidates, skipping eviction again may be better if comm_volume is low.

**Eviction Rules and Guidelines**  
1. **Must evict** if `HR = 0` (cache is empty and unused).  
2. Evict if HR has deteriorated or still low, or if comm_volume is increasing. However, repeated stagnation in HR may indicate that the cache has reached a steady state and further evictions may not help.
3. **Do NOT evict** if:  
   - The last eviction had **no candidates**. (i.e., prefetcher reported no nodes were eligible).
   - The last 2-3 eviction attempts resulted in 0 evicted candidates or stagnant metrics. (repeated attempts are wasteful)
   - HR is steadily increasing without eviction. (eviction might not be needed). !important
   - Training is nearly done. (few minibatches remaining; e.g., less than 5% of total minibatches left). This is to avoid unnecessary overhead in the final stages of training.

### Metric Naming Clarification in Eviction History
- **Rate of sampling changed** describes how the number of remote nodes sampled by the sampler changed since the last measurement. This change is **not necessarily caused by eviction**, since the sampler is nondeterministic, it evolves based on the graph and minibatch progression. Any pattern you may see in this metric is a result of the sampling strategy and the graph structure.
- Hit Rate (HR) and comm_volume are the primary metrics to consider when deciding on eviction. When the eviction history says "HR changed by +8", it means the hit rate increased by 8 percentage points compared to before eviction (e.g., from 20% to 28%). Positive changes are shown with a + sign, and negative changes with a - in both HR and comm_volume.
- **comm_volume changed: -2000 (-50%)** means 2000 less remote nodes were sampled.

### **Decision Task**
Based on the **latest metrics** and **eviction history**, your task is to determine if an eviction should be triggered for the next minibatch window.

### Response Format (JSON)
Your response **must** be a valid JSON with the following fields:

1. `"decision"`: A string that is **either** `"Yes, evict."` or `"No, do not evict."` **(Required)**
2. `"explanation"`: A **brief** (1-2 sentence) textual explanation of your decision, referencing the current and past HR and `comm_volume` that addresses:
    - **HR changes** (increasing, stagnating, or decreasing?)
    - **comm_volume changes** (improving or worsening?)
    - **Eviction effectiveness** (Did previous evictions help?)  
   **(Required)**
3. `"expected_impact"`: A dictionary predicting how HR and `comm_volume` will change if eviction is (or is not) triggered. **(Required)**
    - `"hitrate"`: One of `"increase"`, `"decrease"`, or `"stagnant"`.
    - `"comm_volume"`: One of `"increase"`, `"decrease"`, or `"stagnant"`.
### Example:
{{
  "decision": "Yes, evict.",
  "explanation": "HR is 0, indicating the cache is empty. Eviction is necessary to populate the cache and improve HR. Previous evictions were effective at increasing HR.",
  "expected_impact": {{
    "hitrate": "increase",
    "comm_volume": "decrease"
  }}
}}
"""


    def decide_eviction(self, current_minibatch_id, retries=3, delay=2):
        current_summary = self.state_store.aggregated_metrics
        # Update pending eviction effect using the current aggregated summary.
        self.context_agent.update_eviction_effect(current_summary)
        history_str = self.context_agent.get_formatted_history()
        dynamic_message = (
            f"### Data Provided ###\n"
            f"- Latest Metrics: {current_summary}\n"
            f"- Eviction History (sorted from oldest to newest):\n{history_str}\n\n"
            f"Based on these inputs, reason briefly and decide whether to evict the cache. Output only the required JSON fields.\n"
        )
        full_message = f"""
Static Context:
{self.static_context}

Dynamic Message:
{dynamic_message}
"""
        start_time = time.time()
        for attempt in range(retries):
            try:
                # Trying the chat API call
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.static_context},
                        {"role": "user", "content": dynamic_message}
                    ]
                )
                break  # If request is successful, break out of the loop
            except requests.exceptions.Timeout:
                print(f"Timeout error during LLM decision-making for minibatch {current_minibatch_id}, attempt {attempt + 1}/{retries}")
            except requests.exceptions.ConnectionError:
                print(f"Connection error during LLM decision-making for minibatch {current_minibatch_id}, attempt {attempt + 1}/{retries}")
            except requests.exceptions.HTTPError as e:
                # Handle 500 Internal Server Error specifically
                if response.status_code == 500:
                    print(f"500 Internal Server Error during LLM decision-making for minibatch {current_minibatch_id}, attempt {attempt + 1}/{retries}")
                else:
                    print(f"HTTP error {response.status_code} during LLM decision-making: {str(e)}, attempt {attempt + 1}/{retries}")
            except requests.exceptions.RequestException as e:
                print(f"Request error during LLM decision-making: {str(e)}, attempt {attempt + 1}/{retries}")
            except Exception as e:
                print(f"Unexpected error during LLM decision-making: {str(e)}, attempt {attempt + 1}/{retries}")
            
            if attempt < retries - 1:  # If not the last attempt, wait before retrying
                wait_time = random.uniform(1, delay)  # Random delay to avoid retry storms
                print(f"Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                return "Error after multiple attempts", full_message, None  # If all retries fail
        
        response_time = time.time() - start_time
        
        # Process the response only if no errors occurred
        return response.message.content.strip(), full_message, response_time

    def warmup(self):
        """Send a trivial request to the LLM so we don't hit long cold-start latency on the first real query."""
        print("Warming up the LLM agent with a small request to reduce cold start latencies...")
        start_time = time.time()
        # Just ask for a short response:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Say a brief hello. This is a warmup to reduce first-query overhead."}
            ]
        )
        end_time = time.time()
        print(f"LLM warmup took {end_time - start_time:.2f} seconds. Warmup response:\n{response.message.content.strip()}")
