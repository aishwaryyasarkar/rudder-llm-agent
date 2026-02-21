# #!/usr/bin/env python3
# import re
# import json
# import sys
# import glob
# import os
# import argparse

# ###############################################################################
# # Helper to parse final HR from the parent's summary .txt file
# ###############################################################################
# def parse_parent_final_hr(subdir_path):
#     """
#     Given a subdir path like:
#       .../ogbn-arxiv_metis_n4_samp0_trainer4_<jobnumber>/
#     go one folder up and look for a file with the same base name plus ".txt".
#     That file contains lines like:
#       Rank 11 | ... HitRate 43.0000 ...
#     Returns a dictionary mapping rank (int) to final hit rate (float).
#     """
#     subdir_name = os.path.basename(os.path.normpath(subdir_path))
#     parent_dir = os.path.dirname(os.path.normpath(subdir_path))
#     final_hr_file = os.path.join(parent_dir, subdir_name + ".txt")
    
#     rank2hr = {}
#     if not os.path.isfile(final_hr_file):
#         print(f"[WARNING] No parent final HR file found at: {final_hr_file}")
#         return rank2hr
    
#     rank_pattern = re.compile(r"Rank\s+(\d+).*?\|\s+HitRate\s+(\d+\.\d+)")
#     with open(final_hr_file, "r") as f:
#         for line in f:
#             match = rank_pattern.search(line)
#             if match:
#                 rank = int(match.group(1))
#                 hr_val = float(match.group(2))
#                 rank2hr[rank] = hr_val
    
#     return rank2hr

# ###############################################################################
# # Helper: interpret a value direction as +1 / -1 / 0
# ###############################################################################
# def direction_to_sign(direction_val):
#     if direction_val is None:
#         return None
#     if isinstance(direction_val, (int, float)):
#         if direction_val > 0:
#             return 1
#         elif direction_val < 0:
#             return -1
#         else:
#             return 0
#     s = str(direction_val).lower().strip()
#     if s in ["increase", "increased", "+", "+1"]:
#         return 1
#     elif s in ["decrease", "decreased", "-", "-1"]:
#         return -1
#     elif s in ["remain", "unchanged", "stable", "same", "0", "stagnant"]:
#         return 0
#     return None

# ###############################################################################
# # parse_decision_json: extracts "expected_impact" and "decision" from the agent's JSON
# ###############################################################################
# def parse_decision_json(agent_text):
#     agent_text = re.sub(r"^(?:json)?", "", agent_text).strip()
#     agent_text = re.sub(r"$", "", agent_text).strip()
#     first_brace = agent_text.find("{")
#     last_brace = agent_text.rfind("}")
#     if first_brace == -1 or last_brace == -1:
#         print("DEBUG: Invalid JSON block detected - skipping record.")
#         return None
#     candidate_json = agent_text[first_brace:last_brace+1]
    
#     pred_hr_dir = None
#     pred_comm_dir = None
#     decision_str = None
    
#     try:
#         data = json.loads(candidate_json)
#     except json.JSONDecodeError:
#         print("DEBUG: JSON decoding failed - skipping record.")
#         return None

#     if "decision" in data:
#         decision_str = data["decision"]
#     if not decision_str or not isinstance(decision_str, str) or decision_str.strip() == "":
#         print("DEBUG: No valid decision found - skipping record.")
#         return None
    
#     if "expected_impact" in data:
#         exp = data["expected_impact"]
#         if isinstance(exp, str):
#             try:
#                 exp = json.loads(exp)
#             except:
#                 pass
#         if isinstance(exp, dict):
#             hr_str = exp.get("hitrate", "")
#             comm_str = exp.get("comm_volume", "")
#             if not hr_str or direction_to_sign(hr_str) is None:
#                 print(f"DEBUG: 'hitrate' field found but not parsed correctly. Found value: '{hr_str}'")
#             pred_hr_dir = direction_to_sign(hr_str)
#             pred_comm_dir = direction_to_sign(comm_str)
    
#     return {
#         "pred_hr_dir": pred_hr_dir,
#         "pred_comm_dir": pred_comm_dir,
#         "decision": decision_str.strip().lower(),
#     }

# ###############################################################################
# # parse_latest_metrics: extracts the HR, comm, and minibatch id from the "Latest Metrics" line
# ###############################################################################
# def parse_latest_metrics(block):
#     """
#     Looks for a line of the form:
#       - Latest Metrics: {... 'hitrate': 34.0, 'comm_volume': 39943.0, 'minibatch_id': [1], ...}
#     Returns a tuple (hr_val, comm_val, minibatch_id) or (None, None, None) if not found.
#     """
#     hr_val = None
#     comm_val = None
#     minibatch_id = None
#     latest_line_match = re.search(r"- Latest Metrics:\s*\{([^}]+)\}", block, re.DOTALL)
#     if latest_line_match:
#         content = latest_line_match.group(1)
#         hr_match = re.search(r"'hitrate':\s*([\d\.]+)", content)
#         if hr_match:
#             try:
#                 hr_val = float(hr_match.group(1))
#             except:
#                 pass
#         comm_match = re.search(r"'comm_volume':\s*([\d\.]+)", content)
#         if comm_match:
#             try:
#                 comm_val = float(comm_match.group(1))
#             except:
#                 pass
#         # Extract minibatch_id assuming it's a list with one number, e.g. [1] or [4]
#         mbatch_match = re.search(r"'minibatch_id':\s*\[([0-9]+)\]", content)
#         if mbatch_match:
#             try:
#                 minibatch_id = int(mbatch_match.group(1))
#             except:
#                 pass
#     return (hr_val, comm_val, minibatch_id)

# ###############################################################################
# # parse_one_record
# ###############################################################################
# def parse_one_record(record):
#     """
#     Extracts:
#       - response_time
#       - predicted hr dir
#       - predicted comm dir
#       - this_hr, this_comm (from "Latest Metrics")
#       - minibatch_id (from Latest Metrics)
#       - decision text
#     Returns a dict with the above keys.
#     If the record doesn't have a recognized decision, returns None.
#     """
#     rt_match = re.search(r"Response Time\s+([\d\.]+)s", record)
#     if not rt_match:
#         return None
#     response_time = float(rt_match.group(1))
    
#     agent_marker = "<agent>"
#     idx = record.find(agent_marker)
#     if idx == -1:
#         return None
    
#     parts = record.split(agent_marker, 1)
#     agent_text = parts[1].strip()
    
#     decision_data = parse_decision_json(agent_text)
#     if decision_data is None:
#         return None
    
#     hr_val, comm_val, minibatch_id = parse_latest_metrics(record)
    
#     rec = {
#         "response_time": response_time,
#         "pred_hr_dir": decision_data["pred_hr_dir"],
#         "pred_comm_dir": decision_data["pred_comm_dir"],
#         "this_hr": hr_val,
#         "this_comm": comm_val,
#         "minibatch_id": minibatch_id,
#         "decision": decision_data["decision"]
#     }
#     print(f"Parsed record: {rec}")
#     return rec

# ###############################################################################
# # sign(x): map a difference to +1 / -1 / 0
# ###############################################################################
# def sign(x):
#     if x is None:
#         return None
#     if x > 0:
#         return 1
#     elif x < 0:
#         return -1
#     else:
#         return 0

# ###############################################################################
# # process_file
# ###############################################################################
# def process_file(filename, rank2finalhr):
#     """
#     1) Split the file by lines of "#####".
#     2) For each chunk, parse the record.
#     3) Compute HR and comm accuracies.
#     4) Compute the average interval between consecutive minibatch IDs.
#     5) Compute the % of valid responses (records that parsed) vs. invalid responses.
#     6) Return the overall stats for this file.
#     """
#     with open(filename, "r") as f:
#         text = f.read()
    
#     # Split by lines of "#####"
#     chunks = re.split(r"^\s*#+\s*$", text, flags=re.MULTILINE)
#     chunks = [c for c in chunks if c.strip()]
#     total_chunks = len(chunks)  # Total responses (valid + invalid)
    
#     parsed = []
#     for c in chunks:
#         rec = parse_one_record(c)
#         if rec is not None:
#             parsed.append(rec)
    
#     valid_count = len(parsed)
#     invalid_count = total_chunks - valid_count
#     valid_response_percent = (valid_count / total_chunks * 100) if total_chunks > 0 else 0
#     invalid_response_percent = (invalid_count / total_chunks * 100) if total_chunks > 0 else 0
    
#     correct_hr = 0
#     correct_comm = 0
#     total_decisions = 0
#     response_times = []
    
#     for i in range(len(parsed) - 1):
#         r_current = parsed[i]
#         r_next = parsed[i+1]
#         p_hr = r_current["pred_hr_dir"]
#         p_comm = r_current["pred_comm_dir"]
    
#         if r_current["this_hr"] is not None and r_next["this_hr"] is not None:
#             hr_diff = r_next["this_hr"] - r_current["this_hr"]
#             a_hr = sign(hr_diff)
#         else:
#             a_hr = None
    
#         if r_current["this_comm"] is not None and r_next["this_comm"] is not None:
#             comm_diff = r_next["this_comm"] - r_current["this_comm"]
#             a_comm = sign(comm_diff)
#         else:
#             a_comm = None
    
#         total_decisions += 1
#         if p_hr is not None and a_hr is not None and p_hr == a_hr:
#             correct_hr += 1
#         if p_comm is not None and a_comm is not None and p_comm == a_comm:
#             correct_comm += 1
    
#         response_times.append(r_current["response_time"])
    
#     rec_count = total_decisions
#     hr_acc = (correct_hr / rec_count * 100) if rec_count else 0
#     comm_acc = (correct_comm / rec_count * 100) if rec_count else 0
#     avg_rt = sum(response_times) / len(response_times) if response_times else 0
    
#     # Compute average minibatch interval from consecutive records
#     minibatch_ids = [r.get("minibatch_id") for r in parsed if r.get("minibatch_id") is not None]
#     diff_list = []
#     for i in range(1, len(minibatch_ids)):
#         diff_list.append(minibatch_ids[i] - minibatch_ids[i-1])
#     avg_minibatch_interval = sum(diff_list) / len(diff_list) if diff_list else None
    
#     rank_match = re.search(r"\bRank:\s*(\d+)", text)
#     rank = int(rank_match.group(1)) if rank_match else None
#     final_hr = None
#     if rank is not None and rank in rank2finalhr:
#         final_hr = rank2finalhr[rank]
    
#     yes_evict = sum(1 for r in parsed if r["decision"] == "yes, evict.")
#     no_evict = sum(1 for r in parsed if r["decision"] == "no, do not evict.")
    
#     return {
#         "filename": os.path.basename(filename),
#         "records": rec_count,
#         "avg_response_time": avg_rt,
#         "hr_accuracy": hr_acc,
#         "comm_accuracy": comm_acc,
#         "total_response_time": sum(response_times),
#         "rank": rank,
#         "final_hr": final_hr,
#         "yes_evict_count": yes_evict,
#         "no_evict_count": no_evict,
#         "avg_minibatch_interval": avg_minibatch_interval,  # New field
#         "valid_response_percent": valid_response_percent,  # New field
#         "invalid_response_percent": invalid_response_percent   # New field
#     }

# ###############################################################################
# # Evaluate a directory
# ###############################################################################
# def evaluate_and_return_stats(subdir):
#     """
#     Processes all .txt files in the given subdir and aggregates their results.
#     """
#     rank2hr = parse_parent_final_hr(subdir)
#     pattern = os.path.join(subdir, "*.txt")
#     files = glob.glob(pattern)
#     if not files:
#         print("No .txt files found in subdir:", subdir)
#         sys.exit(1)
    
#     summaries = []
#     overall_records = 0
#     overall_rt_sum = 0
#     overall_correct_hr = 0
#     overall_correct_comm = 0
#     total_yes_evict = 0
#     total_no_evict = 0
#     overall_total_chunks = 0
#     overall_valid_count = 0
    
#     # Updated header now includes AvgMBInt and Vld/Inv(%)
#     table_lines = []
#     header = "{:<30s} {:>5s} {:>8s} {:>5s} {:>5s} {:>12s} {:>12s} {:>10s} {:>12s} {:>12s}".format(
#         "Filename", "Rank", "Records", "Yes", "No", "AvgRT(s)", "HRAcc(%)", "FinalHR", "AvgMBInt", "Vld/Inv(%)"
#     )
#     table_lines.append(header)
#     table_lines.append("-" * 120)
    
#     for file in sorted(files):
#         summary = process_file(file, rank2hr)
#         summaries.append(summary)
#         overall_records += summary["records"]
#         overall_rt_sum += summary["total_response_time"]
#         overall_correct_hr += summary["hr_accuracy"] * summary["records"] / 100
#         overall_correct_comm += summary["comm_accuracy"] * summary["records"] / 100
#         total_yes_evict += summary["yes_evict_count"]
#         total_no_evict += summary["no_evict_count"]
#         overall_total_chunks += summary.get("records", 0) + int(round((100 - summary.get("valid_response_percent", 0)) / 100 * summary.get("records", 0)))
#         overall_valid_count += summary.get("valid_response_percent", 0)  # Not a simple sum; overall ratio is computed below
        
#         avg_mb_interval_str = f"{summary['avg_minibatch_interval']:.2f}" if summary["avg_minibatch_interval"] is not None else "-"
#         valid_inv_str = f"{summary['valid_response_percent']:.1f}/{summary['invalid_response_percent']:.1f}" if summary["valid_response_percent"] is not None else "-"
#         table_lines.append("{:<30s} {:>5s} {:>8d} {:>5d} {:>5d} {:>12.2f} {:>12.2f} {:>10s} {:>12s} {:>12s}".format(
#             summary["filename"],
#             str(summary["rank"]) if summary["rank"] is not None else "-",
#             summary["records"],
#             summary["yes_evict_count"],
#             summary["no_evict_count"],
#             summary["avg_response_time"],
#             summary["hr_accuracy"],
#             f"{summary['final_hr']:.2f}" if summary["final_hr"] is not None else "-",
#             avg_mb_interval_str,
#             valid_inv_str
#         ))
    
#     overall_avg_rt = overall_rt_sum / overall_records if overall_records else 0
#     overall_hr_acc = (overall_correct_hr / overall_records * 100) if overall_records else 0
#     overall_comm_acc = (overall_correct_comm / overall_records * 100) if overall_records else 0
    
#     # Aggregate overall valid/invalid percentages across files:
#     total_valid_percent = sum(s["valid_response_percent"] for s in summaries) / len(summaries) if summaries else 0
#     total_invalid_percent = sum(s["invalid_response_percent"] for s in summaries) / len(summaries) if summaries else 0
    
#     overall_stats = {
#         "overall_records": overall_records,
#         "total_yes_evict": total_yes_evict,
#         "total_no_evict": total_no_evict,
#         "overall_avg_rt": overall_avg_rt,
#         "overall_hr_acc": overall_hr_acc,
#         "overall_comm_acc": overall_comm_acc,
#         "overall_valid_percent": total_valid_percent,      # New overall stat
#         "overall_invalid_percent": total_invalid_percent     # New overall stat
#     }
    
#     return {
#         "table": "\n".join(table_lines),
#         "summaries": summaries,
#         "overall_stats": overall_stats
#     }

# ###############################################################################
# # Main
# ###############################################################################
# def main():
#     parser = argparse.ArgumentParser(
#         description="Evaluate HR and Communication Volume Prediction Accuracy across multiple runs, "
#                     "using next-record metrics as ground truth for each decision."
#     )
#     parser.add_argument("--node-config", required=True, nargs='+', metavar="NODE_CONFIG",
#                         help="Node config filter substring(s) (e.g. 'n4' 'n8' 'n16' 'n32').")
#     parser.add_argument("--buffer-size", required=True, nargs='+', metavar="BUFFER_SIZE",
#                         help="Buffer size filter substring(s) (e.g. '0.05' '0.25').")
#     parser.add_argument("--datasets", required=True,
#                         help="Comma-separated list of dataset names to include (e.g. 'papers100M,arxiv').")
#     parser.add_argument("--agent_models", required=True, nargs='+',
#                         help="List of agent models to include.")
#     parser.add_argument("--agent_dir", required=True,
#                         help="Base directory containing agent model subdirectories.")
#     args = parser.parse_args()
    
#     datasets = [ds.strip() for ds in args.datasets.split(',')]
    
#     all_summaries = []
#     run_info_list = []
#     buffer_groups = {buf: [] for buf in args.buffer_size}
    
#     for agent_model_dir in os.listdir(args.agent_dir):
#         full_agent_model_dir = os.path.join(args.agent_dir, agent_model_dir)
#         # Only process directories containing '-nocutoff' and skip those with '+contextwindow'
#         if "+contextwindow" in agent_model_dir.lower():
#             continue
#         print(f"Processing agent model directory: {full_agent_model_dir}")
#         if not os.path.isdir(full_agent_model_dir):
#             continue
#         if not any(model_str.lower() in agent_model_dir.lower() for model_str in args.agent_models):
#             continue
    
#         for dataset_dir in os.listdir(full_agent_model_dir):
#             full_dataset_dir = os.path.join(full_agent_model_dir, dataset_dir)
#             if not os.path.isdir(full_dataset_dir):
#                 continue
#             if not any(ds.lower() in dataset_dir.lower() for ds in datasets):
#                 continue
    
#             for root, dirs, files in os.walk(full_dataset_dir):
#                 print(f"Processing directory: {root}")
#                 if files:
#                     txt_files = [f for f in files if f.endswith(".txt")]
                    
#                     if txt_files:
#                         run_dir = root
#                         parent_dir = os.path.dirname(run_dir)
#                         summary_file = os.path.join(parent_dir, os.path.basename(run_dir) + ".txt")
#                         if not os.path.isfile(summary_file):
#                             continue
#                         if not any(nc in run_dir for nc in args.node_config):
#                             continue
    
#                         m = re.search(r"/pf_(\d+\.\d+)/", run_dir)
#                         if not m:
#                             continue
#                         extracted_buffer = m.group(1)
#                         if extracted_buffer not in args.buffer_size:
#                             continue
    
#                         run_results = evaluate_and_return_stats(run_dir)
#                         for summary in run_results["summaries"]:
#                             summary["agent_model"] = agent_model_dir
#                             summary["dataset"] = dataset_dir
#                             summary["run_dir"] = os.path.basename(run_dir)
#                             all_summaries.append(summary)
    
#                         run_info = {
#                             "agent_model": agent_model_dir,
#                             "dataset": dataset_dir,
#                             "run_dir": os.path.basename(run_dir),
#                             "buffer": extracted_buffer,
#                             "records": run_results["overall_stats"]["overall_records"],
#                             "avg_response_time": run_results["overall_stats"]["overall_avg_rt"],
#                             "hr_accuracy": run_results["overall_stats"]["overall_hr_acc"],
#                             "comm_accuracy": run_results["overall_stats"]["overall_comm_acc"],
#                         }
#                         run_info_list.append(run_info)
#                         buffer_groups[extracted_buffer].append(run_results["overall_stats"])
    
#     header = "{:<20s} {:<15s} {:<15s} {:<15s} {:>8s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
#         "Agent Model", "Dataset", "Run Dir", "Buffer Size",
#         "Records", "AvgRT(s)", "HRAcc(%)", "CommAcc(%)", "AvgMBInt", "Vld/Inv(%)"
#     )
#     print(header)
#     print("-" * len(header))
#     for run_info in run_info_list:
#         print("{:<20s} {:<15s} {:<15s} {:<15s} {:>8d} {:>12.2f} {:>12.2f} {:>12.2f} {:>12s} {:>12s}".format(
#             run_info["agent_model"],
#             run_info["dataset"],
#             run_info["run_dir"],
#             run_info["buffer"],
#             run_info["records"],
#             run_info["avg_response_time"],
#             run_info["hr_accuracy"],
#             run_info["comm_accuracy"],
#             "-",  # For overall table, we do not aggregate AvgMBInt per file
#             "-"   # Similarly for valid/invalid percentages
#         ))
    
#     grand_total_records = sum(s["records"] for s in all_summaries)
#     grand_correct_hr = sum(s["records"] * s["hr_accuracy"] / 100 for s in all_summaries)
#     grand_correct_comm = sum(s["records"] * s["comm_accuracy"] / 100 for s in all_summaries)
    
#     overall_hr_acc = (grand_correct_hr / grand_total_records * 100) if grand_total_records else 0
#     overall_comm_acc = (grand_correct_comm / grand_total_records * 100) if grand_total_records else 0
    
#     overall_valid = sum(s["valid_response_percent"] for s in all_summaries) / len(all_summaries) if all_summaries else 0
#     overall_invalid = sum(s["invalid_response_percent"] for s in all_summaries) / len(all_summaries) if all_summaries else 0
    
#     print("\nOVERALL STATISTICS ACROSS ALL RUNS")
#     print(f"Total records processed: {grand_total_records}")
#     print(f"Overall HR Prediction Accuracy: {overall_hr_acc:.2f}%")
#     print(f"Overall Communication Volume Prediction Accuracy: {overall_comm_acc:.2f}%")
#     print(f"Overall Valid/Invalid Responses (%): {overall_valid:.1f}/{overall_invalid:.1f}")
    
#     print("\nACCURACY STATISTICS GROUPED BY BUFFER SIZE")
#     for buf, stats_list in buffer_groups.items():
#         if not stats_list:
#             print(f"Buffer size {buf}: No matching runs found.")
#             continue
#         total_records = sum(st["overall_records"] for st in stats_list)
#         hr_correct_sum = sum(st["overall_hr_acc"] * st["overall_records"] / 100 for st in stats_list)
#         comm_correct_sum = sum(st["overall_comm_acc"] * st["overall_records"] / 100 for st in stats_list)
#         hr_agg = (hr_correct_sum / total_records * 100) if total_records else 0
#         comm_agg = (comm_correct_sum / total_records * 100) if total_records else 0
#         print(f"Buffer size {buf}:")
#         print(f"  Total records: {total_records}")
#         print(f"  HR Prediction Accuracy: {hr_agg:.2f}%")
#         print(f"  Communication Volume Prediction Accuracy: {comm_agg:.2f}%\n")
    
# if __name__ == "__main__":
#     main()
