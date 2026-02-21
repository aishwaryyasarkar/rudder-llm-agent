#!/usr/bin/env python3
import re
import json
import sys
import glob
import os
import argparse
import math
from scipy.stats import chi2  # used for file-level chi-square intervals

###############################################################################
# 1) parse_parent_final_hr
###############################################################################
def parse_parent_final_hr(subdir_path):
    subdir_name = os.path.basename(os.path.normpath(subdir_path))
    parent_dir = os.path.dirname(os.path.normpath(subdir_path))
    final_hr_file = os.path.join(parent_dir, subdir_name + ".txt")
    
    rank2hr = {}
    if not os.path.isfile(final_hr_file):
        print(f"[WARNING] No parent final HR file found at: {final_hr_file}")
        return rank2hr
    
    rank_pattern = re.compile(r"Rank\s+(\d+).*?\|\s+HitRate\s+(\d+\.\d+)")
    with open(final_hr_file, "r") as f:
        for line in f:
            match = rank_pattern.search(line)
            if match:
                rank = int(match.group(1))
                hr_val = float(match.group(2))
                rank2hr[rank] = hr_val
    return rank2hr

###############################################################################
# 2) direction_to_sign
###############################################################################
def direction_to_sign(direction_val):
    if direction_val is None:
        return None
    if isinstance(direction_val, (int, float)):
        if direction_val > 0:
            return 1
        elif direction_val < 0:
            return -1
        else:
            return 0
    s = str(direction_val).lower().strip()
    if s in ["increase", "increased", "+", "+1"]:
        return 1
    elif s in ["decrease", "decreased", "-", "-1", "lower", "less"]:
        return -1
    elif s in ["remain", "unchanged", "stable", "same", "0", "stagnant", "no change", "no change"]:
        return 0
    return None

###############################################################################
# 3) parse_decision_json
###############################################################################
def parse_decision_json(text):
    text = re.sub(r"(?im)^```(?:json)?", "", text).strip()
    text = re.sub(r"(?im)```$", "", text).strip()
    text = text.replace("```", "").strip()
    first_agent_marker = "<agent>"
    idx_first = text.find(first_agent_marker)
    if idx_first == -1:
        # print("DEBUG: No <agent> marker found after Decision: - skipping record.")
        return None
    canditextdate = text[idx_first + len(first_agent_marker):]
    idx_second = canditextdate.find("<agent>")
    if idx_second != -1:
        candidate_json = canditextdate[:idx_second].strip()
        candidate_json = re.sub(r"\bjson\b", "", candidate_json).strip()
        # print(f"DEBUG: Found second <agent> marker at {idx_second} - using it to limit JSON.")
    else:
        # print("DEBUG: No second <agent> marker found - using the rest of the text as JSON.")
        last_brace = canditextdate.rfind("}")
        if last_brace == -1:
            # print("DEBUG: No closing brace found - skipping record.")
            return None
        candidate_json = canditextdate[:last_brace+1].strip()
    start_idx = candidate_json.find('{')
    end_idx = candidate_json.rfind('}')
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        # print("DEBUG: Could not extract valid JSON block - skipping record.")
        return None
    candidate_json = candidate_json[start_idx:end_idx+1]
    # print(f"DEBUG: extracted JSON=<<<{candidate_json}>>>")
    try:
        data = json.loads(candidate_json)
    except json.JSONDecodeError as e:
        # print(f"DEBUG: JSON decoding failed - skipping record. Error: {e}")
        return None
    if not isinstance(data, dict):
        if isinstance(data, str) and data.strip():
            return {"pred_hr_dir": None, "pred_comm_dir": None, "decision": data.strip().lower()}
        else:
            print("DEBUG: Candidate JSON is not a valid dictionary or non-empty string - skipping record.")
            return None
    decision_str = data.get("decision")
    if not isinstance(decision_str, str) or not decision_str:
        # print("DEBUG: No valid decision found - skipping record.")
        return None
    
    pred_hr_dir = None
    pred_comm_dir = None
    if "expected_impact" in data:
        exp = data["expected_impact"]
        if isinstance(exp, str):
            try:
                exp = json.loads(exp)
            except:
                pass
        if isinstance(exp, dict):
            hr_str = exp.get("hitrate", "")
            comm_str = exp.get("comm_volume", "")
            pred_hr_dir = direction_to_sign(hr_str)
            pred_comm_dir = direction_to_sign(comm_str)
    
    return {"pred_hr_dir": pred_hr_dir, "pred_comm_dir": pred_comm_dir, "decision": decision_str.strip().lower()}

###############################################################################
# 4) parse_latest_metrics
###############################################################################
def parse_latest_metrics(block):
    hr_val = None
    comm_val = None
    latest_line_match = re.search(r"(?i)-\s*Latest Metrics:\s*\{([^}]+)\}", block, re.DOTALL)
    if latest_line_match:
        content = latest_line_match.group(1)
        hr_match = re.search(r"'hitrate':\s*([\d\.]+)", content)
        if hr_match:
            try:
                hr_val = float(hr_match.group(1))
            except:
                pass
        comm_match = re.search(r"'comm_volume':\s*([\d\.]+)", content)
        if comm_match:
            try:
                comm_val = float(comm_match.group(1))
            except:
                pass
    return (hr_val, comm_val)

###############################################################################
# 5) extract_minibatch_id
###############################################################################
def extract_minibatch_id(chunk):
    m = re.search(r"'minibatch_id'\s*:\s*\[\s*([0-9]+)\s*\]", chunk)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

###############################################################################
# 6) parse_one_record
###############################################################################
def parse_one_record(record):
    rt_match = re.search(r"(?i)response\s*time[:\s]+([\d\.]+)s", record)
    if not rt_match:
        return None
    response_time = float(rt_match.group(1))
    decision_data = parse_decision_json(record)
    if decision_data is None:
        return None
    hr_val, comm_val = parse_latest_metrics(record)
    rec = {
        "response_time": response_time,
        "pred_hr_dir": decision_data["pred_hr_dir"],
        "pred_comm_dir": decision_data["pred_comm_dir"],
        "this_hr": hr_val,
        "this_comm": comm_val,
        "decision": decision_data["decision"]
    }
    # print(f"Parsed record: {rec}")
    return rec

###############################################################################
# 7) sign
###############################################################################
def sign(x):
    if x is None:
        return None
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

###############################################################################
# 8) process_file
###############################################################################
def process_file(filename, rank2finalhr):
    with open(filename, "r") as f:
        text = f.read()
    chunks = re.split(r"^\s*#+\s*$", text, flags=re.MULTILINE)
    chunks = [c for c in chunks if c.strip()]
    parsed = []
    invalid_count = 0
    for c in chunks:
        rec = parse_one_record(c)
        if rec is not None:
            parsed.append(rec)
        else:
            invalid_count += 1

    valid_count = len(parsed)
    correct_hr = 0
    correct_comm = 0
    scorable_decisions = 0
    response_times = []

    for i in range(len(parsed) - 1):
        r_current = parsed[i]
        r_next = parsed[i+1]
        if r_current["pred_hr_dir"] is None and r_current["pred_comm_dir"] is None:
            continue
        scorable_decisions += 1
        p_hr = r_current["pred_hr_dir"]
        p_comm = r_current["pred_comm_dir"]
        
        # Next-step difference
        if r_current["this_hr"] is not None and r_next["this_hr"] is not None:
            hr_diff = r_next["this_hr"] - r_current["this_hr"]
            a_hr = sign(hr_diff)
        else:
            a_hr = None
        if r_current["this_comm"] is not None and r_next["this_comm"] is not None:
            comm_diff = r_next["this_comm"] - r_current["this_comm"]
            a_comm = sign(comm_diff)
        else:
            a_comm = None
        
        if p_hr is not None and a_hr is not None and p_hr == a_hr:
            correct_hr += 1
        if p_comm is not None and a_comm is not None and p_comm == a_comm:
            correct_comm += 1
        
        response_times.append(r_current["response_time"])

    if scorable_decisions > 0:
        hr_acc = (correct_hr / scorable_decisions) * 100
        comm_acc = (correct_comm / scorable_decisions) * 100
        avg_rt = sum(response_times) / len(response_times)
    else:
        hr_acc = 0
        comm_acc = 0
        avg_rt = 0

    minibatch_ids = []
    for c in chunks:
        mb_id = extract_minibatch_id(c)
        if mb_id is not None:
            minibatch_ids.append(mb_id)
    diff_list = [minibatch_ids[i] - minibatch_ids[i-1] for i in range(1, len(minibatch_ids))]
    avg_minibatch_interval = sum(diff_list) / len(diff_list) if diff_list else None

    rank_match = re.search(r"\bRank:\s*(\d+)", text)
    rank = int(rank_match.group(1)) if rank_match else None
    final_hr = rank2finalhr[rank] if (rank and rank in rank2finalhr) else None

    yes_evict = sum(1 for r in parsed if r["decision"].lower().strip().rstrip('.') == "yes, evict")
    no_evict = sum(1 for r in parsed if r["decision"].lower().strip().rstrip('.') == "no, do not evict")

    return {
        "filename": os.path.basename(filename),
        "records": valid_count,
        "attempted_records": len(chunks),
        "hr_correct": correct_hr,
        "hr_scorable": scorable_decisions,
        "avg_response_time": avg_rt,
        "hr_accuracy": hr_acc,
        "comm_accuracy": comm_acc,
        "total_response_time": sum(r["response_time"] for r in parsed),
        "rank": rank,
        "final_hr": final_hr,
        "yes_evict_count": yes_evict,
        "no_evict_count": no_evict,
        "invalid_count": invalid_count,
        "avg_minibatch_interval": avg_minibatch_interval
    }

###############################################################################
# 9) evaluate_and_return_stats
###############################################################################
def evaluate_and_return_stats(subdir):
    rank2finalhr = parse_parent_final_hr(subdir)
    pattern = os.path.join(subdir, "*.txt")
    files = glob.glob(pattern)
    if not files:
        print("No .txt files found in subdir:", subdir)
        sys.exit(1)
    
    summaries = []
    overall_valid = 0
    overall_attempted = 0
    overall_rt_sum = 0
    overall_correct_comm = 0
    total_yes_evict = 0
    total_no_evict = 0
    hr_values = []
    all_mb_intervals = []

    for file in sorted(files):
        summary = process_file(file, rank2finalhr)
        summaries.append(summary)

        overall_valid += summary["records"]
        overall_attempted += summary["attempted_records"]
        overall_rt_sum += summary["total_response_time"]

        # Weighted approach for comm accuracy
        overall_correct_comm += summary["comm_accuracy"] * summary["records"] / 100
        total_yes_evict += summary["yes_evict_count"]
        total_no_evict += summary["no_evict_count"]
        hr_values.append(summary["hr_accuracy"])

        if summary["avg_minibatch_interval"] is not None:
            all_mb_intervals.append(summary["avg_minibatch_interval"])
    
    if overall_valid > 0:
        overall_avg_rt = overall_rt_sum / overall_valid
        overall_comm_acc = (overall_correct_comm / overall_valid) * 100
    else:
        overall_avg_rt = 0
        overall_comm_acc = 0
    
    if overall_attempted > 0:
        valid_percent = overall_valid * 100.0 / overall_attempted
        invalid_percent = (overall_attempted - overall_valid) * 100.0 / overall_attempted
    else:
        valid_percent = invalid_percent = 0.0
    
    if total_yes_evict + total_no_evict > 0:
        yes_percent = total_yes_evict * 100.0 / (total_yes_evict + total_no_evict)
        no_percent = total_no_evict * 100.0 / (total_yes_evict + total_no_evict)
    else:
        yes_percent = no_percent = 0.0

    overall_avg_mb_interval = None
    if all_mb_intervals:
        overall_avg_mb_interval = sum(all_mb_intervals) / len(all_mb_intervals)

    # This is the per-run "File-Level CI" via chi-square
    n_files = len(hr_values)
    if n_files > 1:
        mean_hr = sum(hr_values) / n_files
        s2 = sum((x - mean_hr) ** 2 for x in hr_values) / (n_files - 1)
        chi2_lower = chi2.ppf(0.025, n_files - 1)
        chi2_upper = chi2.ppf(0.975, n_files - 1)
        var_lower = (n_files - 1) * s2 / chi2_upper
        var_upper = (n_files - 1) * s2 / chi2_lower
        sd_lower = math.sqrt(var_lower)
        sd_upper = math.sqrt(var_upper)
        overall_hr_file_ci = (sd_lower, sd_upper)
    else:
        overall_hr_file_ci = (0, 0)

    overall_stats = {
        "overall_records": overall_valid,
        "attempted_records": overall_attempted,
        "total_yes_evict": total_yes_evict,
        "total_no_evict": total_no_evict,
        "total_invalid": overall_attempted - overall_valid,
        "overall_avg_rt": overall_avg_rt,
        "overall_comm_acc": overall_comm_acc,
        "overall_hr_file_ci": overall_hr_file_ci,  # 95% CI (std. dev.) across that run's files
        "valid_percent": valid_percent,
        "invalid_percent": invalid_percent,
        "yes_percent": yes_percent,
        "no_percent": no_percent,
        "overall_avg_mb_interval": overall_avg_mb_interval
    }
    
    return {
        "summaries": summaries,
        "overall_stats": overall_stats
    }

###############################################################################
# 10) main
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Script that maintains your original stats plus File-Level CI (chi-square) per-run, and a GLOBAL Per-Record binomial CI across all runs. 95% CI for both."
    )
    parser.add_argument("--node-config", required=True, nargs='+', metavar="NODE_CONFIG",
                        help="Node config filter substring(s) (e.g. 'n4' 'n8' 'n16' 'n32').")
    parser.add_argument("--buffer-size", required=True, nargs='+', metavar="BUFFER_SIZE",
                        help="Buffer size filter substring(s) (e.g. '0.05' '0.25', etc.).")
    parser.add_argument("--datasets", required=True,
                        help="Comma-separated list of dataset names to include (e.g. 'papers100M,arxiv').")
    parser.add_argument("--agent_models", required=True, nargs='+',
                        help="List of agent models to include.")
    parser.add_argument("--agent_dir", required=True,
                        help="Base directory containing agent model subdirectories.")
    args = parser.parse_args()
    
    datasets = [ds.strip() for ds in args.datasets.split(',')]
    
    all_summaries = []
    run_info_list = []
    buffer_groups = {buf: [] for buf in args.buffer_size}

    # Global accumulators for binomial across *all runs*
    global_hr_correct = 0
    global_hr_scorable = 0

    # We'll also store a global list of *all* hr_values from all files 
    # to do a "Global Per-File CI" across everything.
    global_hr_values = []

    for agent_model_dir in os.listdir(args.agent_dir):
        full_agent_model_dir = os.path.join(args.agent_dir, agent_model_dir)
        if "+contextwindow" in agent_model_dir.lower():
            continue
        if not os.path.isdir(full_agent_model_dir):
            continue
        if not any(model_str.lower() in agent_model_dir.lower() for model_str in args.agent_models):
            continue
        
        for dataset_dir in os.listdir(full_agent_model_dir):
            full_dataset_dir = os.path.join(full_agent_model_dir, dataset_dir)
            if not os.path.isdir(full_dataset_dir):
                continue
            if not any(ds.lower() in dataset_dir.lower() for ds in datasets):
                continue

            for root, dirs, files in os.walk(full_dataset_dir):
                if files:
                    txt_files = [f for f in files if f.endswith(".txt")]
                    if txt_files:
                        run_dir = root
                        parent_dir = os.path.dirname(run_dir)
                        summary_file = os.path.join(parent_dir, os.path.basename(run_dir) + ".txt")
                        if not os.path.isfile(summary_file):
                            continue
                        if not any(nc in run_dir for nc in args.node_config):
                            continue
                        m = re.search(r"/pf_(\d+\.\d+)/", run_dir)
                        if not m:
                            continue
                        extracted_buffer = m.group(1)
                        if extracted_buffer not in args.buffer_size:
                            continue

                        run_results = evaluate_and_return_stats(run_dir)

                        # For each file in this run
                        for summary in run_results["summaries"]:
                            # accumulate correct vs scorable for global binomial
                            global_hr_correct += summary["hr_correct"]
                            global_hr_scorable += summary["hr_scorable"]
                            
                            # keep a global list of hr_values 
                            # (the per-file "hr_accuracy") for a global file-level CI:
                            # (We do 1 hr_value per file.)
                            global_hr_values.append(summary["hr_accuracy"])

                            # keep everything else
                            all_summaries.append(summary)

                            # attach context
                            summary["agent_model"] = agent_model_dir
                            summary["dataset"] = dataset_dir
                            summary["run_dir"] = os.path.basename(run_dir)

                        stats = run_results["overall_stats"]

                        run_info = {
                            "agent_model": agent_model_dir,
                            "dataset": dataset_dir,
                            "run_dir": os.path.basename(run_dir),
                            "buffer": extracted_buffer,
                            "records": stats["overall_records"],
                            "attempted_records": stats["attempted_records"],
                            "avg_response_time": stats["overall_avg_rt"],
                            "hr_accuracy_ci": stats["overall_hr_file_ci"],  # file-level CI for that run
                            "comm_accuracy": stats["overall_comm_acc"],
                            "avg_minibatch_interval": stats.get("overall_avg_mb_interval", None),
                        }
                        run_info_list.append(run_info)
                        buffer_groups[extracted_buffer].append(stats)

    # Now we print the same table as you had, with "File-Level CI" in a column
    header = "{:<20s} {:<15s} {:<15s} {:<15s} {:>10s} {:>12s} {:>20s} {:>12s} {:>12s}".format(
        "Agent Model", "Dataset", "Run Dir", "Buffer Size",
        "Attempted", "AvgRT(s)", "File-Level CI", "CommAcc(%)", "AvgMBInt"
    )
    print(header)
    print("-" * len(header))
    for run_info in run_info_list:
        ci = run_info["hr_accuracy_ci"]
        hr_file_ci_str = f"(-{ci[0]:.2f} to +{ci[1]:.2f})"
        avg_mb_str = (
            f"{run_info['avg_minibatch_interval']:.2f}"
            if run_info["avg_minibatch_interval"] is not None
            else "-"
        )
        print("{:<20s} {:<15s} {:<15s} {:<15s} {:>10d} {:>12.2f} {:>20s} {:>12.2f} {:>12s}".format(
            run_info["agent_model"],
            run_info["dataset"],
            run_info["run_dir"],
            run_info["buffer"],
            run_info["attempted_records"],
            run_info["avg_response_time"],
            hr_file_ci_str,
            run_info["comm_accuracy"],
            avg_mb_str
        ))
    
    # Same aggregator stats as before
    grand_total_attempted = sum(s["attempted_records"] for s in all_summaries)
    grand_total_valid = sum(s["records"] for s in all_summaries)
    # Weighted approach for comm:
    grand_correct_comm = sum(s["records"] * s["comm_accuracy"] / 100 for s in all_summaries)
    if grand_total_valid > 0:
        overall_avg_rt = sum(s["total_response_time"] for s in all_summaries) / grand_total_valid
        overall_comm_acc = (grand_correct_comm / grand_total_valid) * 100
    else:
        overall_avg_rt = 0
        overall_comm_acc = 0
    
    if grand_total_attempted > 0:
        valid_percent = grand_total_valid * 100.0 / grand_total_attempted
        invalid_percent = (grand_total_attempted - grand_total_valid) * 100.0 / grand_total_attempted
    else:
        valid_percent = invalid_percent = 0.0

    total_yes = sum(s['yes_evict_count'] for s in all_summaries)
    total_no = sum(s['no_evict_count'] for s in all_summaries)
    total_decisions = total_yes + total_no
    if total_decisions > 0:
        yes_percent = total_yes * 100.0 / total_decisions
        no_percent = total_no * 100.0 / total_decisions
    else:
        yes_percent = no_percent = 0.0

    all_mb_intervals = [s["avg_minibatch_interval"] for s in all_summaries if s["avg_minibatch_interval"] is not None]
    if all_mb_intervals:
        overall_avg_mb_interval = sum(all_mb_intervals) / len(all_mb_intervals)
    else:
        overall_avg_mb_interval = None

    print("\nOVERALL STATISTICS ACROSS ALL RUNS")
    print(f"Total records processed (Valid + Invalid): {grand_total_attempted}")
    print(f"Total Valid Responses: {grand_total_valid} ({valid_percent:.2f}%), "
          f"Total Invalid: {grand_total_attempted - grand_total_valid} ({invalid_percent:.2f}%)")
    print(f"Overall Comm Accuracy: {overall_comm_acc:.2f}%")
    print(f"Overall Avg Response Time: {overall_avg_rt:.2f}s")
    print(f"Yes Evict: {yes_percent:.2f}%, No Evict: {no_percent:.2f}%")
    print(f"Total Yes: {total_yes}, Total No: {total_no}")
    if overall_avg_mb_interval is not None:
        print(f"Overall Avg MB Interval: {overall_avg_mb_interval:.2f}")
    else:
        print("Overall Avg MB Interval: -")

    # GLOBAL per-record binomial CI across all runs
    if global_hr_scorable > 0:
        p_global = global_hr_correct / global_hr_scorable
        se_global = math.sqrt(p_global * (1 - p_global) / global_hr_scorable)
        margin_global = 1.96 * se_global * 100  # 95% CI
        hr_global_percent = p_global * 100
        print(f"\nGLOBAL Per-Record CI Across ALL Runs = ±{margin_global:.2f}")
        print(f"GLOBAL Overall HR Accuracy (all runs, all records) = {hr_global_percent:.2f}%")
    else:
        print("\nNo scorable decisions found across all runs, skipping global binomial CI.")

    # GLOBAL per-file CI across all runs (chi-square on the entire global_hr_values list)
    n_files_global = len(global_hr_values)
    if n_files_global > 1:
        mean_hr_global = sum(global_hr_values)/n_files_global
        s2_global = sum((x - mean_hr_global)**2 for x in global_hr_values)/(n_files_global-1)
        chi2_lower_g = chi2.ppf(0.025, n_files_global - 1)
        chi2_upper_g = chi2.ppf(0.975, n_files_global - 1)
        var_lower_g = (n_files_global - 1)*s2_global/chi2_upper_g
        var_upper_g = (n_files_global - 1)*s2_global/chi2_lower_g
        sd_lower_g = math.sqrt(var_lower_g)
        sd_upper_g = math.sqrt(var_upper_g)
        print(f"GLOBAL Per-File CI Across ALL Runs = (-{sd_lower_g:.2f} to +{sd_upper_g:.2f})")
    else:
        print("Not enough files globally to compute a global per-file CI.")

    print("\nACCURACY STATISTICS GROUPED BY BUFFER SIZE")
    for buf, stats_list in buffer_groups.items():
        if not stats_list:
            print(f"Buffer size {buf}: No matching runs found.")
            continue
        total_records_buf = sum(st["overall_records"] for st in stats_list)
        comm_correct_sum_buf = sum(st["overall_comm_acc"] * st["overall_records"] / 100 for st in stats_list)
        if total_records_buf > 0:
            comm_agg_buf = (comm_correct_sum_buf / total_records_buf) * 100
        else:
            comm_agg_buf = 0
        print(f"Buffer size {buf}:")
        print(f"  Total records: {total_records_buf}")
        print(f"  Weighted Comm Accuracy: {comm_agg_buf:.2f}%\n")


if __name__ == "__main__":
    main()
