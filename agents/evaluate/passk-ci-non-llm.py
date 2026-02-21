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
# 1) parse_parent_final_hr  (unchanged)
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
# Helpers to parse decisions / metrics from each chunk
###############################################################################
def _between_agent_tags(text: str):
    first = text.find("<agent>")
    if first == -1:
        return None
    rest = text[first + len("<agent>") :]
    second = rest.find("<agent>")
    if second == -1:
        return rest.strip()
    return rest[:second].strip()

def parse_decision(text: str) -> str | None:
    # remove code fences if present
    text = re.sub(r"(?im)^```(?:json)?", "", text).replace("```", "").strip()
    between = _between_agent_tags(text)
    if between is None:
        return None

    # try JSON first, else treat as raw text
    start_idx = between.find("{")
    end_idx = between.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            data = json.loads(between[start_idx:end_idx+1])
            if isinstance(data, dict) and isinstance(data.get("decision"), str):
                return data["decision"].strip().lower()
        except Exception:
            pass
    raw = between.strip().lower()
    return raw if raw else None

def parse_latest_metrics(block):
    hr_val = None
    t_val = None
    # Primary: top-of-chunk dict
    m_hr = re.search(r"[\"']Pre_Avg_Hitrate[\"']\s*:\s*([-+]?\d+(?:\.\d+)?)", block)
    if m_hr:
        try:
            hr_val = float(m_hr.group(1))
        except Exception:
            pass
    m_t  = re.search(r"[\"']Pre_Avg_T_rpc[\"']\s*:\s*([-+]?\d+(?:\.\d+)?)", block)
    if m_t:
        try:
            t_val = float(m_t.group(1))
        except Exception:
            pass

    # Fallback (older "Latest Metrics" line)
    if hr_val is None or t_val is None:
        latest = re.search(r"(?i)-\s*Latest Metrics:\s*\{([^}]+)\}", block, re.DOTALL)
        if latest:
            content = latest.group(1)
            if hr_val is None:
                m = re.search(r"'hitrate':\s*([\d\.]+)", content)
                if m:
                    try: hr_val = float(m.group(1))
                    except: pass
            if t_val is None:
                m = re.search(r"'comm_volume':\s*([\d\.]+)", content)
                if m:
                    try: t_val = float(m.group(1))
                    except: pass
    return hr_val, t_val

def extract_minibatch_id(chunk):
    # Primary: "Minibatch_ID: 4"  (case/space tolerant)
    m = re.search(r"(?i)\bMinibatch[_\s-]*ID\b\s*[:=]\s*([0-9]+)", chunk)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # Fallbacks
    m = re.search(r"(?i)'minibatch[_\s-]*id'\s*:\s*\[?\s*([0-9]+)\s*\]?", chunk)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"(?i)\bDecision\s+for\s+minibatches?\s+([0-9]+)\b", chunk)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

###############################################################################
# Eviction label: y = 1[(ΔHR) - λ*(ΔT_rpc) > ε]
###############################################################################
def eviction_label(pre_hr, pre_t, post_hr, post_t, lam, eps):
    if None in (pre_hr, pre_t, post_hr, post_t):
        return None
    d_hr = post_hr - pre_hr
    d_t  = post_t  - pre_t
    S = d_hr - lam * d_t
    return 1 if S > eps else 0

###############################################################################
# Parse one record chunk
###############################################################################
def parse_one_record(record):
    rt_match = re.search(r"(?i)response\s*time[:\s]+([\d\.]+)s", record)
    if not rt_match:
        return None
    decision = parse_decision(record)
    if decision is None:
        return None
    hr_val, t_val = parse_latest_metrics(record)
    return {
        "response_time": float(rt_match.group(1)),
        "decision": decision,
        "this_hr": hr_val,
        "this_t": t_val,
    }

###############################################################################
# Process a single file -> eviction-label accuracy
###############################################################################
def process_file(filename, rank2finalhr, lam, eps):
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

    # Pairwise scoring
    pairs_scorable = 0
    pairs_correct = 0
    response_times = []
    for i in range(len(parsed) - 1):
        cur, nxt = parsed[i], parsed[i+1]
        response_times.append(cur["response_time"])
        yhat = 1 if cur["decision"].startswith("yes, evict") else 0
        y = eviction_label(cur["this_hr"], cur["this_t"], nxt["this_hr"], nxt["this_t"], lam, eps)
        if y is None:
            continue
        pairs_scorable += 1
        if yhat == y:
            pairs_correct += 1

    hr_accuracy = (pairs_correct / pairs_scorable) * 100.0 if pairs_scorable else 0.0
    avg_rt = (sum(response_times) / len(response_times)) if response_times else 0.0

    # Avg minibatch interval in this file
    mb_ids = [extract_minibatch_id(c) for c in chunks]
    mb_ids = [m for m in mb_ids if m is not None]
    diffs = [mb_ids[i] - mb_ids[i - 1] for i in range(1, len(mb_ids))]
    avg_mb_interval = sum(diffs) / len(diffs) if diffs else None

    m = re.search(r"\bRank:\s*(\d+)", text)
    rank = int(m.group(1)) if m else None
    final_hr = rank2finalhr.get(rank) if rank is not None else None

    yes_evict = sum(1 for r in parsed if r["decision"].strip().lower().rstrip(".") == "yes, evict")
    no_evict  = sum(1 for r in parsed if r["decision"].strip().lower().rstrip(".") == "no, do not evict")

    return {
        "filename": os.path.basename(filename),
        "records": len(parsed),
        "attempted_records": len(chunks),
        "pairs_scorable": pairs_scorable,
        "pairs_correct": pairs_correct,
        "avg_response_time": avg_rt,
        "hr_accuracy": hr_accuracy,     # eviction-label accuracy (%)
        "total_response_time": sum(r["response_time"] for r in parsed),
        "rank": rank,
        "final_hr": final_hr,
        "yes_evict_count": yes_evict,
        "no_evict_count": no_evict,
        "invalid_count": invalid_count,
        "avg_minibatch_interval": avg_mb_interval,
    }

###############################################################################
# Evaluate a run dir (compute per-file stats + per-run CI)
###############################################################################
def evaluate_and_return_stats(subdir, lam, eps):
    rank2finalhr = parse_parent_final_hr(subdir)
    files = glob.glob(os.path.join(subdir, "*.txt"))
    if not files:
        print("No .txt files found in subdir:", subdir)
        sys.exit(1)

    summaries = []
    overall_valid = 0
    overall_attempted = 0
    overall_rt_sum = 0.0
    total_yes = total_no = 0
    all_mb_intervals = []

    # For per-run CI and global accumulation
    hr_values = []           # per-file accuracies (%)
    run_pairs = 0            # total scorable pairs in this run
    run_pairs_correct = 0

    for file in sorted(files):
        s = process_file(file, rank2finalhr, lam, eps)
        summaries.append(s)

        overall_valid += s["records"]
        overall_attempted += s["attempted_records"]
        overall_rt_sum += s["total_response_time"]
        total_yes += s["yes_evict_count"]
        total_no  += s["no_evict_count"]
        hr_values.append(s["hr_accuracy"])
        run_pairs += s["pairs_scorable"]
        run_pairs_correct += s["pairs_correct"]
        if s["avg_minibatch_interval"] is not None:
            all_mb_intervals.append(s["avg_minibatch_interval"])

    overall_avg_rt = (overall_rt_sum / overall_valid) if overall_valid else 0.0
    overall_hr_acc = (run_pairs_correct / run_pairs) * 100.0 if run_pairs else 0.0

    if overall_attempted > 0:
        valid_percent = overall_valid * 100.0 / overall_attempted
        invalid_percent = 100.0 - valid_percent
    else:
        valid_percent = invalid_percent = 0.0

    yes_percent = (total_yes * 100.0 / (total_yes + total_no)) if (total_yes + total_no) else 0.0
    no_percent  = (total_no  * 100.0 / (total_yes + total_no)) if (total_yes + total_no) else 0.0
    overall_avg_mb_interval = sum(all_mb_intervals) / len(all_mb_intervals) if all_mb_intervals else None

    # Per-run "File-Level CI" via chi-square on per-file accuracies (std dev CI)
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
        overall_hr_file_ci = (0.0, 0.0)

    overall_stats = {
        "overall_records": overall_valid,
        "attempted_records": overall_attempted,
        "total_yes_evict": total_yes,
        "total_no_evict": total_no,
        "total_invalid": overall_attempted - overall_valid,
        "overall_avg_rt": overall_avg_rt,
        "overall_hr_acc": overall_hr_acc,         # pairs-weighted eviction accuracy
        "overall_hr_file_ci": overall_hr_file_ci, # chi-square CI on file accuracies
        "valid_percent": valid_percent,
        "invalid_percent": invalid_percent,
        "yes_percent": yes_percent,
        "no_percent": no_percent,
        "overall_avg_mb_interval": overall_avg_mb_interval,
        "overall_pairs": run_pairs,
        "overall_pairs_correct": run_pairs_correct,
    }
    return {"summaries": summaries, "overall_stats": overall_stats}

###############################################################################
# main
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Eviction-label accuracy + CIs. y=1[(ΔHR)-λ*(ΔT_rpc)>ε], decision yes/no -> 1/0."
    )
    parser.add_argument("--node-config", required=True, nargs='+', metavar="NODE_CONFIG",
                        help="Filter substring(s) for node config (e.g. n4 n8 n16).")
    parser.add_argument("--buffer-size", required=True, nargs='+', metavar="BUFFER_SIZE",
                        help="Filter substring(s) for buffer sizes (e.g. 0.05 0.25).")
    parser.add_argument("--datasets", required=True,
                        help="Comma-separated dataset names to include.")
    parser.add_argument("--agent_models", required=True, nargs='+',
                        help="Agent model directory substrings to include.")
    parser.add_argument("--agent_dir", required=True,
                        help="Base directory containing agent model subdirectories.")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Epsilon threshold for S.")
    parser.add_argument("--lambda-t", type=float, default=1.0,
                        help="Lambda scaling for ΔT_rpc.")
    args = parser.parse_args()
    
    datasets = [ds.strip() for ds in args.datasets.split(',')]
    
    all_summaries = []
    run_info_list = []
    buffer_groups = {buf: [] for buf in args.buffer_size}

    # Global accumulators for binomial CI across ALL pairs in ALL runs
    global_pairs = 0
    global_pairs_correct = 0

    # Also store per-file accuracies to compute a GLOBAL per-file CI (chi-square)
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
                    if not txt_files:
                        continue
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

                    run_results = evaluate_and_return_stats(run_dir, lam=args.lambda_t, eps=args.epsilon)

                    for summary in run_results["summaries"]:
                        global_pairs += summary["pairs_scorable"]
                        global_pairs_correct += summary["pairs_correct"]
                        global_hr_values.append(summary["hr_accuracy"])
                        summary["agent_model"] = agent_model_dir
                        summary["dataset"] = dataset_dir
                        summary["run_dir"] = os.path.basename(run_dir)
                        all_summaries.append(summary)

                    stats = run_results["overall_stats"]
                    run_info = {
                        "agent_model": agent_model_dir,
                        "dataset": dataset_dir,
                        "run_dir": os.path.basename(run_dir),
                        "buffer": extracted_buffer,
                        "records": stats["overall_records"],
                        "attempted_records": stats["attempted_records"],
                        "avg_response_time": stats["overall_avg_rt"],
                        "hr_accuracy_ci": stats["overall_hr_file_ci"],
                        "avg_minibatch_interval": stats.get("overall_avg_mb_interval"),
                        "pairs": stats.get("overall_pairs", 0),
                        "hr_accuracy": stats.get("overall_hr_acc", 0.0),
                    }
                    run_info_list.append(run_info)
                    buffer_groups[extracted_buffer].append(stats)

    # Table of per-run summaries
    header = "{:<20s} {:<15s} {:<15s} {:<15s} {:>10s} {:>12s} {:>20s} {:>10s} {:>12s}".format(
        "Agent Model", "Dataset", "Run Dir", "Buffer Size",
        "Attempted", "AvgRT(s)", "File-Level CI", "Pairs", "AvgMBInt"
    )
    print(header)
    print("-" * len(header))
    for r in run_info_list:
        ci = r["hr_accuracy_ci"]
        hr_file_ci_str = f"(-{ci[0]:.2f} to +{ci[1]:.2f})"
        avg_mb_str = f"{r['avg_minibatch_interval']:.2f}" if r["avg_minibatch_interval"] is not None else "-"
        print("{:<20s} {:<15s} {:<15s} {:<15s} {:>10d} {:>12.2f} {:>20s} {:>10d} {:>12s}".format(
            r["agent_model"], r["dataset"], r["run_dir"], r["buffer"],
            r["attempted_records"], r["avg_response_time"], hr_file_ci_str,
            r["pairs"], avg_mb_str
        ))

    # Global roll-ups
    grand_total_attempted = sum(s["attempted_records"] for s in all_summaries)
    grand_total_valid = sum(s["records"] for s in all_summaries)
    overall_avg_rt = (sum(s["total_response_time"] for s in all_summaries) / grand_total_valid) if grand_total_valid else 0.0

    if grand_total_attempted > 0:
        valid_percent = grand_total_valid * 100.0 / grand_total_attempted
        invalid_percent = (grand_total_attempted - grand_total_valid) * 100.0 / grand_total_attempted
    else:
        valid_percent = invalid_percent = 0.0

    all_mb_intervals = [s["avg_minibatch_interval"] for s in all_summaries if s["avg_minibatch_interval"] is not None]
    overall_avg_mb_interval = sum(all_mb_intervals)/len(all_mb_intervals) if all_mb_intervals else None

    print("\nOVERALL STATISTICS ACROSS ALL RUNS")
    print(f"Total records processed (Valid + Invalid): {grand_total_attempted}")
    print(f"Total Valid Responses: {grand_total_valid} ({valid_percent:.2f}%), "
          f"Total Invalid: {grand_total_attempted - grand_total_valid} ({invalid_percent:.2f}%)")
    print(f"Overall Avg Response Time: {overall_avg_rt:.2f}s")
    if overall_avg_mb_interval is not None:
        print(f"Overall Avg MB Interval: {overall_avg_mb_interval:.2f}")
    else:
        print("Overall Avg MB Interval: -")

    # GLOBAL per-pair binomial CI across all runs
    if global_pairs > 0:
        p_global = global_pairs_correct / global_pairs
        se_global = math.sqrt(p_global * (1 - p_global) / global_pairs)
        margin_global = 1.96 * se_global * 100  # 95% CI (± in percentage points)
        hr_global_percent = p_global * 100
        print(f"\nGLOBAL Eviction-Label Accuracy = {hr_global_percent:.2f}% (±{margin_global:.2f} pp, 95% CI)")
    else:
        print("\nNo scorable pairs found across all runs, skipping global binomial CI.")

    # GLOBAL per-file CI across all files (chi-square on per-file accuracies)
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
        print(f"GLOBAL Per-File CI Across ALL Runs (std dev) = (-{sd_lower_g:.2f} to +{sd_upper_g:.2f})")
    else:
        print("Not enough files globally to compute a global per-file CI.")

    # Grouped by buffer size (pairs-weighted)
    print("\nACCURACY STATISTICS GROUPED BY BUFFER SIZE (pairs-weighted)")
    for buf, stats_list in buffer_groups.items():
        if not stats_list:
            print(f"Buffer size {buf}: No matching runs found.")
            continue
        total_pairs_buf = sum(st.get("overall_pairs", 0) for st in stats_list)
        correct_pairs_buf = sum(st.get("overall_pairs_correct", 0) for st in stats_list)
        acc_buf = (correct_pairs_buf / total_pairs_buf) * 100.0 if total_pairs_buf else 0.0
        print(f"Buffer size {buf}:")
        print(f"  Total scorable pairs: {total_pairs_buf}")
        print(f"  Eviction-Label Accuracy: {acc_buf:.2f}%\n")

if __name__ == "__main__":
    main()
