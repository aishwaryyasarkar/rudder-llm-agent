#!/usr/bin/env python3
import re
import json
import sys
import glob
import os
import argparse

###############################################################################
# Helper to parse final HR from the parent's summary .txt file
###############################################################################
def parse_parent_final_hr(subdir_path):
    """
    Given a subdir path, go one folder up and read the file named <dirname>.txt.
    This file is expected to contain lines like:
      Rank 11 | ... HitRate 43.0000 ...
    Returns a dictionary mapping rank (int) to final hit rate (float).
    """
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
# Helper: interpret a value direction as +1 / -1 / 0
###############################################################################
def direction_to_sign(direction_val):
    """
    Maps a direction value to +1, -1, or 0.
    If direction_val is a number: > 0 => +1, < 0 => -1, else 0.
    If a string, it maps common words ("increase", "decrease", "stagnant", etc.)
    accordingly. Returns None if the input is empty or unrecognized.
    """
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
    elif s in ["remain", "unchanged", "stable", "same", "0", "stagnant", "No change", "no change"]:
        return 0
    return None

###############################################################################
# parse_decision_json: extract JSON from the agent text (new approach)
###############################################################################
def parse_decision_json(text):
    """
    Expects that somewhere in 'text' there is a "Decision:" marker,
    followed by the JSON block contained between the first and second
    <agent> markers. If only one <agent> is found, we look up to the last '}'.
    
    Returns a dict with keys:
      - "pred_hr_dir": int or None
      - "pred_comm_dir": int or None
      - "decision": string
    If no valid JSON block or required fields are present, returns None.
    """
    # Remove any code fences (triple backticks) and extraneous whitespace
    text = re.sub(r"(?im)^```(?:json)?", "", text).strip()
    text = re.sub(r"(?im)```$", "", text).strip()
    text = text.replace("```", "").strip()

    # Now look for the first <agent> marker
    first_agent_marker = "<agent>"
    idx_first = text.find(first_agent_marker)
    if idx_first == -1:
        print("DEBUG: No <agent> marker found after Decision: - skipping record.")
        return None

    # Keep substring starting just after the first <agent>
    canditextdate = text[idx_first + len(first_agent_marker):]

    # Look for the second <agent> marker to define our JSON block
    idx_second = canditextdate.find("<agent>")
    if idx_second != -1:
        candidate_json = canditextdate[:idx_second].strip()
        # If there is the word "json", remove it.
        candidate_json = re.sub(r"\bjson\b", "", candidate_json).strip()
        print(f"DEBUG: Found second <agent> marker at {idx_second} - using it to limit JSON.")
    else:
        print("DEBUG: No second <agent> marker found - using the rest of the text as JSON.")
        last_brace = canditextdate.rfind("}")
        if last_brace == -1:
            print("DEBUG: No closing brace found - skipping record.")
            return None
        candidate_json = canditextdate[:last_brace+1].strip()

    # --- New Code: Extract only the JSON object ---
    start_idx = candidate_json.find('{')
    end_idx = candidate_json.rfind('}')
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        print("DEBUG: Could not extract valid JSON block - skipping record.")
        return None
    candidate_json = candidate_json[start_idx:end_idx+1]
    print(f"DEBUG: extracted JSON=<<<{candidate_json}>>>")
    # ----------------------------------------------------

    # Try to load JSON
    try:
        data = json.loads(candidate_json)
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON decoding failed - skipping record. Error: {e}")
        return None

    # If the loaded JSON is not a dictionary but is a string,
    # treat the string as the decision, with no predicted directions.
    if not isinstance(data, dict):
        if isinstance(data, str) and data.strip():
            return {
                "pred_hr_dir": None,
                "pred_comm_dir": None,
                "decision": data.strip().lower(),
            }
        else:
            print("DEBUG: Candidate JSON is not a valid dictionary or non-empty string - skipping record.")
            return None

    # Must have "decision"
    decision_str = data.get("decision")
    if not isinstance(decision_str, str):
        print("DEBUG: 'decision' field is not a string - skipping record.")
        return None
    if not decision_str:
        print("DEBUG: No valid decision found - skipping record.")
        return None

    # If "expected_impact" is present, parse out its directions
    pred_hr_dir = None
    pred_comm_dir = None
    if "expected_impact" in data:
        exp = data["expected_impact"]
        # In case it's nested JSON as a string
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

    return {
        "pred_hr_dir": pred_hr_dir,
        "pred_comm_dir": pred_comm_dir,
        "decision": decision_str.strip().lower(),
    }

###############################################################################
# parse_latest_metrics: extract "Latest Metrics" values from the block
###############################################################################
def parse_latest_metrics(block):
    """
    Searches for a line of the form:
      - Latest Metrics: {... 'hitrate': 34.0, 'comm_volume': 39943.0, ...}
    Returns a tuple (hr_val, comm_val), or (None, None) if not found.
    """
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
# New Function: extract_minibatch_id
###############################################################################
def extract_minibatch_id(chunk):
    """
    Extracts the minibatch_id from the given chunk.
    Expects a pattern like:
      'minibatch_id': [X]
    Returns X as an int, or None if not found.
    """
    m = re.search(r"'minibatch_id'\s*:\s*\[\s*([0-9]+)\s*\]", chunk)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

###############################################################################
# parse_one_record: extract all required info from one record chunk
###############################################################################
def parse_one_record(record):
    """
    Splits a record chunk (separated by lines of "#####") and extracts:
      - Response time (with a case-insensitive regex "response time: 1.23s")
      - The JSON block from within the agent response (looking for "Decision:" and <agent>)
      - Latest Metrics (if available)
    Returns a dict with keys:
      "response_time", "pred_hr_dir", "pred_comm_dir", "this_hr", "this_comm", "decision"
    Returns None if the record cannot be parsed.
    """
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
    print(f"Parsed record: {rec}")
    return rec

###############################################################################
# sign(x): map a number to its sign (+1, -1, or 0)
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
# process_file: split a file into record chunks and calculate accuracies
###############################################################################
def process_file(filename, rank2finalhr):
    """
    Reads a file and splits it into chunks using lines of "#".
    For each chunk, parse_one_record() extracts:
      - The user's decision
      - The predicted direction of HR/comm_volume
      - The metrics (this_hr, this_comm, etc.)
    
    We also count invalid chunks (those that fail to parse).
    
    Then, for accuracy computation, we only consider records that have valid predicted
    directions (i.e. where at least one of pred_hr_dir or pred_comm_dir is not None).
    (Such records with no prediction are still counted as valid responses, but not scored.)
    
    Returns a summary dictionary for the file, including 'invalid_count' and
    the average minibatch interval computed via extract_minibatch_id().
    """
    with open(filename, "r") as f:
        text = f.read()
    
    # Split on lines that consist solely of one or more '#' characters.
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

    # Count valid responses (all parsed records)
    valid_count = len(parsed)

    # Only use records with predictions for accuracy calculation.
    correct_hr = 0
    correct_comm = 0
    scorable_decisions = 0
    response_times = []
    for i in range(len(parsed) - 1):
        r_current = parsed[i]
        r_next = parsed[i+1]

        # Skip this record from accuracy if both predicted directions are missing.
        if r_current["pred_hr_dir"] is None and r_current["pred_comm_dir"] is None:
            continue

        scorable_decisions += 1
        p_hr = r_current["pred_hr_dir"]
        p_comm = r_current["pred_comm_dir"]

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

    # New Code: Compute the average minibatch interval using extract_minibatch_id()
    minibatch_ids = []
    for c in chunks:
        mb_id = extract_minibatch_id(c)
        if mb_id is not None:
            minibatch_ids.append(mb_id)
    diff_list = []
    for i in range(1, len(minibatch_ids)):
        diff_list.append(minibatch_ids[i] - minibatch_ids[i-1])
    avg_minibatch_interval = sum(diff_list) / len(diff_list) if diff_list else None

    rank_match = re.search(r"\bRank:\s*(\d+)", text)
    rank = int(rank_match.group(1)) if rank_match else None

    final_hr = rank2finalhr[rank] if (rank and rank in rank2finalhr) else None

    yes_evict = sum(1 for r in parsed if r["decision"].lower().strip().rstrip('.') == "yes, evict")
    no_evict = sum(1 for r in parsed if r["decision"].lower().strip().rstrip('.') == "no, do not evict")

    return {
        "filename": os.path.basename(filename),
        "records": valid_count,
        "avg_response_time": avg_rt,
        "hr_accuracy": hr_acc,
        "comm_accuracy": comm_acc,
        "total_response_time": sum(r["response_time"] for r in parsed),
        "rank": rank,
        "final_hr": final_hr,
        "yes_evict_count": yes_evict,
        "no_evict_count": no_evict,
        "invalid_count": invalid_count,
        "attempted_records": len(chunks),
        "avg_minibatch_interval": avg_minibatch_interval  # New field
    }

###############################################################################
# evaluate_and_return_stats: process all .txt files in a given subdir
###############################################################################
def evaluate_and_return_stats(subdir):
    """
    - Reads the parent's final HR file to map Rank -> final HR.
    - Processes each *.txt file in subdir with process_file().
    - Aggregates stats.
    - Returns a dictionary with a 'table' (text summary), plus 'summaries'
      for each file, and overall_stats.
    """
    rank2finalhr = parse_parent_final_hr(subdir)
    pattern = os.path.join(subdir, "*.txt")
    files = glob.glob(pattern)
    if not files:
        print("No .txt files found in subdir:", subdir)
        sys.exit(1)
    
    summaries = []
    overall_records = 0
    overall_rt_sum = 0
    overall_correct_hr = 0
    overall_correct_comm = 0
    total_yes_evict = 0
    total_no_evict = 0

    # For aggregating overall avg minibatch interval across files
    all_mb_intervals = []

    table_lines = []
    header = "{:<30s} {:>5s} {:>8s} {:>5s} {:>5s} {:>12s} {:>12s} {:>10s} {:>8s}".format(
        "Filename", "Rank", "Records", "Yes", "No", "AvgRT(s)", "HRAcc(%)", "FinalHR", "Invalid"
    )
    table_lines.append(header)
    table_lines.append("-" * len(header))
    
    for file in sorted(files):
        summary = process_file(file, rank2finalhr)
        summaries.append(summary)

        overall_records += summary["records"]
        overall_rt_sum += summary["total_response_time"]
        overall_correct_hr += summary["hr_accuracy"] * summary["records"] / 100
        overall_correct_comm += summary["comm_accuracy"] * summary["records"] / 100
        total_yes_evict += summary["yes_evict_count"]
        total_no_evict += summary["no_evict_count"]

        if summary["avg_minibatch_interval"] is not None:
            all_mb_intervals.append(summary["avg_minibatch_interval"])

        table_lines.append(
            "{:<30s} {:>5s} {:>8d} {:>5d} {:>5d} {:>12.2f} {:>12.2f} {:>10s} {:>8d}".format(
                summary["filename"],
                str(summary["rank"]) if summary["rank"] is not None else "-",
                summary["records"],
                summary["yes_evict_count"],
                summary["no_evict_count"],
                summary["avg_response_time"],
                summary["hr_accuracy"],
                f"{summary['final_hr']:.2f}" if summary["final_hr"] is not None else "-",
                summary["invalid_count"]
            )
        )
    
    if overall_records > 0:
        overall_avg_rt = overall_rt_sum / overall_records
        overall_hr_acc = (overall_correct_hr / overall_records) * 100
        overall_comm_acc = (overall_correct_comm / overall_records) * 100
    else:
        overall_avg_rt = 0
        overall_hr_acc = 0
        overall_comm_acc = 0
    
    total_valid = overall_records
    total_attempted_overall = total_valid + sum(s['invalid_count'] for s in summaries)
    if total_attempted_overall > 0:
        valid_percent = total_valid * 100.0 / total_attempted_overall
        invalid_percent = (total_attempted_overall - total_valid) * 100.0 / total_attempted_overall
    else:
        valid_percent = invalid_percent = 0.0

    total_decisions = total_yes_evict + total_no_evict
    if total_decisions > 0:
        yes_percent = total_yes_evict * 100.0 / total_decisions
        no_percent = total_no_evict * 100.0 / total_decisions
    else:
        yes_percent = no_percent = 0.0

    # Compute overall average minibatch interval across files.
    overall_avg_mb_interval = sum(all_mb_intervals) / len(all_mb_intervals) if all_mb_intervals else None
    
    overall_stats = {
        "overall_records": overall_records,
        "total_yes_evict": total_yes_evict,
        "total_no_evict": total_no_evict,
        "total_invalid": sum(s['invalid_count'] for s in summaries),
        "attempted_records": total_attempted_overall,
        "overall_avg_rt": overall_avg_rt,
        "overall_hr_acc": overall_hr_acc,
        "overall_comm_acc": overall_comm_acc,
        "valid_percent": valid_percent,
        "invalid_percent": invalid_percent,
        "yes_percent": yes_percent,
        "no_percent": no_percent,
        "overall_avg_mb_interval": overall_avg_mb_interval  # New overall stat
    }
    
    return {
        "table": "\n".join(table_lines),
        "summaries": summaries,
        "overall_stats": overall_stats
    }

###############################################################################
# Main function
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HR and Communication Volume Prediction Accuracy across multiple runs, "
                    "using next-record metrics as ground truth for each decision."
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
    
    for agent_model_dir in os.listdir(args.agent_dir):
        full_agent_model_dir = os.path.join(args.agent_dir, agent_model_dir)
        # Only process directories containing '-nocutoff' and skip those with '+contextwindow'
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
                        for summary in run_results["summaries"]:
                            summary["agent_model"] = agent_model_dir
                            summary["dataset"] = dataset_dir
                            summary["run_dir"] = os.path.basename(run_dir)
                            all_summaries.append(summary)
                        run_info = {
                            "agent_model": agent_model_dir,
                            "dataset": dataset_dir,
                            "run_dir": os.path.basename(run_dir),
                            "buffer": extracted_buffer,
                            "records": run_results["overall_stats"]["overall_records"],
                            "avg_response_time": run_results["overall_stats"]["overall_avg_rt"],
                            "hr_accuracy": run_results["overall_stats"]["overall_hr_acc"],
                            "comm_accuracy": run_results["overall_stats"]["overall_comm_acc"],
                            "avg_minibatch_interval": run_results["overall_stats"].get("overall_avg_mb_interval", None)
                        }
                        run_info_list.append(run_info)
                        buffer_groups[extracted_buffer].append(run_results["overall_stats"])
    
    header = "{:<20s} {:<15s} {:<15s} {:<15s} {:>8s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "Agent Model", "Dataset", "Run Dir", "Buffer Size",
        "Records", "AvgRT(s)", "HRAcc(%)", "CommAcc(%)", "AvgMBInt"
    )
    print(header)
    print("-" * len(header))
    for run_info in run_info_list:
        avg_mb_interval_str = f"{run_info['avg_minibatch_interval']:.2f}" if run_info["avg_minibatch_interval"] is not None else "-"
        print("{:<20s} {:<15s} {:<15s} {:<15s} {:>8d} {:>12.2f} {:>12.2f} {:>12.2f} {:>12s}".format(
            run_info["agent_model"],
            run_info["dataset"],
            run_info["run_dir"],
            run_info["buffer"],
            run_info["records"],
            run_info["avg_response_time"],
            run_info["hr_accuracy"],
            run_info["comm_accuracy"],
            avg_mb_interval_str
        ))
    
    grand_total_records = sum(s["records"] for s in all_summaries)
    grand_correct_hr = sum(s["records"] * s["hr_accuracy"] / 100 for s in all_summaries)
    grand_correct_comm = sum(s["records"] * s["comm_accuracy"] / 100 for s in all_summaries)

    if grand_total_records > 0:
        overall_avg_rt = (sum(s["total_response_time"] for s in all_summaries)
                          / grand_total_records)
        overall_hr_acc = (grand_correct_hr / grand_total_records) * 100
        overall_comm_acc = (grand_correct_comm / grand_total_records) * 100
    else:
        overall_avg_rt = 0
        overall_hr_acc = 0
        overall_comm_acc = 0
    
    total_yes = sum(s['yes_evict_count'] for s in all_summaries)
    total_no = sum(s['no_evict_count'] for s in all_summaries)
    total_valid = grand_total_records
    total_attempted_overall = total_valid + sum(s['invalid_count'] for s in all_summaries)
    if total_attempted_overall > 0:
        valid_percent = total_valid * 100.0 / total_attempted_overall
        invalid_percent = (total_attempted_overall - total_valid) * 100.0 / total_attempted_overall
    else:
        valid_percent = invalid_percent = 0.0

    total_decisions = total_yes + total_no
    if total_decisions > 0:
        yes_percent = total_yes * 100.0 / total_decisions
        no_percent = total_no * 100.0 / total_decisions
    else:
        yes_percent = no_percent = 0.0

    # Compute overall average minibatch interval across all file summaries
    all_mb_intervals = [s["avg_minibatch_interval"] for s in all_summaries if s["avg_minibatch_interval"] is not None]
    overall_avg_mb_interval = sum(all_mb_intervals) / len(all_mb_intervals) if all_mb_intervals else None
    # total records is the sum of all records across all summaries - valid + invalid
    print("\nOVERALL STATISTICS ACROSS ALL RUNS")
    print(f"Total records processed (Valid + Invalid): {total_attempted_overall}")
    print(f"Overall HR Prediction Accuracy: {overall_hr_acc:.2f}%")
    print(f"Overall Communication Volume Prediction Accuracy: {overall_comm_acc:.2f}%")
    print(f"Overall Average Response Time: {overall_avg_rt:.2f} seconds")
    print(f"Total Valid Responses: {total_valid} ({valid_percent:.2f}%), "
          f"Total Invalid: {total_attempted_overall - total_valid} ({invalid_percent:.2f}%)")
    print(f"Among valid responses, Yes Evict: {yes_percent:.2f}% and No Evict: {no_percent:.2f}%")
    print(f"Total Yes decisions: {total_yes}")
    print(f"Total No decisions: {total_no}")
    if overall_avg_mb_interval is not None:
        print(f"Overall Average Minibatch Interval: {overall_avg_mb_interval:.2f}")
    else:
        print("Overall Average Minibatch Interval: -")
    
    print("\nACCURACY STATISTICS GROUPED BY BUFFER SIZE")
    for buf, stats_list in buffer_groups.items():
        if not stats_list:
            print(f"Buffer size {buf}: No matching runs found.")
            continue
        total_records = sum(st["overall_records"] for st in stats_list)
        hr_correct_sum = sum(st["overall_hr_acc"] * st["overall_records"] / 100 for st in stats_list)
        comm_correct_sum = sum(st["overall_comm_acc"] * st["overall_records"] / 100 for st in stats_list)
        if total_records > 0:
            hr_agg = (hr_correct_sum / total_records) * 100
            comm_agg = (comm_correct_sum / total_records) * 100
        else:
            hr_agg = 0
            comm_agg = 0

        print(f"Buffer size {buf}:")
        print(f"  Total records: {total_records}")
        print(f"  HR Prediction Accuracy: {hr_agg:.2f}%")
        print(f"  Communication Volume Prediction Accuracy: {comm_agg:.2f}%\n")

if __name__ == "__main__":
    main()
