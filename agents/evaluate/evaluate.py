#!/usr/bin/env python3
import re
import json
import sys
import glob
import os

###############################################################################
# Helper to parse final HR from the parent's summary .txt file
###############################################################################
def parse_parent_final_hr(subdir_path):
    """
    Given a subdir path like:
      .../ogbn-arxiv_metis_n4_samp0_trainer4_<jobnumber>/
    go one folder up and look for a file with the same base name plus ".txt".
    That file contains lines like:
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
# Parse a single record from the agent log.
###############################################################################
def parse_record(record):
    """
    Parses a single record in the agent log.
    Requirements:
      - Must contain a "Response Time" line.
      - Uses the markers <user> and <agent> to extract the agent's JSON response.
      - Extracts the JSON block from the agent's response (removing any markdown formatting).
      - Also extracts the last line of the "Eviction History" section to get numeric changes.
    """
    result = {}

    # 1) Extract response time.
    rt_match = re.search(r"Response Time\s+([\d\.]+)s", record)
    if rt_match:
        result["response_time"] = float(rt_match.group(1))
    else:
        result["response_time"] = None

    # 2) Extract decision JSON block using the <agent> marker.
    agent_marker = "<agent>"
    agent_index = record.find(agent_marker)
    if agent_index == -1:
        result["expected_impact"] = {}
        result["decision"] = ""
    else:
        # Split the record by the <agent> marker.
        parts = record.split(agent_marker)
        # We assume the agent's response is the first block after <agent>.
        if len(parts) < 2:
            result["expected_impact"] = {}
            result["decision"] = ""
        else:
            agent_text = parts[1].strip()
            # Remove markdown formatting if present (e.g., triple backticks).
            agent_text = re.sub(r"^```(?:json)?", "", agent_text).strip()
            agent_text = re.sub(r"```$", "", agent_text).strip()
            # Find the JSON block by locating the first "{" and the last "}".
            first_brace = agent_text.find("{")
            last_brace = agent_text.rfind("}")
            if first_brace == -1 or last_brace == -1:
                candidate_json = agent_text
            else:
                candidate_json = agent_text[first_brace:last_brace+1]
            try:
                data = json.loads(candidate_json)
            except json.JSONDecodeError as e:
                # Fallback: attempt to extract using regex.
                impact_match = re.search(r'"expected_impact"\s*:\s*"([^"]+)"', candidate_json)
                decision_match = re.search(r'"decision"\s*:\s*"([^"]+)"', candidate_json)
                if impact_match and decision_match:
                    data = {
                        "expected_impact": impact_match.group(1),
                        "decision": decision_match.group(1)
                    }
                else:
                    data = {}
            # Ensure expected_impact is a dictionary. If it's not, try to parse it.
            exp_impact = data.get("expected_impact", {})
            if isinstance(exp_impact, str):
                try:
                    exp_impact = json.loads(exp_impact)
                except Exception:
                    exp_impact = {}
            result["expected_impact"] = exp_impact
            result["decision"] = data.get("decision", "").strip().lower()

    # 3) Extract the last eviction history line.
    history_match = re.search(
        r"Eviction History \(sorted from oldest to newest\):(.*?)(?:\n\s*\n|Dynamic Message:|<agent>|Decision for minibatches)",
        record, re.DOTALL)
    history_block = history_match.group(1).strip() if history_match else ""
    history_lines = [line.strip() for line in history_block.splitlines() if line.strip()]
    last_line = history_lines[-1] if history_lines else ""

    # 4) Extract numeric values from the last eviction history line.
    hr_change_match = re.search(r"Hitrate changed:\s*([+\-]?\d+\.?\d*)", last_line)
    comm_change_match = re.search(r"comm_volume changed:\s*([+\-]?\d+\.?\d*)", last_line)
    try:
        result["actual_hr_change"] = float(hr_change_match.group(1)) if hr_change_match else None
    except Exception:
        result["actual_hr_change"] = None
    try:
        result["actual_comm_change"] = float(comm_change_match.group(1)) if comm_change_match else None
    except Exception:
        result["actual_comm_change"] = None

    return result

###############################################################################
# Helper to determine predicted direction.
###############################################################################
def get_predicted_direction(impact, filename, metric="hr", debug=False):
    """
    Determines predicted direction from expected_impact.
    If impact is a dictionary, extracts the appropriate field.
    For metric "hr", expects key "hitrate".
    For metric "comm", expects key "comm_volume".
    Returns 1 for increase, -1 for decrease, 0 for stagnant, or None if not matched.
    """
    if impact is None:
        if debug:
            print(f"[DEBUG] Impact is None; Filename: {filename}")
        return None

    if isinstance(impact, dict):
        try:
            if metric == "hr":
                val = impact.get("hitrate", "")
            elif metric == "comm":
                val = impact.get("comm_volume", "")
            else:
                if debug:
                    print(f"[DEBUG] Invalid metric specified: '{metric}'; Filename: {filename}")
                return None

            if val is None or val == "":
                if debug:
                    print(f"[DEBUG] Impact dictionary for metric '{metric}' is missing a value. Impact record: {impact}; Filename: {filename}")
                return None

            # Convert the extracted value to lower case.
            val = val.lower()
        except Exception as e:
            if debug:
                print(f"[DEBUG] Exception encountered while processing impact: {e}. Impact record: {impact}; Filename: {filename}")
            return None

        if val in ["increase", "increased"]:
            return 1
        elif val in ["decrease", "decreased"]:
            return -1
        elif val in ["stagnant", "unchanged", "stable", "constant", "same"]:
            return 0
        # Explicitly handle "none"
        elif val == "none":
            if debug:
                print(f"[DEBUG] Value 'none' encountered for metric '{metric}'. Impact record: {impact}; Filename: {filename}")
            return None
        # If it contains "-" or "+" as characters, then:
        elif "-" in val:
            return -1
        elif "+" in val:
            return 1
        # Attempt integer conversion if applicable.
        try:
            num_val = int(val)
        except ValueError:
            if debug:
                print(f"[DEBUG] Value '{val}' cannot be converted to an integer for metric '{metric}'. Impact record: {impact}; Filename: {filename}")
            return None

        if num_val < 0:
            return -1
        elif num_val > 0:
            return 1
        elif num_val == 0:
            return 0
        else:
            if debug:
                print(f"[DEBUG] Unexpected value '{val}' for metric '{metric}' in impact dictionary. Impact record: {impact}; Filename: {filename}")
            return None
    else:
        # Fallback: if impact is not a dict, assume it's a string and use regex matching.
        if metric == "hr":
            pattern = r"(HR will (increase|decrease|remain|unchanged))"
        elif metric == "comm":
            pattern = r"(comm_volume will (increase|decrease|remain|unchanged))"
        else:
            if debug:
                print(f"[DEBUG] Invalid metric specified: '{metric}'; Filename: {filename}")
            return None

        if debug:
            print(f"\n[DEBUG] Processing text for metric '{metric}': {impact}; Filename: {filename}")

        match = re.search(pattern, impact, re.IGNORECASE)
        if match:
            direction = match.group(2).lower()
            if debug:
                print(f"[DEBUG] Match found: '{match.group(1)}'; Direction extracted: '{direction}'; Filename: {filename}")
            if direction == "increase":
                return 1
            elif direction == "decrease":
                return -1
            elif direction in ["remain", "unchanged"]:
                return 0
        else:
            if debug:
                print(f"[DEBUG] No match found for metric '{metric}'. Impact text: {impact}; Filename: {filename}")
        return None

def sign(x):
    """Return the sign of x: 1 if positive, -1 if negative, 0 if zero, or None."""
    if x is None:
        return None
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

###############################################################################
# Process one file.
###############################################################################
def process_file(filename, rank2finalhr):
    """
    Processes an agent log file:
      - Splits the file into records using lines that consist solely of '#' characters.
      - Only counts a record if it contains "Response Time" and a valid decision JSON block.
      - Computes average response time, HR and comm prediction accuracy.
      - Counts the number of "Yes, evict." and "No, do not evict." decisions.
      - Extracts the rank from the file and looks up the final HR from the parent's file.
    """
    with open(filename, "r") as f:
        log_text = f.read()

    # Split records by lines that consist solely of '#' characters.
    records = re.split(r"^\s*#+\s*$", log_text, flags=re.MULTILINE)
    records = [rec for rec in records if rec.strip()]

    total_records = 0
    correct_hr = 0
    correct_comm = 0
    response_times = []
    yes_evict_count = 0
    no_evict_count = 0

    # Extract rank from the file (first occurrence).
    rank_match = re.search(r"\bRank:\s*(\d+)", log_text)
    rank = int(rank_match.group(1)) if rank_match else None

    for rec in records:
        if "Response Time" not in rec:
            continue

        data = parse_record(rec)
        # Only count records that have both a response time and an expected impact.
        if data["response_time"] is None or not data.get("expected_impact"):
            continue

        total_records += 1
        response_times.append(data["response_time"])

        decision = data.get("decision", "").lower()
        if decision == "yes, evict.":
            yes_evict_count += 1
        elif decision == "no, do not evict.":
            no_evict_count += 1

        pred_hr = get_predicted_direction(data["expected_impact"], filename, metric="hr")
        pred_comm = get_predicted_direction(data["expected_impact"], filename, metric="comm")
        actual_hr_sign = sign(data["actual_hr_change"])
        actual_comm_sign = sign(data["actual_comm_change"])

        if pred_hr is not None and actual_hr_sign is not None and pred_hr == actual_hr_sign:
            correct_hr += 1
        if pred_comm is not None and actual_comm_sign is not None and pred_comm == actual_comm_sign:
            correct_comm += 1

    avg_rt = sum(response_times) / len(response_times) if response_times else 0
    hr_acc = (correct_hr / total_records * 100) if total_records else 0
    comm_acc = (correct_comm / total_records * 100) if total_records else 0

    final_hr = None
    if rank is not None and rank in rank2finalhr:
        final_hr = rank2finalhr[rank]

    return {
        "filename": os.path.basename(filename),
        "records": total_records,
        "avg_response_time": avg_rt,
        "hr_accuracy": hr_acc,
        "comm_accuracy": comm_acc,
        "total_response_time": sum(response_times),
        "rank": rank,
        "final_hr": final_hr,
        "yes_evict_count": yes_evict_count,
        "no_evict_count": no_evict_count
    }

###############################################################################
# Evaluate a directory and return the results.
###############################################################################
def evaluate_and_return_stats(subdir):
    """
    Processes all .txt files in the given subdir and computes per-file summaries
    and overall statistics. Returns a dictionary with:
      - "table": a formatted string table of per-file results,
      - "summaries": a list of per-file summary dictionaries,
      - "overall_stats": a dictionary of overall statistics.
    """
    rank2hr = parse_parent_final_hr(subdir)
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
    overall_final_hr_sum = 0.0
    overall_final_hr_count = 0
    total_yes_evict = 0
    total_no_evict = 0

    table_lines = []
    header = "{:<30s} {:>5s} {:>8s} {:>5s} {:>5s} {:>12s} {:>12s} {:>10s}".format(
        "Filename", "Rank", "Records", "Yes", "No", "AvgRT(s)", "HRAcc(%)", "FinalHR"
    )
    table_lines.append(header)
    table_lines.append("-" * 100)
    
    for file in sorted(files):
        summary = process_file(file, rank2hr)
        summaries.append(summary)
        overall_records += summary["records"]
        overall_rt_sum += summary["total_response_time"]
        overall_correct_hr += summary["hr_accuracy"] * summary["records"] / 100
        overall_correct_comm += summary["comm_accuracy"] * summary["records"] / 100
        total_yes_evict += summary["yes_evict_count"]
        total_no_evict += summary["no_evict_count"]

        if summary["final_hr"] is not None:
            overall_final_hr_sum += summary["final_hr"]
            overall_final_hr_count += 1

        table_lines.append("{:<30s} {:>5s} {:>8d} {:>5d} {:>5d} {:>12.2f} {:>12.2f} {:>10s}".format(
            summary["filename"],
            str(summary["rank"]) if summary["rank"] is not None else "-",
            summary["records"],
            summary["yes_evict_count"],
            summary["no_evict_count"],
            summary["avg_response_time"],
            summary["hr_accuracy"],
            f"{summary['final_hr']:.2f}" if summary["final_hr"] is not None else "-"
        ))

    overall_avg_rt = overall_hr_acc = overall_comm_acc = overall_final_hr_avg = 0
    if overall_records > 0:
        overall_avg_rt = overall_rt_sum / overall_records
        overall_hr_acc = (overall_correct_hr / overall_records) * 100
        overall_comm_acc = (overall_correct_comm / overall_records) * 100

    if overall_final_hr_count > 0:
        overall_final_hr_avg = overall_final_hr_sum / overall_final_hr_count

    overall_stats = {
        "overall_records": overall_records,
        "total_yes_evict": total_yes_evict,
        "total_no_evict": total_no_evict,
        "overall_avg_rt": overall_avg_rt,
        "overall_hr_acc": overall_hr_acc,
        "overall_comm_acc": overall_comm_acc,
        "overall_final_hr_avg": overall_final_hr_avg,
    }

    return {"table": "\n".join(table_lines), "summaries": summaries, "overall_stats": overall_stats}

###############################################################################
# Main: Process all .txt files in the given subdir; then output table and overall stats.
###############################################################################
def main(subdir):
    results = evaluate_and_return_stats(subdir)
    print(results["table"])
    print("\nOVERALL STATISTICS")
    print(f"Total records processed: {results['overall_stats']['overall_records']}")
    print(f"Overall Yes Evict decisions: {results['overall_stats']['total_yes_evict']}")
    print(f"Overall No Evict decisions: {results['overall_stats']['total_no_evict']}")
    print(f"Overall Average Response Time: {results['overall_stats']['overall_avg_rt']:.2f} seconds")
    print(f"Overall HR Prediction Accuracy: {results['overall_stats']['overall_hr_acc']:.2f}%")
    print(f"Overall Communication Volume Prediction Accuracy: {results['overall_stats']['overall_comm_acc']:.2f}%")
    print(f"Overall Final HR (avg across ranks found): {results['overall_stats']['overall_final_hr_avg']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <subdir>")
        sys.exit(1)
    main(sys.argv[1])
