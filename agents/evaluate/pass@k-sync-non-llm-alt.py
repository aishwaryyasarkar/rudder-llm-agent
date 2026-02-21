#!/usr/bin/env python3
"""
Evaluate agent logs for LLM and non‑LLM agents.

Non‑LLM rule (requested):
  - decision == "yes, evict"     -> expect ΔHR > 0 (pred_hr_dir = +1)
  - decision == "no, do not evict" -> expect ΔHR >= 0 (pred_hr_dir = 0)
We do not score comm_volume for non‑LLM (pred_comm_dir=None).

LLM rule (existing): rely on explicit expected_impact.hitrate in JSON
  ("increase" -> +1, "decrease" -> -1, "same/0" -> 0).

Optional: --epsilon lets you treat small |ΔHR| as 0 when comparing.
"""
import re
import json
import sys
import glob
import os
import argparse
from typing import Dict, Tuple, Optional, Any

###############################################################################
# Helper to parse final HR from the parent's summary .txt file
###############################################################################
def parse_parent_final_hr(subdir_path: str) -> Dict[int, float]:
    """
    Given a subdir path, go one folder up and read <dirname>.txt which should
    contain lines like: "Rank 11 | ... HitRate 43.0000 ...". Returns
    {rank:int -> final_hr:float}. Missing file -> empty dict.
    """
    subdir_name = os.path.basename(os.path.normpath(subdir_path))
    parent_dir = os.path.dirname(os.path.normpath(subdir_path))
    final_hr_file = os.path.join(parent_dir, subdir_name + ".txt")

    rank2hr: Dict[int, float] = {}
    if not os.path.isfile(final_hr_file):
        print(f"[WARNING] No parent final HR file found at: {final_hr_file}")
        return rank2hr

    rank_pattern = re.compile(r"Rank\s+(\d+).*?\|\s+HitRate\s+(\d+\.?\d*)")
    with open(final_hr_file, "r") as f:
        for line in f:
            m = rank_pattern.search(line)
            if m:
                rank = int(m.group(1))
                hr_val = float(m.group(2))
                rank2hr[rank] = hr_val
    return rank2hr

###############################################################################
# Map a textual direction to sign
###############################################################################
def direction_to_sign(direction_val: Any) -> Optional[int]:
    if direction_val is None:
        return None
    if isinstance(direction_val, (int, float)):
        if direction_val > 0:
            return 1
        if direction_val < 0:
            return -1
        return 0
    s = str(direction_val).lower().strip()
    if s in {"increase", "increased", "+", "+1", "up"}:
        return 1
    if s in {"decrease", "decreased", "-", "-1", "down", "lower", "less"}:
        return -1
    if s in {"remain", "unchanged", "stable", "same", "0", "stagnant", "no change"}:
        return 0
    return None

###############################################################################
# Utility: extract the text between FIRST and SECOND <agent> markers.
###############################################################################
def _between_agent_tags(text: str) -> Optional[str]:
    first = text.find("<agent>")
    if first == -1:
        return None
    rest = text[first + len("<agent>") :]
    second = rest.find("<agent>")
    if second == -1:
        return rest.strip()
    return rest[:second].strip()

###############################################################################
# Non‑LLM expectation rule based on the decision string
###############################################################################
def infer_expectation_from_decision(decision_str: str) -> Tuple[Optional[int], Optional[int]]:
    d = decision_str.strip().lower().rstrip(".")
    if d == "yes, evict":
        return 1, None       # expect ΔHR > 0
    if d == "no, do not evict":
        return 0, None       # expect ΔHR >= 0 (we treat as 0 during compare)
    # If your non‑LLM has different phrasing, add more branches here
    return None, None

###############################################################################
# Parse decision block for either LLM (JSON) or non‑LLM (plain) agents
###############################################################################
def parse_decision_block(text: str, agent_type: str) -> Optional[Dict[str, Any]]:
    # Clean code fences if present
    text = re.sub(r"(?im)^```(?:json)?", "", text).replace("```", "").strip()

    between = _between_agent_tags(text)
    if between is None:
        return None

    # Try to find JSON first (LLM path). If not present and agent_type==nonllm,
    # treat the between‑tags text as the raw decision string.
    start_idx = between.find('{')
    end_idx = between.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cand = between[start_idx : end_idx + 1]
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            decision = data.get("decision")
            if not isinstance(decision, str) or not decision:
                return None
            pred_hr_dir = None
            pred_comm_dir = None
            if "expected_impact" in data:
                exp = data["expected_impact"]
                if isinstance(exp, str):
                    try:
                        exp = json.loads(exp)
                    except Exception:
                        pass
                if isinstance(exp, dict):
                    pred_hr_dir = direction_to_sign(exp.get("hitrate"))
                    pred_comm_dir = direction_to_sign(exp.get("comm_volume"))
            return {
                "decision": decision.strip().lower(),
                "pred_hr_dir": pred_hr_dir,
                "pred_comm_dir": pred_comm_dir,
            }

    # Non‑JSON path
    raw = between.strip()
    if not raw:
        return None

    # If agent_type is nonllm, infer expected sign from decision
    if agent_type == "nonllm":
        ph, pc = infer_expectation_from_decision(raw)
        return {
            "decision": raw.lower(),
            "pred_hr_dir": ph,
            "pred_comm_dir": pc,
        }

    # Otherwise (LLM without JSON) treat as a decision with no expectations
    return {
        "decision": raw.lower(),
        "pred_hr_dir": None,
        "pred_comm_dir": None,
    }

###############################################################################
# Extract Latest Metrics from a record chunk
###############################################################################
def parse_latest_metrics(block: str) -> Tuple[Optional[float], Optional[float]]:
    hr_val: Optional[float] = None
    comm_val: Optional[float] = None

    # HR from header field
    mh = re.search(r'[\'"]Pre_Avg_Hitrate[\'"]\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', block)
    if mh:
        try:
            hr_val = float(mh.group(1))
        except Exception:
            pass

    # Optional: use Pre_Avg_T_rpc as a stand-in for "comm"
    mc = re.search(r'[\'"]Pre_Avg_T_rpc[\'"]\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', block)
    if mc:
        try:
            comm_val = float(mc.group(1))
        except Exception:
            pass

    return hr_val, comm_val

###############################################################################
# Optional: get minibatch_id
###############################################################################
def extract_minibatch_id(chunk: str) -> Optional[int]:
    patterns = [
        # e.g., Rank: 1, Minibatch_ID: 4
        r"(?i)\bminibatch[_\s-]?id\b\s*[:=]\s*\[?\s*([0-9]+)\s*\]?",
    ]
    for pat in patterns:
        m = re.search(pat, chunk)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


###############################################################################
# Parse one record
###############################################################################
def parse_one_record(record: str, agent_type: str) -> Optional[Dict[str, Any]]:
    rt_match = re.search(r"(?i)response\s*time[:\s]+([\d\.]+)s", record)
    if not rt_match:
        return None
    response_time = float(rt_match.group(1))

    decision_data = parse_decision_block(record, agent_type)
    if decision_data is None:
        return None

    hr_val, comm_val = parse_latest_metrics(record)

    return {
        "response_time": response_time,
        "pred_hr_dir": decision_data["pred_hr_dir"],
        "pred_comm_dir": decision_data["pred_comm_dir"],
        "this_hr": hr_val,
        "this_comm": comm_val,
        "decision": decision_data["decision"],
    }

###############################################################################
# Sign with epsilon tolerance
###############################################################################
def sign(x: Optional[float], epsilon: float) -> Optional[int]:
    if x is None:
        return None
    if x > epsilon:
        return 1
    if x < -epsilon:
        return -1
    return 0

###############################################################################
# Process a single log file
###############################################################################
def process_file(filename: str, rank2finalhr: Dict[int, float], agent_type: str, epsilon: float) -> Dict[str, Any]:
    with open(filename, "r") as f:
        text = f.read()

    chunks = re.split(r"^\s*#+\s*$", text, flags=re.MULTILINE)
    chunks = [c for c in chunks if c.strip()]

    parsed = []
    invalid_count = 0
    for c in chunks:
        rec = parse_one_record(c, agent_type)
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
        cur = parsed[i]
        nxt = parsed[i + 1]

        p_hr = cur["pred_hr_dir"]
        p_comm = cur["pred_comm_dir"]

        # For LLM: require explicit prediction to score.
        # For non‑LLM: p_hr is set by inference so it will be scored.
        if p_hr is None and p_comm is None:
            continue

        scorable_decisions += 1

        a_hr = None
        if cur["this_hr"] is not None and nxt["this_hr"] is not None:
            a_hr = sign(nxt["this_hr"] - cur["this_hr"], epsilon)

        a_comm = None
        if cur["this_comm"] is not None and nxt["this_comm"] is not None:
            a_comm = sign(nxt["this_comm"] - cur["this_comm"], epsilon)

        # --- HR correctness ---
        if p_hr is not None and a_hr is not None:
            if agent_type == "nonllm" and cur["decision"].startswith("no, do not evict"):
                # Relaxed rule: expectation is ΔHR >= 0; we treat pred 0; accept actual 0 or +1
                if a_hr in (0, 1):
                    correct_hr += 1
            else:
                if p_hr == a_hr:
                    correct_hr += 1

        # --- Comm correctness ---
        if p_comm is not None and a_comm is not None and p_comm == a_comm:
            correct_comm += 1

        response_times.append(cur["response_time"])

    if scorable_decisions > 0:
        hr_acc = (correct_hr / scorable_decisions) * 100.0
        comm_acc = (correct_comm / scorable_decisions) * 100.0
        avg_rt = sum(response_times) / len(response_times) if response_times else 0.0
    else:
        hr_acc = comm_acc = avg_rt = 0.0

    # Average minibatch interval
    mb_ids = [extract_minibatch_id(c) for c in chunks]
    mb_ids = [m for m in mb_ids if m is not None]
    diffs = [mb_ids[i] - mb_ids[i - 1] for i in range(1, len(mb_ids))]
    avg_mb_interval = sum(diffs) / len(diffs) if diffs else None

    m = re.search(r"\bRank:\s*(\d+)", text)
    rank = int(m.group(1)) if m else None
    final_hr = rank2finalhr.get(rank) if rank is not None else None

    yes_evict = sum(1 for r in parsed if r["decision"].strip().lower().rstrip(".") == "yes, evict")
    no_evict = sum(1 for r in parsed if r["decision"].strip().lower().rstrip(".") == "no, do not evict")

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
        "avg_minibatch_interval": avg_mb_interval,
    }

###############################################################################
# Aggregate a run directory
###############################################################################
def evaluate_and_return_stats(subdir: str, agent_type: str, epsilon: float) -> Dict[str, Any]:
    rank2finalhr = parse_parent_final_hr(subdir)
    files = glob.glob(os.path.join(subdir, "*.txt"))
    if not files:
        print("No .txt files found in subdir:", subdir)
        sys.exit(1)

    summaries = []
    overall_records = 0
    overall_rt_sum = 0.0
    overall_correct_hr = 0.0
    overall_correct_comm = 0.0
    total_yes = 0
    total_no = 0
    all_mb_intervals = []

    table_lines = []
    header = "{:<30s} {:>5s} {:>8s} {:>5s} {:>5s} {:>12s} {:>12s} {:>10s} {:>8s}".format(
        "Filename", "Rank", "Records", "Yes", "No", "AvgRT(s)", "HRAcc(%)", "FinalHR", "Invalid"
    )
    table_lines.append(header)
    table_lines.append("-" * len(header))

    for file in sorted(files):
        s = process_file(file, rank2finalhr, agent_type, epsilon)
        summaries.append(s)

        overall_records += s["records"]
        overall_rt_sum += s["total_response_time"]
        overall_correct_hr += s["hr_accuracy"] * s["records"] / 100.0
        overall_correct_comm += s["comm_accuracy"] * s["records"] / 100.0
        total_yes += s["yes_evict_count"]
        total_no += s["no_evict_count"]
        if s["avg_minibatch_interval"] is not None:
            all_mb_intervals.append(s["avg_minibatch_interval"])

        table_lines.append(
            "{:<30s} {:>5s} {:>8d} {:>5d} {:>5d} {:>12.2f} {:>12.2f} {:>10s} {:>8d}".format(
                s["filename"],
                str(s["rank"]) if s["rank"] is not None else "-",
                s["records"],
                s["yes_evict_count"],
                s["no_evict_count"],
                s["avg_response_time"],
                s["hr_accuracy"],
                f"{s['final_hr']:.2f}" if s["final_hr"] is not None else "-",
                s["invalid_count"],
            )
        )

    if overall_records > 0:
        overall_avg_rt = overall_rt_sum / overall_records
        overall_hr_acc = (overall_correct_hr / overall_records) * 100.0
        overall_comm_acc = (overall_correct_comm / overall_records) * 100.0
    else:
        overall_avg_rt = overall_hr_acc = overall_comm_acc = 0.0

    total_attempted_overall = overall_records + sum(s['invalid_count'] for s in summaries)
    valid_percent = (overall_records * 100.0 / total_attempted_overall) if total_attempted_overall else 0.0
    invalid_percent = 100.0 - valid_percent if total_attempted_overall else 0.0

    total_decisions = total_yes + total_no
    yes_percent = (total_yes * 100.0 / total_decisions) if total_decisions else 0.0
    no_percent = (total_no * 100.0 / total_decisions) if total_decisions else 0.0

    overall_avg_mb_interval = sum(all_mb_intervals) / len(all_mb_intervals) if all_mb_intervals else None

    overall_stats = {
        "overall_records": overall_records,
        "total_yes_evict": total_yes,
        "total_no_evict": total_no,
        "total_invalid": sum(s['invalid_count'] for s in summaries),
        "attempted_records": total_attempted_overall,
        "overall_avg_rt": overall_avg_rt,
        "overall_hr_acc": overall_hr_acc,
        "overall_comm_acc": overall_comm_acc,
        "valid_percent": valid_percent,
        "invalid_percent": invalid_percent,
        "yes_percent": yes_percent,
        "no_percent": no_percent,
        "overall_avg_mb_interval": overall_avg_mb_interval,
    }

    return {"table": "\n".join(table_lines), "summaries": summaries, "overall_stats": overall_stats}

###############################################################################
# CLI
###############################################################################
def main():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate HR/Comm direction accuracy over logs. "
            "Use --agent-type nonllm to infer expectations from decisions; "
            "otherwise expect LLM JSON with expected_impact."
        )
    )
    p.add_argument("--node-config", required=True, nargs='+', help="Filter substrings for node config (e.g. n4 n8 n16)")
    p.add_argument("--buffer-size", required=True, nargs='+', help="Filter substrings for buffer sizes (e.g. 0.05 0.25)")
    p.add_argument("--datasets", required=True, help="Comma‑separated dataset names to include")
    p.add_argument("--agent_models", required=True, nargs='+', help="Agent model directory substrings to include")
    p.add_argument("--agent_dir", required=True, help="Base directory containing agent model subdirectories")
    p.add_argument("--agent-type", choices=["llm", "nonllm"], default="llm")
    p.add_argument("--epsilon", type=float, default=0.0, help="Tolerance for treating ΔHR≈0 as no‑change")
    args = p.parse_args()

    datasets = [ds.strip() for ds in args.datasets.split(',')]

    all_summaries = []
    run_info_list = []
    buffer_groups = {buf: [] for buf in args.buffer_size}

    for agent_model_dir in os.listdir(args.agent_dir):
        full_agent_model_dir = os.path.join(args.agent_dir, agent_model_dir)
        if "+contextwindow" in agent_model_dir.lower():
            continue
        if not os.path.isdir(full_agent_model_dir):
            continue
        if not any(m.lower() in agent_model_dir.lower() for m in args.agent_models):
            continue
        for dataset_dir in os.listdir(full_agent_model_dir):
            full_dataset_dir = os.path.join(full_agent_model_dir, dataset_dir)
            if not os.path.isdir(full_dataset_dir):
                continue
            if not any(ds.lower() in dataset_dir.lower() for ds in datasets):
                continue
            for root, dirs, files in os.walk(full_dataset_dir):
                if not files:
                    continue
                if not any(f.endswith(".txt") for f in files):
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

                run_results = evaluate_and_return_stats(run_dir, args.agent_type, args.epsilon)
                for s in run_results["summaries"]:
                    s["agent_model"] = agent_model_dir
                    s["dataset"] = dataset_dir
                    s["run_dir"] = os.path.basename(run_dir)
                    all_summaries.append(s)

                run_info = {
                    "agent_model": agent_model_dir,
                    "dataset": dataset_dir,
                    "run_dir": os.path.basename(run_dir),
                    "buffer": extracted_buffer,
                    "records": run_results["overall_stats"]["overall_records"],
                    "avg_response_time": run_results["overall_stats"]["overall_avg_rt"],
                    "hr_accuracy": run_results["overall_stats"]["overall_hr_acc"],
                    "comm_accuracy": run_results["overall_stats"]["overall_comm_acc"],
                    "avg_minibatch_interval": run_results["overall_stats"].get("overall_avg_mb_interval"),
                }
                run_info_list.append(run_info)
                buffer_groups[extracted_buffer].append(run_results["overall_stats"])

    header = (
        "{:<20s} {:<15s} {:<15s} {:<15s} {:>8s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
            "Agent Model", "Dataset", "Run Dir", "Buffer Size", "Records", "AvgRT(s)", "HRAcc(%)", "CommAcc(%)", "AvgMBInt"
        )
    )
    print(header)
    print("-" * len(header))
    for r in run_info_list:
        avg_mb_str = f"{r['avg_minibatch_interval']:.2f}" if r["avg_minibatch_interval"] is not None else "-"
        print(
            "{:<20s} {:<15s} {:<15s} {:<15s} {:>8d} {:>12.2f} {:>12.2f} {:>12.2f} {:>12s}".format(
                r["agent_model"], r["dataset"], r["run_dir"], r["buffer"], r["records"], r["avg_response_time"], r["hr_accuracy"], r["comm_accuracy"], avg_mb_str
            )
        )

    # Overall roll‑up across all file summaries
    grand_total_records = sum(s["records"] for s in all_summaries)
    grand_correct_hr = sum(s["records"] * s["hr_accuracy"] / 100.0 for s in all_summaries)
    grand_correct_comm = sum(s["records"] * s["comm_accuracy"] / 100.0 for s in all_summaries)

    if grand_total_records > 0:
        overall_avg_rt = (sum(s["total_response_time"] for s in all_summaries) / grand_total_records)
        overall_hr_acc = (grand_correct_hr / grand_total_records) * 100.0
        overall_comm_acc = (grand_correct_comm / grand_total_records) * 100.0
    else:
        overall_avg_rt = overall_hr_acc = overall_comm_acc = 0.0

    total_yes = sum(s['yes_evict_count'] for s in all_summaries)
    total_no = sum(s['no_evict_count'] for s in all_summaries)
    total_attempted_overall = grand_total_records + sum(s['invalid_count'] for s in all_summaries)
    valid_percent = (grand_total_records * 100.0 / total_attempted_overall) if total_attempted_overall else 0.0
    invalid_percent = 100.0 - valid_percent if total_attempted_overall else 0.0

    all_mb_intervals = [s["avg_minibatch_interval"] for s in all_summaries if s["avg_minibatch_interval"] is not None]
    overall_avg_mb_interval = sum(all_mb_intervals) / len(all_mb_intervals) if all_mb_intervals else None

    print("\nOVERALL STATISTICS ACROSS ALL RUNS")
    print(f"Total records processed (Valid + Invalid): {total_attempted_overall}")
    print(f"Overall HR Prediction Accuracy: {overall_hr_acc:.2f}%")
    print(f"Overall Communication Volume Prediction Accuracy: {overall_comm_acc:.2f}%")
    print(f"Overall Average Response Time: {overall_avg_rt:.2f} seconds")
    print(f"Total Valid Responses: {grand_total_records} ({valid_percent:.2f}%), Total Invalid: {total_attempted_overall - grand_total_records} ({invalid_percent:.2f}%)")
    if total_yes + total_no > 0:
        print(f"Among valid responses, Yes Evict: {total_yes*100/(total_yes+total_no):.2f}% and No Evict: {total_no*100/(total_yes+total_no):.2f}%")
    print(f"Total Yes decisions: {total_yes}")
    print(f"Total No decisions: {total_no}")
    print(f"Overall Average Minibatch Interval: {overall_avg_mb_interval:.2f}" if overall_avg_mb_interval is not None else "Overall Average Minibatch Interval: -")

    print("\nACCURACY STATISTICS GROUPED BY BUFFER SIZE")
    for buf, stats_list in buffer_groups.items():
        if not stats_list:
            print(f"Buffer size {buf}: No matching runs found.")
            continue
        total_records = sum(st["overall_records"] for st in stats_list)
        hr_correct_sum = sum(st["overall_hr_acc"] * st["overall_records"] / 100.0 for st in stats_list)
        comm_correct_sum = sum(st["overall_comm_acc"] * st["overall_records"] / 100.0 for st in stats_list)
        hr_agg = (hr_correct_sum / total_records) * 100.0 if total_records else 0.0
        comm_agg = (comm_correct_sum / total_records) * 100.0 if total_records else 0.0
        print(f"Buffer size {buf}:")
        print(f"  Total records: {total_records}")
        print(f"  HR Prediction Accuracy: {hr_agg:.2f}%")
        print(f"  Communication Volume Prediction Accuracy: {comm_agg:.2f}%\n")

if __name__ == "__main__":
    main()
