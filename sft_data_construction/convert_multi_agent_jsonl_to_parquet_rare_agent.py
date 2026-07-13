# -*- coding: utf-8 -*-
# @File    : convert_multi_agent_jsonl_to_parquet_rare_agent.py
# @Desc    : Convert multi-agent JSONL SFT records to VERL SFT parquet.
#            This version adds rare-agent oversampling for generation-style
#            agent-name SFT targets.
#
# Supports response format:
#   {"label":"A","agents":["Planner"]}
#   {"label":"A","agents":["Solver"]}
#
# Also backward-compatible with old response format:
#   {"label":"A","agent":"Planner"}
#   {"label":"B","agent":null}
#
# Output parquet keeps:
#   - prompt: str
#   - response: canonical response string, possibly multiple JSON lines
#   - candidate_agents: list[str]
#   - label: "A"/"B"/"C"
#   - agent_names: list[str]
#   - agent_name: first agent or None, for backward-compatible debugging only
#
# Rare-agent oversampling:
#   - Only applies to train data. Do NOT enable it for validation/test.
#   - Only oversamples label=A rows with non-empty agent_names.
#   - The repeat count is based on the rarest gold agent in the row:
#       repeat = ceil((max_agent_freq / min_agent_freq) ** alpha)
#     then clipped by --rare-agent-repeat-cap.
#   - To avoid making the whole dataset collapse to label=A, extra duplicated
#     A rows are globally capped by --rare-agent-max-extra-ratio.

import argparse
import copy
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


VALID_LABELS = {"A", "B", "C"}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse line {i} in {path}: {e}") from e
    return rows


def parse_candidate_agents_from_prompt(prompt: str) -> List[str]:
    """
    Parse:
        Candidate agents:
        - RoleAssigner
        - Solver
        - Evaluator
    """
    marker = "Candidate agents:"
    idx = prompt.find(marker)
    if idx < 0:
        return []

    tail = prompt[idx + len(marker):]
    agents: List[str] = []
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            if agents:
                break
            continue
        if line.startswith("- "):
            agent = line[2:].strip()
            if agent and agent.lower() not in {"none", "null", "__none__"}:
                agents.append(agent)
        else:
            if agents:
                break

    return dedupe_str_list(agents)


def parse_list_like(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                value = parsed
            else:
                value = [text]
        except Exception:
            value = [x.strip() for x in text.replace(";", ",").split(",")]
    if not isinstance(value, (list, tuple)):
        return []
    return dedupe_str_list([str(x).strip() for x in value if x is not None])


def dedupe_str_list(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in values:
        s = str(x).strip()
        if not s:
            continue
        if s.lower() in {"none", "null", "__none__"}:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def normalize_candidate_agents(rec: Dict[str, Any], prompt: str) -> List[str]:
    candidate_agents = rec.get("candidate_agents", None)
    if candidate_agents is None:
        candidate_agents = rec.get("agents", None)
    if candidate_agents is None:
        candidate_agents = rec.get("candidate_agent_list", None)

    parsed = parse_list_like(candidate_agents)
    if not parsed:
        parsed = parse_candidate_agents_from_prompt(prompt)
    return parsed


def _json_objects_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse either:
      1) one JSON object
      2) one JSON list of objects
      3) multiple JSON objects separated by newlines or whitespace
      4) multiple JSON objects separated by literal "\\n"
    """
    text = str(text).strip()
    if not text:
        raise ValueError("Empty response text.")

    # Normalize literal escaped newline from some generated/serialized files.
    # Example: '{...}\\n{...}' -> '{...}\n{...}'
    text = text.replace("\\\\n", "\n")

    # Fast path: normal JSON object/list.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return list(obj)
        raise ValueError(f"Response JSON must be object or list[object], got: {type(obj)}")
    except json.JSONDecodeError:
        pass

    # Multi-object path: repeatedly raw_decode from the string.
    decoder = json.JSONDecoder()
    idx = 0
    objs: List[Dict[str, Any]] = []
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as e:
            preview = text[idx:idx + 200]
            raise ValueError(f"Invalid multi-JSON response near: {preview!r}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"Each response object must be a dict, got: {type(obj)}")
        objs.append(obj)
        idx = end

    if not objs:
        raise ValueError(f"Invalid response JSON: {text}")
    return objs


def parse_joint_response_multi(response_text: str) -> Dict[str, Any]:
    """
    New target format:
        {"label":"A","agents":["Planner"]}
        {"label":"A","agents":["Solver"]}

    Backward-compatible format:
        {"label":"A","agent":"Planner"}
        {"label":"B","agent":null}
    """
    objs = _json_objects_from_text(response_text)

    labels: List[str] = []
    agents: List[str] = []

    for obj in objs:
        label = str(obj.get("label", "")).strip().upper()
        if label not in VALID_LABELS:
            raise ValueError(f"Response label must be A/B/C, got object: {obj}")
        labels.append(label)

        # Prefer new field "agents"; keep old "agent" compatibility.
        if "agents" in obj:
            obj_agents = parse_list_like(obj.get("agents"))
        else:
            old_agent = obj.get("agent", None)
            obj_agents = parse_list_like([old_agent] if old_agent is not None else [])
        agents.extend(obj_agents)

    uniq_labels = sorted(set(labels))
    if len(uniq_labels) != 1:
        raise ValueError(
            f"All JSON response objects must have the same label, "
            f"got labels={uniq_labels}, text={response_text}"
        )

    label = uniq_labels[0]
    agent_names = dedupe_str_list(agents)

    # For B/C, the canonical response should not contain any responsible agents.
    if label in {"B", "C"}:
        agent_names = []

    return {
        "label": label,
        "agent_names": agent_names,
        "num_response_objects": len(objs),
    }


def sort_agents_by_candidate_order(agent_names: List[str], candidate_agents: List[str]) -> List[str]:
    """Make multi-agent target order deterministic."""
    if not agent_names:
        return []
    order = {a: i for i, a in enumerate(candidate_agents)}
    return sorted(dedupe_str_list(agent_names), key=lambda a: order.get(a, 10**9))


def build_canonical_response(label: str, agent_names: List[str]) -> str:
    label = str(label).strip().upper()
    if label not in VALID_LABELS:
        raise ValueError(f"Unsupported label: {label}")

    if label == "A":
        # One JSON object per responsible agent, matching the requested SFT target.
        # If A has no confident agent, keep one empty-agent object.
        if not agent_names:
            objs = [{"label": "A", "agents": []}]
        else:
            objs = [{"label": "A", "agents": [a]} for a in agent_names]
    else:
        objs = [{"label": label, "agents": []}]

    return "\n".join(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) for obj in objs)


def validate_or_extend_candidates(
    candidate_agents: List[str],
    agent_names: List[str],
    *,
    append_missing_agents: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Returns:
      candidate_agents, missing_agents
    """
    candidate_agents = dedupe_str_list(candidate_agents)
    candidate_set = set(candidate_agents)
    missing = [a for a in agent_names if a not in candidate_set]
    if append_missing_agents and missing:
        candidate_agents = dedupe_str_list(candidate_agents + missing)
    return candidate_agents, missing


def _keep_label(label: str, keep_prob_a: float, keep_prob_b: float, keep_prob_c: float) -> bool:
    if label == "A":
        return random.random() <= keep_prob_a
    if label == "B":
        return random.random() <= keep_prob_b
    if label == "C":
        return random.random() <= keep_prob_c
    raise ValueError(f"Unsupported label: {label}")


def count_gold_agents(rows: Iterable[Dict[str, Any]]) -> Counter:
    counter = Counter()
    for row in rows:
        if row.get("label") != "A":
            continue
        for a in row.get("agent_names", []) or []:
            counter[a] += 1
    return counter


def print_agent_frequency(counter: Counter, title: str, top_k: int = 20) -> None:
    if not counter:
        print(f"[{title}] No positive gold agents found.")
        return
    print(f"[{title}] num_agents={len(counter)}, total_positive_mentions={sum(counter.values())}")
    print(f"[{title}] Top-{top_k} frequent agents: {counter.most_common(top_k)}")
    rare = sorted(counter.items(), key=lambda x: (x[1], x[0]))[:top_k]
    print(f"[{title}] Top-{top_k} rare agents: {rare}")


def oversample_rare_agent_rows(
    rows: List[Dict[str, Any]],
    *,
    enabled: bool = False,
    alpha: float = 0.5,
    repeat_cap: int = 5,
    max_extra_ratio: float = 0.5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Oversample label=A rows containing rare gold agents.

    Args:
        rows: converted rows after label downsampling.
        enabled: if False, return rows unchanged.
        alpha: smoothing factor. 0.5 means sqrt(max_freq / min_freq).
               Larger alpha makes oversampling stronger.
        repeat_cap: maximum total copies for each row, including the original.
        max_extra_ratio: cap extra duplicated A rows by len(A_rows) * ratio.
                         Example: 0.5 means A rows increase by at most 50%.
                         Set <0 to disable this global cap.
        seed: random seed for sampling extra rows if capped.

    Returns:
        A shuffled list of rows.
    """
    if not enabled:
        return rows

    if repeat_cap <= 1:
        print("[RareAgent] repeat_cap <= 1, skip oversampling.")
        return rows

    rng = random.Random(seed)
    base_rows = list(rows)
    agent_counter = count_gold_agents(base_rows)
    print_agent_frequency(agent_counter, "RareAgent before oversampling")

    if not agent_counter:
        return base_rows

    max_freq = max(agent_counter.values())
    extra_pool: List[Dict[str, Any]] = []
    repeat_counter = Counter()

    for row in base_rows:
        if row.get("label") != "A":
            continue

        agent_names = row.get("agent_names", []) or []
        if not agent_names:
            continue

        # For multi-agent targets, use the rarest positive agent in this row.
        min_freq = min(agent_counter.get(a, max_freq) for a in agent_names)
        raw_repeat = math.ceil((max_freq / max(min_freq, 1)) ** alpha)
        repeat = min(repeat_cap, max(1, raw_repeat))

        if repeat <= 1:
            continue

        row_id = row.get("id") or row.get("trajectory_id") or "unknown"
        repeat_counter[tuple(agent_names)] += repeat - 1

        for _ in range(repeat - 1):
            duplicated = copy.deepcopy(row)
            duplicated["oversampled"] = True
            duplicated["oversample_repeat"] = repeat
            duplicated["oversample_min_agent_freq"] = min_freq
            duplicated["oversample_row_id"] = row_id
            extra_pool.append(duplicated)

    a_count = sum(1 for r in base_rows if r.get("label") == "A")
    total_extra = len(extra_pool)

    if max_extra_ratio is not None and max_extra_ratio >= 0:
        max_extra = int(a_count * max_extra_ratio)
        if total_extra > max_extra:
            print(
                f"[RareAgent] Extra rows capped: {total_extra} -> {max_extra} "
                f"by max_extra_ratio={max_extra_ratio}"
            )
            extra_pool = rng.sample(extra_pool, max_extra)

    for row in base_rows:
        row.setdefault("oversampled", False)
        row.setdefault("oversample_repeat", 1)
        row.setdefault("oversample_min_agent_freq", None)
        row.setdefault("oversample_row_id", None)

    out_rows = base_rows + extra_pool
    rng.shuffle(out_rows)

    before_label = Counter(r.get("label") for r in base_rows)
    after_label = Counter(r.get("label") for r in out_rows)
    before_agent = count_gold_agents(base_rows)
    after_agent = count_gold_agents(out_rows)

    print(f"[RareAgent] Rows before={len(base_rows)}, after={len(out_rows)}, added={len(extra_pool)}")
    print(f"[RareAgent] Label counts before: {dict(before_label)}")
    print(f"[RareAgent] Label counts after:  {dict(after_label)}")
    print_agent_frequency(after_agent, "RareAgent after oversampling")

    return out_rows


def convert_jsonl_to_verl_rows(
    records: Iterable[Dict[str, Any]],
    *,
    data_source: str = "maserror_nli_verifier",
    ability: str = "nli_verification",
    keep_prob_a: float = 1.0,
    keep_prob_b: float = 1.0,
    keep_prob_c: float = 1.0,
    append_missing_agents: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    token_counter_before = Counter()
    token_counter_after = Counter()
    missing_agent_counter = Counter()
    num_multi_agent_a = 0
    num_multi_json_response = 0

    for rec_idx, rec in enumerate(records):
        prompt = str(rec.get("prompt", "")).strip()
        response_text = str(rec.get("response", "")).strip()

        if not prompt:
            continue
        if not response_text:
            continue

        try:
            parsed = parse_joint_response_multi(response_text)
        except Exception as e:
            rec_id = rec.get("id", f"record_{rec_idx}")
            raise ValueError(f"Failed to parse response for id={rec_id}: {e}") from e

        label = parsed["label"]
        agent_names = parsed["agent_names"]
        token_counter_before[label] += 1
        if parsed.get("num_response_objects", 1) > 1:
            num_multi_json_response += 1
        if label == "A" and len(agent_names) > 1:
            num_multi_agent_a += 1

        if not _keep_label(label, keep_prob_a, keep_prob_b, keep_prob_c):
            continue

        candidate_agents = normalize_candidate_agents(rec, prompt)
        candidate_agents, missing_agents = validate_or_extend_candidates(
            candidate_agents,
            agent_names,
            append_missing_agents=append_missing_agents,
        )
        for a in missing_agents:
            missing_agent_counter[a] += 1

        # Make target agent order deterministic. This is important for generation SFT.
        agent_names = sort_agents_by_candidate_order(agent_names, candidate_agents)
        response_str = build_canonical_response(label, agent_names)

        row = {
            "prompt": prompt,
            "response": response_str,
            "candidate_agents": candidate_agents,
            "data_source": rec.get("data_source", data_source),
            "ability": rec.get("ability", ability),
            # Debug / bookkeeping fields
            "label": label,
            "agent_names": agent_names,
            "agent_name": agent_names[0] if agent_names else None,  # backward-compatible debug only
            "num_agents": len(agent_names),
            "target_error_type": rec.get("target_error_type"),
            "trajectory_id": rec.get("trajectory_id"),
            "id": rec.get("id"),
            # Filled later if rare-agent oversampling is enabled.
            "oversampled": False,
            "oversample_repeat": 1,
            "oversample_min_agent_freq": None,
            "oversample_row_id": None,
        }

        rows.append(row)
        token_counter_after[label] += 1

    print(
        f"Before filtering: "
        f"A={token_counter_before['A']}, "
        f"B={token_counter_before['B']}, "
        f"C={token_counter_before['C']}"
    )
    print(
        f"After filtering:  "
        f"A={token_counter_after['A']}, "
        f"B={token_counter_after['B']}, "
        f"C={token_counter_after['C']}"
    )
    print(f"Multi-JSON response records: {num_multi_json_response}")
    print(f"A records with multiple agents: {num_multi_agent_a}")
    if missing_agent_counter:
        print(f"[WARN] Found gold agents missing from candidate_agents; appended them. Top missing: {missing_agent_counter.most_common(20)}")

    print_agent_frequency(count_gold_agents(rows), "Gold agent frequency after filtering")

    return rows


def save_parquet(rows: List[Dict[str, Any]], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)


def inspect_parquet(path: str, n: int = 3) -> None:
    df = pd.read_parquet(path)
    print("[Inspect] columns:", list(df.columns))
    print("[Inspect] shape:", df.shape)
    if len(df) > 0:
        print("[Inspect] label counts:", df["label"].value_counts(dropna=False).to_dict())
        if "oversampled" in df.columns:
            print("[Inspect] oversampled counts:", df["oversampled"].value_counts(dropna=False).to_dict())
        if "agent_names" in df.columns:
            counter = Counter()
            for agents in df[df["label"] == "A"]["agent_names"].tolist():
                if isinstance(agents, list):
                    for a in agents:
                        counter[a] += 1
            print("[Inspect] positive agent top frequent:", counter.most_common(10))
            print("[Inspect] positive agent top rare:", sorted(counter.items(), key=lambda x: (x[1], x[0]))[:10])
        print("[Inspect] first rows:")
        cols = [
            c for c in [
                "id", "label", "agent_names", "response", "candidate_agents",
                "oversampled", "oversample_repeat", "oversample_min_agent_freq",
            ]
            if c in df.columns
        ]
        print(df[cols].head(n).to_string(index=False))


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert multi-agent JSONL SFT data to VERL SFT parquet.")

    parser.add_argument(
        "--input",
        default="./data/train_aegis_sft_no_agent_multi_agent_jsonl.jsonl",
        help="Path to input SFT JSONL",
    )
    parser.add_argument(
        "--output",
        default="./data/train_aegis_sft_no_agent_multi_agent_jsonl_balanced.parquet",
        help="Path to output parquet",
    )
    parser.add_argument("--data-source", default="maserror_nli_verifier")
    parser.add_argument("--ability", default="nli_verification")

    # Optional class downsampling.
    # Important: for train, be careful with B because too small keep_prob_b can make the model ignore B.
    parser.add_argument("--keep-prob-a", type=float, default=1.0)
    parser.add_argument("--keep-prob-b", type=float, default=0.3)
    parser.add_argument("--keep-prob-c", type=float, default=0.7)

    # If a response agent is not in candidate_agents due to source data issues,
    # append it to avoid an impossible training target.
    parser.add_argument("--append-missing-agents", type=str2bool, default=True)
    parser.add_argument("--inspect", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)

    # Rare-agent oversampling. Use only for TRAIN set.
    parser.add_argument(
        "--rare-agent-oversample",
        type=str2bool,
        default=False,
        help="Enable rare-agent oversampling. Use only for train data, not validation/test.",
    )
    parser.add_argument(
        "--rare-agent-alpha",
        type=float,
        default=0.5,
        help="Oversampling strength. 0.5 means sqrt(max_freq / min_freq).",
    )
    parser.add_argument(
        "--rare-agent-repeat-cap",
        type=int,
        default=3,
        help="Max total copies per rare-agent row, including the original row.",
    )
    parser.add_argument(
        "--rare-agent-max-extra-ratio",
        type=float,
        default=0.2,
        help=(
            "Max extra duplicated A rows as a ratio of original A rows. "
            "0.5 means A rows increase by at most 50%. Set -1 to disable this global cap."
        ),
    )

    args = parser.parse_args()
    random.seed(args.seed)

    records = load_jsonl(args.input)
    rows = convert_jsonl_to_verl_rows(
        records,
        data_source=args.data_source,
        ability=args.ability,
        keep_prob_a=args.keep_prob_a,
        keep_prob_b=args.keep_prob_b,
        keep_prob_c=args.keep_prob_c,
        append_missing_agents=args.append_missing_agents,
    )

    rows = oversample_rare_agent_rows(
        rows,
        enabled=args.rare_agent_oversample,
        alpha=args.rare_agent_alpha,
        repeat_cap=args.rare_agent_repeat_cap,
        max_extra_ratio=args.rare_agent_max_extra_ratio,
        seed=args.seed,
    )

    save_parquet(rows, args.output)
    print(f"Saved {len(rows)} rows to {args.output}")
    if args.inspect:
        inspect_parquet(args.output)


if __name__ == "__main__":
    main()
