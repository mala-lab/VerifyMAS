# -*- coding: utf-8 -*-
# @File    : inference_zeroshot_multi_agent_json_openai.py
# @Desc    : Zero-shot joint inference for multi-agent failure attribution using OpenAI API.
#            Supports gpt-4o and gpt-4o-mini while keeping the prompt unchanged.

import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Iterable, Optional, Set, Tuple

from openai import OpenAI
import time

# =========================================================
# 1) failure templates
# =========================================================

FM_TYPE_TEMPLATES: Dict[str, str] = {
    "FM-1.1": "An agent deviated from the specified task requirements in the trajectory.",
    "FM-1.2": "An agent acted outside its designated role in the trajectory.",
    "FM-1.3": "An agent introduced redundant or unnecessary steps in the trajectory.",
    "FM-1.4": "An agent overlooked important context from earlier conversation history in the trajectory.",
    "FM-1.5": "An agent failed to maintain a proper termination condition in the trajectory.",
    "FM-2.1": "An agent repeated a task that had already been handled in the trajectory.",
    "FM-2.2": "An agent made a request unclear or ambiguous for other agents in the trajectory.",
    "FM-2.3": "An agent drifted away from the main goal of the task in the trajectory.",
    "FM-2.4": "An agent omitted important information needed by other agents in the trajectory.",
    "FM-2.5": "An agent overlooked input or corrections from other agents in the trajectory.",
    "FM-2.6": "An agent used reasoning that was inconsistent with earlier statements in the trajectory.",
    "FM-3.1": "An agent ended the task before all requirements were met in the trajectory.",
    "FM-3.2": "An agent skipped necessary verification or validation steps in the trajectory.",
    "FM-3.3": "An agent performed flawed or unreliable verification in the trajectory.",
}

TOKEN_LABEL_MAP = {
    "A": "entail",
    "B": "neutral",
    "C": "contradict",
}
CANONICAL_LABELS = {"A", "B", "C"}

# Keep this prompt unchanged from your current code.
SYSTEM_PROMPT_ZERO_SHOT_JOINT = """You are a careful verifier for multi-agent trajectory failure attribution.

You will be given:
1. A trajectory
2. A failure hypothesis
3. The list of candidate agents appearing in the trajectory

Your task is to jointly predict:
- label: one of A, B, C
- agents: responsible agent(s) for the hypothesized failure

Label meanings:
A = entail
B = neutral
C = contradict

Decision rules:
- Choose A when the trajectory provides clear or reasonably strong evidence that the hypothesized failure occurred, and that it negatively affected, or likely affected, the final outcome, final answer, final decision, or successful task completion.
- Choose B when the evidence is mixed, incomplete, weak, or the impact on the final outcome is uncertain.
- Choose C when the trajectory provides clear evidence that the hypothesis is false or inconsistent with what actually happened.
- If the failure appears to have been minor, corrected later, or not connected to the final outcome, prefer B over A.
- Use C only when the hypothesis is clearly contradicted by the trajectory, not merely because support is weak.

Agent attribution rules:
- A failure may be caused by one agent or multiple agents.
- If label = A, output one JSON object for each likely responsible agent.
- Each JSON object should contain exactly one responsible agent in the "agents" list.
- Include an agent only when there is clear or reasonably strong evidence that this agent contributed to the hypothesized failure.
- Do not include agents merely because they are mentioned near the error.
- Do not include agents that only followed instructions from another faulty agent unless their own action also contributed to the failure.
- Only output agent names that appear exactly in the provided candidate agent list.
- If label = A but no responsible agent can be confidently identified, output one JSON object with "agents": [].
- If label is B or C, output exactly one JSON object with "agents": [].

Output format:
Return JSON objects only.
If there are multiple responsible agents, return multiple JSON objects, one per line.
Each JSON object must have exactly these keys:
{"label":"A","agents":["agent_name"]}

Valid examples:
{"label":"A","agents":["Planner"]}
{"label":"A","agents":["Solver"]}

{"label":"B","agents":[]}

{"label":"C","agents":[]}

Do not output explanations.
Do not output markdown.
Do not output extra text.
"""


# =========================================================
# 2) data classes
# =========================================================

@dataclass
class Candidate:
    trajectory_id: str
    error_type: str
    hypothesis: str
    agents: List[str]
    user_content: str


# =========================================================
# 3) jsonl utils
# =========================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] skip invalid line {i}: {e}")
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================================================
# 4) trajectory helpers
# =========================================================

def trajectory_to_text(sample: Dict[str, Any]) -> str:
    inp = sample.get("input", {}) if isinstance(sample.get("input", {}), dict) else {}
    history = inp.get("conversation_history", [])
    parts = []
    query = str(inp.get("query", "")).strip()
    parts.append("Query: " + query)
    for item in history:
        step = item.get("step", "")
        agent = item.get("agent_name", "UnknownAgent")
        content = str(item.get("content", "")).strip()
        parts.append(f"[Step {step}] Agent: {agent}\nContent: {content}")
    return "\n\n".join(parts)


def extract_agents(sample: Dict[str, Any]) -> List[str]:
    inp = sample.get("input", {}) if isinstance(sample.get("input", {}), dict) else {}
    history = inp.get("conversation_history", [])
    agents = []
    seen = set()
    for item in history:
        agent = item.get("agent_name")
        if agent and agent not in seen:
            seen.add(agent)
            agents.append(agent)
    return agents


def build_user_content(trajectory_text: str, hypothesis: str, agents: List[str]) -> str:
    agent_list_text = "\n".join([f"- {a}" for a in agents]) if agents else "- None"
    return (
        f"Trajectory:\n{trajectory_text}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        f"Candidate agents:\n{agent_list_text}\n\n"
        "Return JSON objects only. If there are multiple responsible agents, "
        "return multiple JSON objects, one per line. Use keys label and agents."
    )


def build_joint_candidates_for_sample(sample: Dict[str, Any]) -> List[Candidate]:
    trajectory_id = sample.get("id", "unknown_trajectory")
    trajectory_text = trajectory_to_text(sample)
    agents = extract_agents(sample)

    candidates = []
    for error_type, template in FM_TYPE_TEMPLATES.items():
        hypothesis = template
        user_content = build_user_content(trajectory_text, hypothesis, agents)
        candidates.append(
            Candidate(
                trajectory_id=trajectory_id,
                error_type=error_type,
                hypothesis=hypothesis,
                agents=agents,
                user_content=user_content,
            )
        )
    return candidates


# =========================================================
# 5) OpenAI generation + parse
# =========================================================

def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    raw = str(text).strip()
    if not raw:
        return []

    # Direct parse: one object or array of objects.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass

    # Fallback: scan balanced JSON objects.
    objects: List[Dict[str, Any]] = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = raw[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except Exception:
                        pass
                    start = None

    return objects


def normalize_label(raw_label: Any) -> Optional[str]:
    if raw_label is None:
        return None
    s = str(raw_label).strip().upper()
    if s in CANONICAL_LABELS:
        return s
    mapping = {
        "ENTAIL": "A",
        "ENTAILED": "A",
        "NEUTRAL": "B",
        "CONTRADICT": "C",
        "CONTRADICTION": "C",
        "CONTRADICTED": "C",
    }
    return mapping.get(s)


def _dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_agents(raw_agents: Any, valid_agents: List[str], label_token: Optional[str]) -> List[str]:
    if label_token != "A":
        return []

    if raw_agents is None:
        return []
    if isinstance(raw_agents, str):
        candidates = [raw_agents]
    elif isinstance(raw_agents, (list, tuple, set)):
        candidates = list(raw_agents)
    else:
        return []

    lower_map = {a.lower(): a for a in valid_agents}
    normalized: List[str] = []
    for x in candidates:
        if x is None:
            continue
        s = str(x).strip()
        if not s or s.lower() in {"null", "none", "unknown", "[]"}:
            continue

        if s in valid_agents:
            normalized.append(s)
        elif s.lower() in lower_map:
            normalized.append(lower_map[s.lower()])

    return _dedup_keep_order(normalized)


def parse_joint_json_output(text: str, valid_agents: List[str]) -> Dict[str, Any]:
    raw_text = str(text).strip()
    objects = extract_json_objects(raw_text)

    if not objects:
        return {
            "parse_ok": False,
            "raw_output": raw_text,
            "json_objects": [],
            "label_token": None,
            "label": None,
            "agent_names": [],
        }

    valid_objects = []
    labels: List[str] = []
    agent_names: List[str] = []

    for obj in objects:
        label_token = normalize_label(obj.get("label"))
        if label_token not in CANONICAL_LABELS:
            continue

        labels.append(label_token)
        raw_agents = obj.get("agents", None)
        if raw_agents is None and "agent" in obj:
            raw_agents = obj.get("agent")

        agent_names.extend(normalize_agents(raw_agents, valid_agents, label_token))
        valid_objects.append(obj)

    if not labels:
        return {
            "parse_ok": False,
            "raw_output": raw_text,
            "json_objects": objects,
            "label_token": None,
            "label": None,
            "agent_names": [],
            "structured": objects,
        }

    # If any A object appears, treat this hypothesis as entailed.
    # Otherwise B/C should have exactly one object, so use the first valid non-A label.
    final_label_token = "A" if "A" in labels else labels[0]
    final_agent_names = _dedup_keep_order(agent_names) if final_label_token == "A" else []
    label = TOKEN_LABEL_MAP[final_label_token] if final_label_token in TOKEN_LABEL_MAP else None

    return {
        "parse_ok": final_label_token in CANONICAL_LABELS,
        "raw_output": raw_text,
        "json_objects": valid_objects,
        "json_str": "\n".join(json.dumps(x, ensure_ascii=False) for x in valid_objects),
        "label_token": final_label_token,
        "label": label,
        "agent_names": final_agent_names,
        "structured": valid_objects,
    }


def _extract_response_text(resp: Any) -> str:
    # OpenAI Python SDK Responses API exposes output_text in recent versions.
    text = getattr(resp, "output_text", None)
    if isinstance(text, str):
        return text

    # Defensive fallback for SDK/object changes.
    try:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                t = getattr(content, "text", None)
                if isinstance(t, str):
                    chunks.append(t)
        if chunks:
            return "".join(chunks)
    except Exception:
        pass

    return str(resp)


def call_openai_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return _extract_response_text(resp)


def call_openai_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return call_openai_once(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_content=user_content,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                break
            sleep_s = retry_sleep * (2 ** attempt)
            print(f"[WARN] OpenAI call failed at attempt {attempt + 1}/{max_retries + 1}: {e}. sleep={sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI call failed after {max_retries + 1} attempts: {last_error}")


def batch_generate_joint_json_openai(
    client: OpenAI,
    model: str,
    candidates: List[Candidate],
    max_output_tokens: int = 128,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
    sleep_between_calls: float = 0.0,
    save_every: int = 0,
    tmp_output_jsonl: Optional[str] = None,
) -> List[Dict[str, Any]]:
    outputs_all: List[Dict[str, Any]] = []
    total = len(candidates)
    t0 = time.time()

    for idx, cand in enumerate(candidates):
        elapsed = max(time.time() - t0, 1e-6)
        speed = (idx + 1) / elapsed
        print(
            f"[INFO] OpenAI {model}: candidate {idx + 1}/{total}, "
            f"speed={speed:.3f}/s, tid={cand.trajectory_id}, error={cand.error_type}",
            flush=True,
        )

        raw_text = call_openai_with_retry(
            client=client,
            model=model,
            system_prompt=SYSTEM_PROMPT_ZERO_SHOT_JOINT,
            user_content=cand.user_content,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        parsed = parse_joint_json_output(raw_text, cand.agents)
        outputs_all.append(parsed)

        if save_every > 0 and tmp_output_jsonl and len(outputs_all) % save_every == 0:
            tmp_rows = []
            for c, out in zip(candidates[:len(outputs_all)], outputs_all):
                tmp_rows.append({
                    "trajectory_id": c.trajectory_id,
                    "error_type": c.error_type,
                    "hypothesis": c.hypothesis,
                    "raw_output": out.get("raw_output", ""),
                    "parsed": out,
                })
            write_jsonl(tmp_output_jsonl, tmp_rows)
            print(f"[INFO] saved temporary raw predictions to {tmp_output_jsonl}")

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return outputs_all


# =========================================================
# 6) zero-shot joint prediction logic
# =========================================================

def batch_infer_jsonl_zeroshot_joint_openai(
    samples: List[Dict[str, Any]],
    client: OpenAI,
    model: str,
    positive_labels: Optional[Set[str]] = None,
    max_output_tokens: int = 128,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
    sleep_between_calls: float = 0.0,
    save_every: int = 0,
    tmp_output_jsonl: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if positive_labels is None:
        positive_labels = {"entail"}

    all_candidates: List[Candidate] = []
    for sample in samples:
        all_candidates.extend(build_joint_candidates_for_sample(sample))

    if not all_candidates:
        return []

    print(f"[INFO] OpenAI zero-shot joint inference with {model}; total candidates={len(all_candidates)}")

    model_outputs = batch_generate_joint_json_openai(
        client=client,
        model=model,
        candidates=all_candidates,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        sleep_between_calls=sleep_between_calls,
        save_every=save_every,
        tmp_output_jsonl=tmp_output_jsonl,
    )

    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for cand, gen_out in zip(all_candidates, model_outputs):
        row = {
            "error_type": cand.error_type,
            "hypothesis": cand.hypothesis,
            "label": gen_out.get("label"),
            "label_token": gen_out.get("label_token"),
            "agent_names": gen_out.get("agent_names", []),
            "parse_ok": gen_out.get("parse_ok", False),
            "raw_output": gen_out.get("raw_output", ""),
            "json_str": gen_out.get("json_str"),
            "structured": gen_out.get("structured"),
            "model": model,
        }
        grouped.setdefault(cand.trajectory_id, []).append(row)

    sample_map = {s.get("id", "unknown_trajectory"): s for s in samples}
    results = []

    for sample in samples:
        trajectory_id = sample.get("id", "unknown_trajectory")
        src = sample_map.get(trajectory_id, {})
        all_rows = grouped.get(trajectory_id, [])

        selected_failure_types = []
        predicted_failures = []

        for row in all_rows:
            if row.get("label") in positive_labels:
                selected_failure_types.append(
                    {
                        "error_type": row["error_type"],
                        "hypothesis": row["hypothesis"],
                        "label": row["label"],
                        "label_token": row["label_token"],
                        "parse_ok": row["parse_ok"],
                        "raw_output": row["raw_output"],
                        "agent_names": row.get("agent_names", []),
                        "model": model,
                    }
                )
                for agent_name in row.get("agent_names", []):
                    predicted_failures.append(
                        {
                            "agent_name": agent_name,
                            "error_type": row["error_type"],
                            "hypothesis": row["hypothesis"],
                            "label": row["label"],
                            "label_token": row["label_token"],
                            "parse_ok": row["parse_ok"],
                            "model": model,
                        }
                    )

        query = None
        if isinstance(src.get("input", {}), dict):
            query = src["input"].get("query")

        results.append(
            {
                "trajectory_id": trajectory_id,
                "query": query,
                "selected_failure_types": selected_failure_types,
                "predicted_failures": predicted_failures,
                "all_scores": all_rows,
            }
        )

    return results


# =========================================================
# 7) evaluation
# =========================================================

def _evaluate_classwise_from_sets(
    all_gold_pred_sets: List[Tuple[Set[Any], Set[Any]]],
    label_space: Optional[Set[Any]] = None,
) -> Dict[str, float]:
    if label_space is None:
        label_space = set()
        for gold_set, pred_set in all_gold_pred_sets:
            label_space.update(gold_set)
            label_space.update(pred_set)
    else:
        label_space = set(label_space)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0
    num_classes = 0

    for label in sorted(label_space, key=str):
        tp = 0
        fp = 0
        fn = 0

        for gold_set, pred_set in all_gold_pred_sets:
            in_gold = label in gold_set
            in_pred = label in pred_set

            if in_gold and in_pred:
                tp += 1
            elif (not in_gold) and in_pred:
                fp += 1
            elif in_gold and (not in_pred):
                fn += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_i = (
            2 * precision_i * recall_i / (precision_i + recall_i)
            if (precision_i + recall_i) > 0
            else 0.0
        )

        macro_precision_sum += precision_i
        macro_recall_sum += recall_i
        macro_f1_sum += f1_i
        num_classes += 1

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    macro_precision = macro_precision_sum / num_classes if num_classes > 0 else 0.0
    macro_recall = macro_recall_sum / num_classes if num_classes > 0 else 0.0
    macro_f1 = macro_f1_sum / num_classes if num_classes > 0 else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "num_classes": num_classes,
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(micro_f1, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "precision": round(micro_precision, 6),
        "recall": round(micro_recall, 6),
        "f1": round(micro_f1, 6),
    }


def _get_gold_faults(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Supports both test-style {"output":{"faulty_agents":...}} and reward_model style.
    out = sample.get("output", {})
    if isinstance(out, dict) and isinstance(out.get("faulty_agents"), list):
        return out.get("faulty_agents", [])

    reward = sample.get("reward_model", {})
    gt = reward.get("ground_truth") if isinstance(reward, dict) else None
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except Exception:
            gt = None
    if isinstance(gt, dict) and isinstance(gt.get("faulty_agents"), list):
        return gt.get("faulty_agents", [])
    return []


def evaluate_tuple_f1(
    prediction_rows: List[Dict[str, Any]],
    source_samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    sample_map = {x.get("id", "unknown_trajectory"): x for x in source_samples}
    all_gold_pred_sets = []

    for row in prediction_rows:
        trajectory_id = row["trajectory_id"]
        gold_faults = _get_gold_faults(sample_map.get(trajectory_id, {}))

        gold_set = set()
        for g in gold_faults:
            agent_name = g.get("agent_name")
            error_type = g.get("error_type")
            if agent_name is not None and error_type is not None:
                gold_set.add((agent_name, error_type))

        pred_set = set()
        for p in row.get("predicted_failures", []):
            agent_name = p.get("agent_name")
            error_type = p.get("error_type")
            if agent_name is not None and error_type is not None:
                pred_set.add((agent_name, error_type))

        all_gold_pred_sets.append((gold_set, pred_set))

    return _evaluate_classwise_from_sets(all_gold_pred_sets)


def evaluate_tuple_failure(
    prediction_rows: List[Dict[str, Any]],
    source_samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    sample_map = {x.get("id", "unknown_trajectory"): x for x in source_samples}
    all_gold_pred_sets = []
    fixed_error_types = set(FM_TYPE_TEMPLATES.keys())

    for row in prediction_rows:
        trajectory_id = row["trajectory_id"]
        gold_faults = _get_gold_faults(sample_map.get(trajectory_id, {}))

        gold_set = set()
        for g in gold_faults:
            error_type = g.get("error_type")
            if error_type is not None:
                gold_set.add(error_type)

        pred_set = set()
        for p in row.get("selected_failure_types", []):
            error_type = p.get("error_type")
            if error_type is not None:
                pred_set.add(error_type)

        all_gold_pred_sets.append((gold_set, pred_set))

    return _evaluate_classwise_from_sets(all_gold_pred_sets, label_space=fixed_error_types)


def evaluate_tuple_agent(
    prediction_rows: List[Dict[str, Any]],
    source_samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    sample_map = {x.get("id", "unknown_trajectory"): x for x in source_samples}
    all_gold_pred_sets = []

    for row in prediction_rows:
        trajectory_id = row["trajectory_id"]
        gold_faults = _get_gold_faults(sample_map.get(trajectory_id, {}))

        gold_set = set()
        for g in gold_faults:
            agent_name = g.get("agent_name")
            if agent_name is not None:
                gold_set.add(agent_name)

        pred_set = set()
        for p in row.get("predicted_failures", []):
            agent_name = p.get("agent_name")
            if agent_name is not None:
                pred_set.add(agent_name)

        all_gold_pred_sets.append((gold_set, pred_set))

    return _evaluate_classwise_from_sets(all_gold_pred_sets)


# =========================================================
# 8) args / main
# =========================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot multi-agent JSON inference with OpenAI GPT-4o / GPT-4o-mini.")

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        choices=["gpt-4.1", "gpt-4o", "gpt-4o-mini"],
        help="OpenAI model name. Use gpt-4o for stronger quality, gpt-4o-mini for lower cost/latency.",
    )
    parser.add_argument("--input_jsonl", type=str, default="./data/whowhen.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="./outputs/predictions_openai_multi_agent_json.jsonl")
    parser.add_argument("--eval_with_gold", type=str2bool, default=True)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_output_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--sleep_between_calls", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=20, help="Save temporary raw prediction file every N candidates; set 0 to disable.")
    parser.add_argument("--tmp_output_jsonl", type=str, default="./outputs/tmp_openai_raw_predictions.jsonl")
    parser.add_argument("--openai_api_key", type=str, default="")
    parser.add_argument("--openai_base_url", type=str, default=None)

    return parser.parse_args()

def create_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)

def main():
    args = parse_args()

    # if not os.getenv("OPENAI_API_KEY"):
    #     raise EnvironmentError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='sk-...'")
    #
    # print(f"[INFO] loading OpenAI client; model={args.model}")
    # client = OpenAI()
    client = create_openai_client(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
    )

    print("[INFO] reading jsonl...")
    samples = read_jsonl(args.input_jsonl)
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"[INFO] loaded {len(samples)} trajectories")

    print("[INFO] running OpenAI zero-shot joint inference...")
    prediction_rows = batch_infer_jsonl_zeroshot_joint_openai(
        samples=samples,
        client=client,
        model=args.model,
        positive_labels={"entail"},
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        sleep_between_calls=args.sleep_between_calls,
        save_every=args.save_every,
        tmp_output_jsonl=args.tmp_output_jsonl,
    )

    print(f"[INFO] writing outputs to {args.output_jsonl}")
    write_jsonl(args.output_jsonl, prediction_rows)

    if args.eval_with_gold:
        metrics_pair = evaluate_tuple_f1(prediction_rows, samples)
        metrics_agent = evaluate_tuple_agent(prediction_rows, samples)
        metrics_failure = evaluate_tuple_failure(prediction_rows, samples)

        print("[INFO] tuple-level metrics:")
        print("Metric_pair", metrics_pair)
        print("Metric_agent", metrics_agent)
        print("Metric_failure", metrics_failure)

    print("[INFO] done.")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"running time: {end - start:.4f} second")
