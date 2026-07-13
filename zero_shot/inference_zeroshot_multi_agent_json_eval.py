# -*- coding: utf-8 -*-
# @File    : inference_zeroshot_joint_json.py
# @Desc    : Zero-shot joint inference for multi-agent failure attribution.
#            For each (trajectory, error_type), the LLM generates one JSON object:
#            {"label":"A|B|C", "agent":"<agent name>|null"}.

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Iterable, Optional, Set, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


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

# SYSTEM_PROMPT_ZERO_SHOT_JOINT = """You are a verifier for multi-agent failure attribution.
#
# Given a trajectory, a failure hypothesis, and candidate agents, predict:
# - label: A, B, or C
# - agents: responsible agent(s)
#
# Labels:
# A = the failure is supported and likely affected the task outcome.
# B = evidence is weak, incomplete, mixed, or impact is unclear.
# C = the hypothesis is clearly contradicted.
#
# Rules:
# - If label = A, output one JSON object per responsible agent.
# - Each JSON object must contain exactly one agent in "agents".
# - Include only agents with clear or reasonably strong evidence of contributing to the failure.
# - Use only names from the candidate agent list.
# - If label = A but no agent is identifiable, output {"label":"A","agents":[]}.
# - If label = B or C, output exactly one JSON object with "agents":[].
# - Do not include explanations.
#
# Output JSON objects only, one per line:
# {"label":"A","agents":["agent_name"]}
#
# Examples:
# {"label":"A","agents":["Planner"]}
# {"label":"A","agents":["Solver"]}
# {"label":"B","agents":[]}
# {"label":"C","agents":[]}
# """

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
    prompt_text: str


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
    history = sample.get("input", {}).get("conversation_history", [])
    parts = []
    query = str(sample.get("input", {}).get("query", "")).strip()
    parts.append("Query: " + query)
    for item in history:
        step = item.get("step", "")
        agent = item.get("agent_name", "UnknownAgent")
        content = str(item.get("content", "")).strip()
        parts.append(f"[Step {step}] Agent: {agent}\nContent: {content}")
    return "\n\n".join(parts)


def extract_agents(sample: Dict[str, Any]) -> List[str]:
    history = sample.get("input", {}).get("conversation_history", [])
    agents = []
    seen = set()
    for item in history:
        agent = item.get("agent_name")
        if agent and agent not in seen:
            seen.add(agent)
            agents.append(agent)
    return agents


def build_joint_prompt(tokenizer, trajectory_text: str, hypothesis: str, agents: List[str]) -> str:
    agent_list_text = "\n".join([f"- {a}" for a in agents]) if agents else "- None"

    user_content = (
        f"Trajectory:\n{trajectory_text}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        f"Candidate agents:\n{agent_list_text}\n\n"
        "Return JSON objects only. If there are multiple responsible agents, "
        "return multiple JSON objects, one per line. Use keys label and agents."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ZERO_SHOT_JOINT},
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"System: {SYSTEM_PROMPT_ZERO_SHOT_JOINT}\n\nUser: {user_content}\n\nAssistant:"


def build_joint_candidates_for_sample(sample: Dict[str, Any], tokenizer) -> List[Candidate]:
    trajectory_id = sample.get("id", "unknown_trajectory")
    trajectory_text = trajectory_to_text(sample)
    agents = extract_agents(sample)

    candidates = []
    for error_type, template in FM_TYPE_TEMPLATES.items():
        hypothesis = template
        prompt_text = build_joint_prompt(tokenizer, trajectory_text, hypothesis, agents)
        candidates.append(
            Candidate(
                trajectory_id=trajectory_id,
                error_type=error_type,
                hypothesis=hypothesis,
                agents=agents,
                prompt_text=prompt_text,
            )
        )
    return candidates


# =========================================================
# 5) model loading
# =========================================================

def resolve_dtype(dtype_str: str):
    dtype_str = dtype_str.lower()
    if dtype_str == "auto":
        return "auto"
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def load_qwen_from_hf(
    model_name_or_path: str,
    adapter_name_or_path: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
    dtype: str = "auto",
):
    torch_dtype = resolve_dtype(dtype)

    tokenizer_source = tokenizer_name_or_path or adapter_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_name_or_path:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "adapter_name_or_path was provided, but peft is not installed. "
                "Please run: pip install peft"
            )
        model = PeftModel.from_pretrained(model, adapter_name_or_path)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 6) zero-shot generation + parse
# =========================================================

def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Extract one or more balanced JSON objects from model output.

    The new prompt asks for JSONL-style output, e.g.:
        {"label":"A","agents":["Planner"]}
        {"label":"A","agents":["Solver"]}

    This helper is robust to extra whitespace and also supports a single JSON
    object or a JSON array of objects.
    """
    raw = str(text).strip()
    if not raw:
        return []

    # Direct parse first: one object or an array of objects.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass

    # Fallback: scan balanced {...} blocks.
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
    """Normalize both new field agents=[...] and old field agent=... ."""
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
        # New format: {"label":"A", "agents":["Planner"]}
        # Backward-compatible fallback: {"label":"A", "agent":"Planner"}
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

    # For the new prompt, multiple objects should all be A for positive cases.
    # If any A object appears, treat this hypothesis as entailed and collect all agents.
    # Otherwise use the first valid non-A label, because B/C should have exactly one object.
    if "A" in labels:
        final_label_token = "A"
    else:
        final_label_token = labels[0]

    final_agent_names = _dedup_keep_order(agent_names) if final_label_token == "A" else []
    parse_ok = final_label_token in CANONICAL_LABELS
    label = TOKEN_LABEL_MAP[final_label_token] if final_label_token in TOKEN_LABEL_MAP else None

    return {
        "parse_ok": parse_ok,
        "raw_output": raw_text,
        "json_objects": valid_objects,
        "json_str": "\n".join(json.dumps(x, ensure_ascii=False) for x in valid_objects),
        "label_token": final_label_token,
        "label": label,
        "agent_names": final_agent_names,
        "structured": valid_objects,
    }


@torch.no_grad()
def batch_generate_joint_json(
    model,
    tokenizer,
    prompts: List[str],
    candidate_agents: List[List[str]],
    batch_size: int = 4,
    max_input_length: int = 8192,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    device = get_model_device(model)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    outputs_all: List[Dict[str, Any]] = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(prompts), batch_size), 1):
        batch_prompts = prompts[start:start + batch_size]
        batch_agent_lists = candidate_agents[start:start + batch_size]
        end = min(start + batch_size, len(prompts))
        print(f"[INFO] Processing batch {batch_idx}/{total_batches}, samples {start}~{end - 1}")

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(getattr(tokenizer, "model_max_length", max_input_length), max_input_length),
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]

        gen_kwargs = dict(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        generated = model.generate(**gen_kwargs)
        generated_only = generated[:, input_len:]
        decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

        for text, valid_agents in zip(decoded, batch_agent_lists):
            parsed = parse_joint_json_output(text, valid_agents)
            outputs_all.append(parsed)

    tokenizer.padding_side = old_padding_side
    return outputs_all


# =========================================================
# 7) selection helpers
# =========================================================

def select_by_labels(rows: List[Dict[str, Any]], positive_labels: Set[str]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("label") in positive_labels]


# =========================================================
# 8) zero-shot joint prediction logic
# =========================================================

def batch_infer_jsonl_zeroshot_joint(
    samples: List[Dict[str, Any]],
    model,
    tokenizer,
    model_batch_size: int = 4,
    positive_labels: Optional[Set[str]] = None,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    For each trajectory and each failure type, do one zero-shot generation:
    {"label":"A|B|C", "agents":["<agent>"]} (possibly multiple JSON objects)
    """
    if positive_labels is None:
        positive_labels = {"entail"}

    all_candidates: List[Candidate] = []
    for sample in samples:
        all_candidates.extend(build_joint_candidates_for_sample(sample, tokenizer))

    if not all_candidates:
        return []

    print("[INFO] Zero-shot joint inference...")
    prompts = [c.prompt_text for c in all_candidates]
    candidate_agents = [c.agents for c in all_candidates]

    model_outputs = batch_generate_joint_json(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        candidate_agents=candidate_agents,
        batch_size=model_batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
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
                        }
                    )

        results.append(
            {
                "trajectory_id": trajectory_id,
                "query": src.get("input", {}).get("query"),
                "selected_failure_types": selected_failure_types,
                "predicted_failures": predicted_failures,
                "all_scores": all_rows,
            }
        )

    return results


# =========================================================
# 9) evaluation
# =========================================================
#

from typing import List, Dict, Any, Optional, Set, Tuple

def _evaluate_classwise_from_sets(
    all_gold_pred_sets: List[Tuple[Set[Any], Set[Any]]],
    label_space: Optional[Set[Any]] = None,
) -> Dict[str, float]:
    """
    all_gold_pred_sets:
        A list of (gold_set, pred_set), one tuple per sample.

    label_space:
        If provided, use this fixed class space for macro averaging.
        If None, infer from the union of gold/pred labels appearing in the data.

    Micro:
        Aggregate TP/FP/FN over all classes first, then compute metrics.

    Macro:
        Compute precision/recall/F1 for each class independently, then average
        over classes (class-wise macro).
    """
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


def evaluate_tuple_f1(
    prediction_rows: List[Dict[str, Any]],
    source_samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Pair-level evaluation on classes = (agent_name, error_type)
    Macro is class-wise over pair classes.
    """
    sample_map = {x.get("id", "unknown_trajectory"): x for x in source_samples}
    all_gold_pred_sets = []

    for row in prediction_rows:
        trajectory_id = row["trajectory_id"]
        src = sample_map.get(trajectory_id, {}).get("output", {})
        gold_faults = src.get("faulty_agents", [])

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
    """
    Failure-type-level evaluation.
    Macro is class-wise over error_type classes.
    """
    sample_map = {x.get("id", "unknown_trajectory"): x for x in source_samples}
    all_gold_pred_sets = []

    # Optional: fixed 14-class macro over all predefined error modes
    fixed_error_types = {
        "FM-1.1", "FM-1.2", "FM-1.3", "FM-1.4", "FM-1.5",
        "FM-2.1", "FM-2.2", "FM-2.3", "FM-2.4", "FM-2.5", "FM-2.6",
        "FM-3.1", "FM-3.2", "FM-3.3",
    }

    for row in prediction_rows:
        trajectory_id = row["trajectory_id"]
        src = sample_map.get(trajectory_id, {}).get("output", {})
        gold_faults = src.get("faulty_agents", [])

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

    return _evaluate_classwise_from_sets(
        all_gold_pred_sets,
        label_space=fixed_error_types,
    )


def evaluate_tuple_agent(
    prediction_rows: List[Dict[str, Any]],
    source_samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Agent-level evaluation.
    Macro is class-wise over agent_name classes appearing in gold/pred.
    """
    sample_map = {x.get("id", "unknown_trajectory"): x for x in source_samples}
    all_gold_pred_sets = []

    for row in prediction_rows:
        trajectory_id = row["trajectory_id"]
        src = sample_map.get(trajectory_id, {}).get("output", {})
        gold_faults = src.get("faulty_agents", [])

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
#
# =========================================================
# 10) args
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
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--model_name_or_path",
    #     type=str,
    #     default="Qwen/Qwen2.5-7B-Instruct",
    #     help="HF base model id or local base model path"
    # )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HF base model id or local base model path"
    )

    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default=None,
        help="Optional LoRA adapter or saved checkpoint path"
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Optional tokenizer path; defaults to adapter path, then model path"
    )
    parser.add_argument(
        "--use_single_checkpoint_dir",
        type=str2bool,
        default="False",
        help="If true, load model+tokenizer directly from --adapter_name_or_path and ignore base model."
    )

    parser.add_argument("--input_jsonl", type=str, default="./data/test_aegis.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="./outputs/predictions_zeroshot_joint.jsonl")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--model_batch_size", type=int, default=8)
    parser.add_argument("--eval_with_gold", type=str2bool, default=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)

    return parser.parse_args()


# =========================================================
# 11) main
# =========================================================

def main():
    args = parse_args()

    print("[INFO] loading model...")
    if args.use_single_checkpoint_dir:
        if not args.adapter_name_or_path:
            raise ValueError("--use_single_checkpoint_dir=true requires --adapter_name_or_path to be set")
        tokenizer = AutoTokenizer.from_pretrained(
            args.adapter_name_or_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.adapter_name_or_path,
            torch_dtype=resolve_dtype(args.dtype),
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer, model = load_qwen_from_hf(
            model_name_or_path=args.model_name_or_path,
            adapter_name_or_path=args.adapter_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            dtype=args.dtype,
        )

    model.eval()

    print("[INFO] reading jsonl...")
    samples = read_jsonl(args.input_jsonl)
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"[INFO] loaded {len(samples)} trajectories")

    print("[INFO] running zero-shot joint inference...")
    prediction_rows = batch_infer_jsonl_zeroshot_joint(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        model_batch_size=args.model_batch_size,
        positive_labels={"entail"},
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
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
    # start = time.time()
    main()
    # end = time.time()
    # print(f"running time: {end - start:.4f} second")