# -*- coding: utf-8 -*-
# @Date    :  4/28/2026 
# version： Python 3.7.8
# @File : inference_zeroshot_mas_qa_json.py.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @File: inference_zeroshot_mas_qa_json_no_filter.py

import os
import json
import argparse
from typing import Dict, List, Any, Iterable, Optional, Set, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


MAS_QA_PROMPT_TEMPLATE = """## ROLE AND GOAL
You are a meticulous Multi-Agent System (MAS) Quality Assurance analyst.

Your task is to analyze a multi-agent conversation trajectory and identify faulty agent-error pairs using an ERROR-FIRST diagnostic process.

You must first detect what error type(s) occur in the whole trajectory, and then attribute each detected error type to the agent who introduced, caused, or most directly contributed to it.

Do NOT start by judging each agent independently.

## ERROR DEFINITIONS WITH EXAMPLES
You MUST use the exact error codes provided below.

### Functional Mistakes (FM-1.x - Task Execution Errors):
- FM-1.1: **Task specification deviation** - Agent deviates from specified task requirements.
- FM-1.2: **Role specification deviation** - Agent acts outside its designated role.
- FM-1.3: **Add redundant steps** - Agent adds unnecessary or duplicate steps.
- FM-1.4: **Remove conversation history** - Agent ignores or removes important context from previous turns.
- FM-1.5: **Remove termination conditions** - Agent fails to define proper stopping criteria.

### Functional Mistakes (FM-2.x - Communication & Coordination Errors):
- FM-2.1: **Repeat handled tasks** - Agent redundantly handles already completed tasks.
- FM-2.2: **Make request ambiguous** - Agent provides unclear or confusing instructions to other agents.
- FM-2.3: **Deviate from main goal** - Agent pursues objectives unrelated to the main task.
- FM-2.4: **Hide important information** - Agent withholds crucial information needed by other agents.
- FM-2.5: **Ignore other agents** - Agent fails to consider input, corrections, or questions from other agents.
- FM-2.6: **Inconsistent reasoning** - Agent's logic contradicts its own previous statements.

### Functional Mistakes (FM-3.x - Quality & Verification Errors):
- FM-3.1: **Premature termination** - Agent stops or declares the task complete before all requirements are met.
- FM-3.2: **Remove verification steps** - Agent skips necessary validation or testing steps.
- FM-3.3: **Incorrect verification** - Agent performs flawed or wrong verification.

## ANALYSIS WORKFLOW
Follow these steps internally only. Do not output reasoning.

1. Understand the full trajectory, including the user task, agent interactions, key decisions, and final outcome.
2. Detect which error type(s) are clearly supported at the trajectory level.
3. For each detected error, identify the most directly responsible agent.
4. Output only clear faulty agent-error pairs. If the evidence is weak or the responsible agent is unclear, do not include the pair.

## REQUIRED OUTPUT FORMAT
Your entire response must be a single valid JSON object.
Do not output any reasoning.
Do not output any text before or after the JSON object.
Do not output markdown fences.
Do not output <think>...</think>.
If you are unsure, still output a JSON object only.

JSON Format:
{{"faulty_agents": [{{"agent_name": "XXX", "error_type": "FM-X.X"}}]}}

Examples:
{{"faulty_agents": [{{"agent_name": "XXX1", "error_type": "FM-1.1"}}, {{"agent_name": "XXX2", "error_type": "FM-3.2"}}]}}
{{"faulty_agents": []}}

## CONVERSATION TO ANALYZE:
\"\"\"
{conversation_text}
\"\"\"

## YOUR ANALYSIS (JSON ONLY):
"""


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


def trajectory_to_text(sample: Dict[str, Any]) -> str:
    input_obj = sample.get("input", {})
    history = input_obj.get("conversation_history", [])

    parts = []

    query = str(input_obj.get("query", "")).strip()
    if query:
        parts.append("Query:\n" + query)

    for item in history:
        step = item.get("step", "")
        agent = item.get("agent_name", "UnknownAgent")
        content = str(item.get("content", "")).strip()
        parts.append(f"[Step {step}] Agent: {agent}\nContent:\n{content}")

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


def build_full_prompt(tokenizer, trajectory_text: str) -> str:
    user_content = MAS_QA_PROMPT_TEMPLATE.format(
        conversation_text=trajectory_text
    )

    messages = [
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return user_content


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
            raise ImportError("Please install peft: pip install peft")

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


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text).strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

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
                            return obj
                    except Exception:
                        return None

    return None


def normalize_error_type(error_type: Any) -> Optional[str]:
    if error_type is None:
        return None

    s = str(error_type).strip()

    if not s:
        return None

    # 只做轻量格式统一，不做合法类别过滤
    s = s.upper()
    s = s.replace(" ", "")
    s = s.replace("_", "-")

    # FM1.1 -> FM-1.1
    if s.startswith("FM") and not s.startswith("FM-"):
        s = "FM-" + s[2:]

    return s


def parse_faulty_agents_output(
    text: str,
    valid_agents: List[str],
) -> Dict[str, Any]:
    raw_text = str(text).strip()
    obj = extract_first_json_object(raw_text)

    if not isinstance(obj, dict):
        return {
            "parse_ok": False,
            "raw_output": raw_text,
            "json_obj": None,
            "faulty_agents": [],
        }

    raw_faults = obj.get("faulty_agents", [])
    if not isinstance(raw_faults, list):
        raw_faults = []

    valid_agent_set = set(valid_agents)
    lower_agent_map = {a.lower(): a for a in valid_agents}

    normalized = []
    seen = set()

    for item in raw_faults:
        if not isinstance(item, dict):
            continue

        agent_name = item.get("agent_name")
        error_type = normalize_error_type(item.get("error_type"))

        if agent_name is None or error_type is None:
            continue

        agent_name = str(agent_name).strip()

        # agent name 仍然对齐 candidate，避免模型 hallucinate agent
        if agent_name not in valid_agent_set:
            agent_name = lower_agent_map.get(agent_name.lower())

        if not agent_name:
            continue

        key = (agent_name, error_type)

        if key not in seen:
            seen.add(key)
            normalized.append({
                "agent_name": agent_name,
                "error_type": error_type,
            })

    return {
        "parse_ok": True,
        "raw_output": raw_text,
        "json_obj": obj,
        "faulty_agents": normalized,
    }


@torch.no_grad()
def batch_generate_faulty_agents(
    model,
    tokenizer,
    prompts: List[str],
    candidate_agents: List[List[str]],
    batch_size: int = 4,
    max_input_length: int = 8192,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:

    device = get_model_device(model)

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    outputs_all = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(prompts), batch_size), 1):
        batch_prompts = prompts[start:start + batch_size]
        batch_agent_lists = candidate_agents[start:start + batch_size]
        end = min(start + batch_size, len(prompts))

        print(f"[INFO] Processing batch {batch_idx}/{total_batches}, samples {start}~{end - 1}")

        max_len = min(
            getattr(tokenizer, "model_max_length", max_input_length),
            max_input_length,
        )

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
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
        decoded = tokenizer.batch_decode(
            generated_only,
            skip_special_tokens=True,
        )

        for text, valid_agents in zip(decoded, batch_agent_lists):
            parsed = parse_faulty_agents_output(text, valid_agents)
            outputs_all.append(parsed)

    tokenizer.padding_side = old_padding_side

    return outputs_all


def batch_infer_jsonl_zeroshot_joint(
    samples: List[Dict[str, Any]],
    model,
    tokenizer,
    model_batch_size: int = 4,
    max_input_length: int = 8192,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:

    prompts = []
    candidate_agents = []
    trajectory_ids = []

    for sample in samples:
        trajectory_id = sample.get("id", "unknown_trajectory")
        trajectory_text = trajectory_to_text(sample)
        agents = extract_agents(sample)
        prompt_text = build_full_prompt(tokenizer, trajectory_text)

        prompts.append(prompt_text)
        candidate_agents.append(agents)
        trajectory_ids.append(trajectory_id)

    if not prompts:
        return []

    model_outputs = batch_generate_faulty_agents(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        candidate_agents=candidate_agents,
        batch_size=model_batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )

    results = []

    for sample, trajectory_id, gen_out in zip(samples, trajectory_ids, model_outputs):
        predicted_failures = gen_out.get("faulty_agents", [])

        selected_failure_types = []
        seen_error_types = set()

        for p in predicted_failures:
            error_type = p.get("error_type")
            if error_type and error_type not in seen_error_types:
                seen_error_types.add(error_type)
                selected_failure_types.append({
                    "error_type": error_type,
                })

        results.append({
            "trajectory_id": trajectory_id,
            "query": sample.get("input", {}).get("query"),
            "selected_failure_types": selected_failure_types,
            "predicted_failures": predicted_failures,
            "parse_ok": gen_out.get("parse_ok", False),
            "raw_output": gen_out.get("raw_output", ""),
            "json_obj": gen_out.get("json_obj"),
        })

    return results


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
        tp = fp = fn = 0

        for gold_set, pred_set in all_gold_pred_sets:
            in_gold = label in gold_set
            in_pred = label in pred_set

            if in_gold and in_pred:
                tp += 1
            elif not in_gold and in_pred:
                fp += 1
            elif in_gold and not in_pred:
                fn += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision_i = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall_i = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_i = (
            2 * precision_i * recall_i / (precision_i + recall_i)
            if precision_i + recall_i > 0
            else 0.0
        )

        macro_precision_sum += precision_i
        macro_recall_sum += recall_i
        macro_f1_sum += f1_i
        num_classes += 1

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "num_classes": num_classes,
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(micro_f1, 6),
        "macro_precision": round(macro_precision_sum / num_classes, 6) if num_classes else 0.0,
        "macro_recall": round(macro_recall_sum / num_classes, 6) if num_classes else 0.0,
        "macro_f1": round(macro_f1_sum / num_classes, 6) if num_classes else 0.0,
        "precision": round(micro_precision, 6),
        "recall": round(micro_recall, 6),
        "f1": round(micro_f1, 6),
    }


def evaluate_tuple_f1(prediction_rows, source_samples):
    sample_map = {x.get("id", "unknown_trajectory"): x for x in source_samples}
    all_gold_pred_sets = []

    for row in prediction_rows:
        src = sample_map.get(row["trajectory_id"], {}).get("output", {})
        gold_faults = src.get("faulty_agents", [])

        gold_set = {
            (g.get("agent_name"), normalize_error_type(g.get("error_type")))
            for g in gold_faults
            if g.get("agent_name") is not None and g.get("error_type") is not None
        }

        pred_set = {
            (p.get("agent_name"), normalize_error_type(p.get("error_type")))
            for p in row.get("predicted_failures", [])
            if p.get("agent_name") is not None and p.get("error_type") is not None
        }

        all_gold_pred_sets.append((gold_set, pred_set))

    return _evaluate_classwise_from_sets(all_gold_pred_sets)


def evaluate_tuple_failure(prediction_rows, source_samples):
    sample_map = {x.get("id", "unknown_trajectory"): x for x in source_samples}
    all_gold_pred_sets = []

    for row in prediction_rows:
        src = sample_map.get(row["trajectory_id"], {}).get("output", {})
        gold_faults = src.get("faulty_agents", [])

        gold_set = {
            normalize_error_type(g.get("error_type"))
            for g in gold_faults
            if g.get("error_type") is not None
        }

        pred_set = {
            normalize_error_type(p.get("error_type"))
            for p in row.get("selected_failure_types", [])
            if p.get("error_type") is not None
        }

        all_gold_pred_sets.append((gold_set, pred_set))

    return _evaluate_classwise_from_sets(all_gold_pred_sets)


def evaluate_tuple_agent(prediction_rows, source_samples):
    sample_map = {x.get("id", "unknown_trajectory"): x for x in source_samples}
    all_gold_pred_sets = []

    for row in prediction_rows:
        src = sample_map.get(row["trajectory_id"], {}).get("output", {})
        gold_faults = src.get("faulty_agents", [])

        gold_set = {
            g.get("agent_name")
            for g in gold_faults
            if g.get("agent_name") is not None
        }

        pred_set = {
            p.get("agent_name")
            for p in row.get("predicted_failures", [])
            if p.get("agent_name") is not None
        }

        all_gold_pred_sets.append((gold_set, pred_set))

    return _evaluate_classwise_from_sets(all_gold_pred_sets)


def evaluate_exact_match(prediction_rows, source_samples):
    sample_map = {x.get("id", "unknown_trajectory"): x for x in source_samples}

    total = 0
    exact = 0

    for row in prediction_rows:
        src = sample_map.get(row["trajectory_id"], {}).get("output", {})
        gold_faults = src.get("faulty_agents", [])

        gold_set = {
            (g.get("agent_name"), normalize_error_type(g.get("error_type")))
            for g in gold_faults
            if g.get("agent_name") is not None and g.get("error_type") is not None
        }

        pred_set = {
            (p.get("agent_name"), normalize_error_type(p.get("error_type")))
            for p in row.get("predicted_failures", [])
            if p.get("agent_name") is not None and p.get("error_type") is not None
        }

        total += 1
        if gold_set == pred_set:
            exact += 1

    return {
        "total": total,
        "exact": exact,
        "exact_match": round(exact / total, 6) if total > 0 else 0.0,
    }


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

    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--use_single_checkpoint_dir", type=str2bool, default=False)

    parser.add_argument("--input_jsonl", type=str, default="./data/whowhen.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="./outputs/predictions_mas_qa.jsonl")

    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--model_batch_size", type=int, default=2)
    parser.add_argument("--eval_with_gold", type=str2bool, default=True)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] loading model...")

    if args.use_single_checkpoint_dir:
        if not args.adapter_name_or_path:
            raise ValueError("--use_single_checkpoint_dir=true requires --adapter_name_or_path")

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

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()

    else:
        tokenizer, model = load_qwen_from_hf(
            model_name_or_path=args.model_name_or_path,
            adapter_name_or_path=args.adapter_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            dtype=args.dtype,
        )

    print("[INFO] reading jsonl...")
    samples = read_jsonl(args.input_jsonl)

    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[:args.max_samples]

    print(f"[INFO] loaded {len(samples)} trajectories")

    print("[INFO] running inference...")
    prediction_rows = batch_infer_jsonl_zeroshot_joint(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        model_batch_size=args.model_batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )

    print(f"[INFO] writing outputs to {args.output_jsonl}")
    write_jsonl(args.output_jsonl, prediction_rows)

    if args.eval_with_gold:
        print("[INFO] evaluation metrics:")
        print("Metric_pair:", evaluate_tuple_f1(prediction_rows, samples))
        print("Metric_agent:", evaluate_tuple_agent(prediction_rows, samples))
        print("Metric_failure:", evaluate_tuple_failure(prediction_rows, samples))
        print("Metric_exact_match:", evaluate_exact_match(prediction_rows, samples))

    print("[INFO] done.")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"running time: {end - start:.4f} second")