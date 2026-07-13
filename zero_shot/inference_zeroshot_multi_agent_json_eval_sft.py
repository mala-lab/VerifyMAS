# -*- coding: utf-8 -*-
# @File    : inference_zeroshot_multi_agent_json_eval_sft.py
# @Desc    : Direct-generation inference for trained FSDP SFT checkpoints.
#            - No cls_head.
#            - Directly generates JSON/JSONL: {"label":"A/B/C","agents":[...]}
#            - Supports checkpoints saved as model.pt by fsdp_direct_full_json_generation_sft_trainer.py
#            - Keeps the previous three evaluations: pair, agent, failure.

import os
import re
import json
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    PeftModel = None
    PEFT_AVAILABLE = False


# =========================================================
# 1) Failure templates
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

SYSTEM_PROMPT_DIRECT_GENERATION = """You are a careful verifier for multi-agent trajectory failure attribution.

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
Do not output thinking, reasoning, analysis, or <think>...</think>.
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
# 2) Data classes
# =========================================================

@dataclass
class Candidate:
    trajectory_id: str
    error_type: Optional[str]
    hypothesis: Optional[str]
    agents: List[str]
    prompt_text: str
    query: Optional[str] = None


@dataclass
class Runtime:
    tokenizer: Any
    model: Any
    device: torch.device
    checkpoint_meta: Optional[Dict[str, Any]] = None


# =========================================================
# 3) Model wrapper for FSDP model.pt loading
# =========================================================

class DirectJSONGenerationLMModel(nn.Module):
    """
    Same wrapper structure as the direct-generation FSDP trainer.
    The wrapper is only needed to load model.pt correctly.
    Inference calls wrapper.base_model.generate().
    """

    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        self.base_model = base_model
        self.lm_head = self.base_model.get_output_embeddings()
        if self.lm_head is None:
            raise ValueError("base_model.get_output_embeddings() returned None")


# =========================================================
# 4) IO helpers
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
                print(f"[WARN] skip invalid jsonl line {i}: {e}")
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json_if_exists(path: str) -> Optional[Any]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# =========================================================
# 5) Prompt / candidate helpers
# =========================================================

def trajectory_to_text(sample: Dict[str, Any]) -> str:
    inp = sample.get("input", {}) if isinstance(sample.get("input", {}), dict) else {}
    history = inp.get("conversation_history", [])
    parts = []

    query = str(inp.get("query", "")).strip()
    if query:
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
    agents, seen = [], set()

    for item in history:
        agent = item.get("agent_name")
        if agent and agent not in seen:
            seen.add(agent)
            agents.append(agent)

    return agents


def normalize_agent_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                value = parsed
            else:
                value = re.split(r"[\n,;]+", text)
        except Exception:
            value = re.split(r"[\n,;]+", text)

    if not isinstance(value, (list, tuple, set)):
        return []

    out, seen = [], set()
    for x in value:
        if x is None:
            continue
        s = str(x).strip()
        s = re.sub(r"^[-*\d\.\)\s]+", "", s).strip()
        if not s or s.lower() in {"none", "null", "__none__"}:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)

    return out


def parse_candidate_agents_from_prompt(prompt: str) -> List[str]:
    prompt = str(prompt)
    m = re.search(r"Candidate agents\s*:\s*(.*)", prompt, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []

    block = m.group(1)
    stop_patterns = [r"\n\s*Return\b", r"\n\s*Output\b", r"\n\s*Answer\b", r"\n\s*Label meanings\b"]
    stop = len(block)
    for pat in stop_patterns:
        sm = re.search(pat, block, flags=re.IGNORECASE)
        if sm:
            stop = min(stop, sm.start())
    return normalize_agent_list(block[:stop])


def extract_agents_from_prompt_row(row: Dict[str, Any], prompt: str) -> List[str]:
    for key in ["candidate_agents", "agents", "agent_list", "candidate_agent_list"]:
        if key in row:
            agents = normalize_agent_list(row.get(key))
            if agents:
                return agents
    if isinstance(row.get("input"), dict):
        for key in ["candidate_agents", "agents", "agent_list"]:
            if key in row["input"]:
                agents = normalize_agent_list(row["input"].get(key))
                if agents:
                    return agents
    return parse_candidate_agents_from_prompt(prompt)


def build_direct_prompt(tokenizer, trajectory_text: str, hypothesis: str, agents: List[str], disable_thinking: bool = True) -> str:
    agent_list_text = "\n".join([f"- {a}" for a in agents]) if agents else "- None"

    user_content = (
        f"Trajectory:\n{trajectory_text}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        f"Candidate agents:\n{agent_list_text}\n\n"
        "Return JSON objects only. If there are multiple responsible agents, "
        "return multiple JSON objects, one per line. Use keys label and agents."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DIRECT_GENERATION},
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not disable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return f"System: {SYSTEM_PROMPT_DIRECT_GENERATION}\n\nUser: {user_content}\n\nAssistant:"


def build_candidates(samples: List[Dict[str, Any]], tokenizer, input_format: str, disable_thinking: bool = True) -> List[Candidate]:
    candidates: List[Candidate] = []

    if input_format == "prompt":
        for i, row in enumerate(samples):
            prompt = str(row.get("prompt", ""))
            if not prompt:
                continue
            agents = extract_agents_from_prompt_row(row, prompt)
            tid = str(row.get("id", row.get("trajectory_id", f"sample_{i}")))
            candidates.append(
                Candidate(
                    trajectory_id=tid,
                    error_type=row.get("error_type"),
                    hypothesis=row.get("hypothesis"),
                    agents=agents,
                    prompt_text=prompt,
                    query=row.get("query"),
                )
            )
        return candidates

    if input_format in {"aegis", "whowhen"}:
        for sample in samples:
            tid = str(sample.get("id", sample.get("trajectory_id", "unknown_trajectory")))
            trajectory_text = trajectory_to_text(sample)
            agents = extract_agents(sample)
            query = None
            if isinstance(sample.get("input"), dict):
                query = sample["input"].get("query")
            for error_type, hypothesis in FM_TYPE_TEMPLATES.items():
                prompt = build_direct_prompt(
                    tokenizer=tokenizer,
                    trajectory_text=trajectory_text,
                    hypothesis=hypothesis,
                    agents=agents,
                    disable_thinking=disable_thinking,
                )
                candidates.append(
                    Candidate(
                        trajectory_id=tid,
                        error_type=error_type,
                        hypothesis=hypothesis,
                        agents=agents,
                        prompt_text=prompt,
                        query=query,
                    )
                )
        return candidates

    raise ValueError(f"Unsupported input_format: {input_format}")


# =========================================================
# 6) Loading helpers
# =========================================================

def resolve_dtype(dtype_str: str):
    dtype_str = str(dtype_str).lower()
    if dtype_str == "auto":
        return "auto"
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ["module.", "_fsdp_wrapped_module."]
    out = {}
    for k, v in state_dict.items():
        kk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p):]
                    changed = True
        out[kk] = v
    return out


def infer_checkpoint_vocab_size(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    preferred_suffixes = [
        "lm_head.weight",
        "base_model.lm_head.weight",
        "base_model.model.embed_tokens.weight",
        "base_model.model.model.embed_tokens.weight",
        "embed_tokens.weight",
        "model.embed_tokens.weight",
    ]
    sizes = []
    for k, v in state_dict.items():
        if torch.is_tensor(v) and v.ndim == 2 and int(v.shape[0]) > 10000:
            if any(k.endswith(suf) for suf in preferred_suffixes):
                sizes.append((k, int(v.shape[0])))
    if sizes:
        vocab_size = max(x[1] for x in sizes)
        print(f"[INFO] inferred checkpoint vocab_size={vocab_size} from {sizes[:6]}")
        return vocab_size
    return None


def filter_state_dict_by_shape(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    current = module.state_dict()
    kept = {}
    skipped = []
    for k, v in state_dict.items():
        if k in current and torch.is_tensor(v) and tuple(v.shape) != tuple(current[k].shape):
            skipped.append((k, tuple(v.shape), tuple(current[k].shape)))
            continue
        kept[k] = v
    if skipped:
        print(f"[WARN] skipped {len(skipped)} tensors due to shape mismatch")
        for name, ckpt_shape, cur_shape in skipped[:20]:
            print(f"[WARN] shape mismatch skip: {name} checkpoint={ckpt_shape} current={cur_shape}")
    return kept


def maybe_apply_lora(base_model, args):
    if int(args.lora_rank) <= 0:
        return base_model
    if not PEFT_AVAILABLE or get_peft_model is None:
        raise ImportError("peft is required for --lora_rank > 0")
    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    if not target_modules:
        raise ValueError("--lora_target_modules must be non-empty when --lora_rank > 0")
    base_model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(base_model, lora_config)


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_direct_generation_runtime(args) -> Runtime:
    """
    Load direct-generation SFT checkpoints.

    Main path:
      --model_name_or_path Qwen/Qwen3-8B
      --checkpoint_dir ./direct_json_sft_qwen3_8b/global_step_400

    The checkpoint_dir should contain model.pt saved by the FSDP trainer.
    """
    checkpoint_dir = args.checkpoint_dir
    meta = None
    if checkpoint_dir:
        meta = read_json_if_exists(os.path.join(checkpoint_dir, "direct_generation_meta.json"))
        if meta is None:
            meta = read_json_if_exists(os.path.join(checkpoint_dir, "joint_head_meta.json"))

    tokenizer_source = args.tokenizer_name_or_path or checkpoint_dir or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have pad_token_id")

    torch_dtype = resolve_dtype(args.dtype)

    model_pt = os.path.join(checkpoint_dir, "model.pt") if checkpoint_dir else None
    state = None
    ckpt_vocab_size = None

    if model_pt and os.path.exists(model_pt):
        print(f"[INFO] reading FSDP checkpoint: {model_pt}")
        state = torch.load(model_pt, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ["state_dict", "model_state_dict", "module", "model"]:
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint object: {type(state)}")
        state = strip_known_prefixes(state)
        ckpt_vocab_size = infer_checkpoint_vocab_size(state)
    else:
        print("[INFO] no model.pt found; loading as a normal HuggingFace model/adapter if provided")

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    from_pretrained_kwargs = dict(
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if args.attn_implementation:
        from_pretrained_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map and args.device_map.lower() != "none":
        from_pretrained_kwargs["device_map"] = args.device_map

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **from_pretrained_kwargs)

    current_vocab_size = int(base_model.get_input_embeddings().weight.shape[0])
    tokenizer_vocab_size = len(tokenizer)
    target_vocab_size = max(current_vocab_size, tokenizer_vocab_size, ckpt_vocab_size or 0)
    print(
        "[INFO] vocab sizes: "
        f"base={current_vocab_size}, tokenizer={tokenizer_vocab_size}, "
        f"checkpoint={ckpt_vocab_size}, target={target_vocab_size}"
    )
    if target_vocab_size != current_vocab_size:
        print(f"[INFO] resizing token embeddings: {current_vocab_size} -> {target_vocab_size}")
        base_model.resize_token_embeddings(target_vocab_size)

    # Optional PEFT adapter path for non-FSDP LoRA checkpoints.
    if args.adapter_name_or_path:
        if not PEFT_AVAILABLE:
            raise ImportError("adapter_name_or_path was provided, but peft is not installed")
        base_model = PeftModel.from_pretrained(base_model, args.adapter_name_or_path)
        if args.merge_adapter and hasattr(base_model, "merge_and_unload"):
            base_model = base_model.merge_and_unload()

    # If the FSDP trainer used LoRA and saved full model.pt, pass --lora_rank to recreate LoRA modules.
    base_model = maybe_apply_lora(base_model, args)

    wrapper = DirectJSONGenerationLMModel(base_model=base_model)

    if state is not None:
        state = filter_state_dict_by_shape(wrapper, state)
        incompatible = wrapper.load_state_dict(state, strict=False)
        print(f"[INFO] missing keys: {len(incompatible.missing_keys)}")
        print(f"[INFO] unexpected keys: {len(incompatible.unexpected_keys)}")
        if incompatible.missing_keys:
            print("[INFO] sample missing:", incompatible.missing_keys[:20])
        if incompatible.unexpected_keys:
            print("[INFO] sample unexpected:", incompatible.unexpected_keys[:20])

    if not args.device_map or args.device_map.lower() == "none":
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        wrapper.to(device)
    else:
        device = get_model_device(wrapper.base_model)

    wrapper.eval()
    wrapper.base_model.eval()

    return Runtime(
        tokenizer=tokenizer,
        model=wrapper,
        device=device,
        checkpoint_meta=meta,
    )


# =========================================================
# 7) Generation + parsing
# =========================================================

def cleanup_generated_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return text


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    raw = cleanup_generated_text(text)
    if not raw:
        return []

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass

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
        s = str(x).strip().strip("`\"'").strip()
        if not s or s.lower() in {"null", "none", "unknown", "[]", "__none__"}:
            continue

        matched = None
        if s in valid_agents:
            matched = s
        elif s.lower() in lower_map:
            matched = lower_map[s.lower()]
        else:
            # Fallback for outputs such as "The responsible agent is Planner".
            hits = [v for v in valid_agents if v.lower() in s.lower()]
            if len(hits) == 1:
                matched = hits[0]

        if matched:
            normalized.append(matched)

    return _dedup_keep_order(normalized)


def parse_joint_json_output(text: str, valid_agents: List[str]) -> Dict[str, Any]:
    raw_text = cleanup_generated_text(text)
    objects = extract_json_objects(raw_text)

    if not objects:
        return {
            "parse_ok": False,
            "raw_output": raw_text,
            "json_objects": [],
            "label_token": None,
            "label": None,
            "agent_names": [],
            "structured": [],
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

    # If any A object appears, treat this hypothesis as entailed and collect all agents.
    # Otherwise B/C should normally have exactly one object.
    final_label_token = "A" if "A" in labels else labels[0]
    final_agent_names = _dedup_keep_order(agent_names) if final_label_token == "A" else []
    final_label = TOKEN_LABEL_MAP.get(final_label_token)

    return {
        "parse_ok": final_label_token in CANONICAL_LABELS,
        "raw_output": raw_text,
        "json_objects": valid_objects,
        "json_str": "\n".join(json.dumps(x, ensure_ascii=False) for x in valid_objects),
        "label_token": final_label_token,
        "label": final_label,
        "agent_names": final_agent_names,
        "structured": valid_objects,
    }


@torch.no_grad()
def batch_generate_joint_json(
    runtime: Runtime,
    prompts: List[str],
    candidate_agents: List[List[str]],
    batch_size: int = 4,
    max_input_length: int = 8192,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.0,
    progress_interval: int = 10,
) -> List[Dict[str, Any]]:
    tokenizer = runtime.tokenizer
    model = runtime.model.base_model
    device = get_model_device(model)

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    outputs_all: List[Dict[str, Any]] = []
    total = len(prompts)
    total_batches = (total + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_idx, start in enumerate(range(0, total, batch_size), 1):
        batch_prompts = prompts[start:start + batch_size]
        batch_agent_lists = candidate_agents[start:start + batch_size]
        end = min(start + batch_size, total)

        if progress_interval > 0 and (batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == total_batches):
            elapsed = max(time.time() - t0, 1e-6)
            speed = end / elapsed
            print(
                f"[PROGRESS][generate] batch {batch_idx}/{total_batches}, "
                f"candidate_prompts {end}/{total}, elapsed={elapsed:.1f}s, speed={speed:.2f}/s",
                flush=True,
            )

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
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
# 8) Inference logic
# =========================================================

def run_inference(samples: List[Dict[str, Any]], runtime: Runtime, args) -> List[Dict[str, Any]]:
    candidates = build_candidates(
        samples=samples,
        tokenizer=runtime.tokenizer,
        input_format=args.input_format,
        disable_thinking=args.disable_thinking,
    )
    if not candidates:
        return []

    prompts = [c.prompt_text for c in candidates]
    candidate_agents = [c.agents for c in candidates]

    if args.input_format == "prompt" and args.disable_thinking:
        print(
            "[WARN] input_format=prompt uses pre-built prompts. "
            "disable_thinking only works if those prompts were created with enable_thinking=False."
        )

    print(f"[INFO] total candidate prompts: {len(prompts)}")
    model_outputs = batch_generate_joint_json(
        runtime=runtime,
        prompts=prompts,
        candidate_agents=candidate_agents,
        batch_size=args.model_batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        progress_interval=args.progress_interval,
    )

    if args.input_format == "prompt":
        rows = []
        for cand, gen_out in zip(candidates, model_outputs):
            label_token = gen_out.get("label_token")
            label = gen_out.get("label")
            final_agents = gen_out.get("agent_names", []) if label_token == "A" else []
            rows.append({
                "trajectory_id": cand.trajectory_id,
                "error_type": cand.error_type,
                "hypothesis": cand.hypothesis,
                "label": label,
                "label_token": label_token,
                "agent_names": final_agents,
                "agent_name": final_agents[0] if final_agents else None,
                "parse_ok": gen_out.get("parse_ok", False),
                "raw_output": gen_out.get("raw_output", ""),
                "json_objects": gen_out.get("json_objects", []),
                "mode": "direct_generation_full_json",
            })
        return rows

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for cand, gen_out in zip(candidates, model_outputs):
        label_token = gen_out.get("label_token")
        label = gen_out.get("label")
        final_agents = gen_out.get("agent_names", []) if label_token == "A" else []
        row = {
            "error_type": cand.error_type,
            "hypothesis": cand.hypothesis,
            "label": label,
            "label_token": label_token,
            "agent_names": final_agents,
            "agent_name": final_agents[0] if final_agents else None,
            "parse_ok": gen_out.get("parse_ok", False),
            "raw_output": gen_out.get("raw_output", ""),
            "json_str": gen_out.get("json_str"),
            "json_objects": gen_out.get("json_objects", []),
            "structured": gen_out.get("structured", []),
            "mode": "direct_generation_full_json",
        }
        grouped.setdefault(cand.trajectory_id, []).append(row)

    sample_map = {str(s.get("id", s.get("trajectory_id", "unknown_trajectory"))): s for s in samples}
    results = []

    for sample in samples:
        tid = str(sample.get("id", sample.get("trajectory_id", "unknown_trajectory")))
        src = sample_map.get(tid, {})
        query = src.get("input", {}).get("query") if isinstance(src.get("input"), dict) else None
        rows = grouped.get(tid, [])

        selected_failure_types = []
        predicted_failures = []
        seen_failure_agents = set()

        for row in rows:
            if row.get("label_token") == "A":
                selected_failure_types.append({
                    "error_type": row["error_type"],
                    "hypothesis": row["hypothesis"],
                    "label": row["label"],
                    "label_token": row["label_token"],
                    "agent_names": row.get("agent_names", []),
                    "agent_name": row.get("agent_name"),
                    "parse_ok": row.get("parse_ok"),
                    "raw_output": row.get("raw_output"),
                    "json_objects": row.get("json_objects", []),
                    "mode": row.get("mode"),
                })

                for agent_name in row.get("agent_names", []):
                    key = (agent_name, row["error_type"])
                    if key in seen_failure_agents:
                        continue
                    seen_failure_agents.add(key)
                    predicted_failures.append({
                        "agent_name": agent_name,
                        "error_type": row["error_type"],
                        "hypothesis": row["hypothesis"],
                        "label": row["label"],
                        "label_token": row["label_token"],
                        "parse_ok": row.get("parse_ok"),
                        "mode": row.get("mode"),
                    })

        results.append({
            "trajectory_id": tid,
            "query": query,
            "selected_failure_types": selected_failure_types,
            "predicted_failures": predicted_failures,
            "all_scores": rows,
        })

    return results


# =========================================================
# 9) Evaluation: pair / agent / failure
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

    total_tp = total_fp = total_fn = 0
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
            elif (not in_gold) and in_pred:
                fp += 1
            elif in_gold and (not in_pred):
                fn += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_i = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0.0

        macro_precision_sum += precision_i
        macro_recall_sum += recall_i
        macro_f1_sum += f1_i
        num_classes += 1

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

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


def evaluate_tuple_pair(prediction_rows: List[Dict[str, Any]], source_samples: List[Dict[str, Any]]) -> Dict[str, float]:
    sample_map = {str(x.get("id", x.get("trajectory_id", "unknown_trajectory"))): x for x in source_samples}
    all_gold_pred_sets = []

    for row in prediction_rows:
        tid = str(row["trajectory_id"])
        gold_faults = sample_map.get(tid, {}).get("output", {}).get("faulty_agents", [])
        gold_set = {
            (g.get("agent_name"), g.get("error_type"))
            for g in gold_faults
            if g.get("agent_name") is not None and g.get("error_type") is not None
        }
        pred_set = {
            (p.get("agent_name"), p.get("error_type"))
            for p in row.get("predicted_failures", [])
            if p.get("agent_name") is not None and p.get("error_type") is not None
        }
        all_gold_pred_sets.append((gold_set, pred_set))

    return _evaluate_classwise_from_sets(all_gold_pred_sets)


def evaluate_tuple_agent(prediction_rows: List[Dict[str, Any]], source_samples: List[Dict[str, Any]]) -> Dict[str, float]:
    sample_map = {str(x.get("id", x.get("trajectory_id", "unknown_trajectory"))): x for x in source_samples}
    all_gold_pred_sets = []

    for row in prediction_rows:
        tid = str(row["trajectory_id"])
        gold_faults = sample_map.get(tid, {}).get("output", {}).get("faulty_agents", [])
        gold_set = {g.get("agent_name") for g in gold_faults if g.get("agent_name") is not None}
        pred_set = {p.get("agent_name") for p in row.get("predicted_failures", []) if p.get("agent_name") is not None}
        all_gold_pred_sets.append((gold_set, pred_set))

    return _evaluate_classwise_from_sets(all_gold_pred_sets)


def evaluate_tuple_failure(prediction_rows: List[Dict[str, Any]], source_samples: List[Dict[str, Any]]) -> Dict[str, float]:
    sample_map = {str(x.get("id", x.get("trajectory_id", "unknown_trajectory"))): x for x in source_samples}
    fixed_error_types = set(FM_TYPE_TEMPLATES.keys())
    all_gold_pred_sets = []

    for row in prediction_rows:
        tid = str(row["trajectory_id"])
        gold_faults = sample_map.get(tid, {}).get("output", {}).get("faulty_agents", [])
        gold_set = {g.get("error_type") for g in gold_faults if g.get("error_type") is not None}
        pred_set = {p.get("error_type") for p in row.get("selected_failure_types", []) if p.get("error_type") is not None}
        all_gold_pred_sets.append((gold_set, pred_set))

    return _evaluate_classwise_from_sets(all_gold_pred_sets, label_space=fixed_error_types)


def metrics_to_percent_row(metrics_pair: Dict[str, float], metrics_agent: Dict[str, float], metrics_failure: Dict[str, float]) -> str:
    values = [
        metrics_pair.get("micro_f1", 0.0) * 100,
        metrics_pair.get("macro_f1", 0.0) * 100,
        metrics_agent.get("micro_f1", 0.0) * 100,
        metrics_agent.get("macro_f1", 0.0) * 100,
        metrics_failure.get("micro_f1", 0.0) * 100,
        metrics_failure.get("macro_f1", 0.0) * 100,
    ]
    return " & ".join(f"{x:.2f}" for x in values)


# =========================================================
# 10) Args / main
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
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-8B",
    )

    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument("--input_jsonl", type=str, default="./data/whowhen.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="./outputs/predictions_direct_generation.jsonl")
    parser.add_argument("--input_format", type=str, default="aegis", choices=["aegis", "whowhen", "prompt"])

    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device_map", type=str, default="auto", help="Use 'auto' for multi-GPU loading, or 'none' for manual .to(device).")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    parser.add_argument("--model_batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable_thinking", type=str2bool, default=True)
    parser.add_argument("--progress_interval", type=int, default=10)

    parser.add_argument("--eval_with_gold", type=str2bool, default=True)
    parser.add_argument("--max_samples", type=int, default=-1)

    # Needed only if your FSDP SFT used LoRA and model.pt stores unmerged LoRA modules.
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--merge_adapter", type=str2bool, default=True)

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] loading direct-generation runtime...")
    runtime = load_direct_generation_runtime(args)

    print("[INFO] reading input jsonl...")
    samples = read_jsonl(args.input_jsonl)
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"[INFO] loaded trajectories: {len(samples)}")

    print("[INFO] running direct-generation inference...")
    prediction_rows = run_inference(samples=samples, runtime=runtime, args=args)

    print(f"[INFO] writing outputs to {args.output_jsonl}")
    write_jsonl(args.output_jsonl, prediction_rows)

    if args.eval_with_gold and args.input_format in {"aegis", "whowhen"}:
        metrics_pair = evaluate_tuple_pair(prediction_rows, samples)
        metrics_agent = evaluate_tuple_agent(prediction_rows, samples)
        metrics_failure = evaluate_tuple_failure(prediction_rows, samples)

        print("[INFO] tuple-level metrics:")
        print("Metric_pair", metrics_pair)
        print("Metric_agent", metrics_agent)
        print("Metric_failure", metrics_failure)
        print("[INFO] percent F1 row: Pair_mu Pair_M Agent_mu Agent_M Error_mu Error_M")
        print(metrics_to_percent_row(metrics_pair, metrics_agent, metrics_failure))

    print("[INFO] done.")


if __name__ == "__main__":
    main()
