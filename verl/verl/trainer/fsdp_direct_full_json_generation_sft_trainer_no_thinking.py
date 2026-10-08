# -*- coding: utf-8 -*-
# @Author  : qiaohezhe / ChatGPT
# @Date    : 2026-04-28
# @File    : fsdp_direct_full_json_generation_sft_trainer.py
# @Desc    : Direct generation SFT trainer with FSDP.
#            - No cls_head.
#            - No label CE loss.
#            - label and agent are both learned as JSON generation.
#            - LM loss is computed only on response JSON tokens.
#            - Qwen3 no-thinking SFT: strip <think> blocks and add /no_think.
#
# Data format:
# {"prompt": "...", "response": "{\"label\":\"A\",\"agents\":[\"Solver\"]}"}
# Multi-agent A format:
# {"label":"A","agents":["Planner"]}\n{"label":"A","agents":["Solver"]}
# B/C format:
# {"label":"B","agents":[]}

import os
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
import verl.utils.hdfs_io as hdfs_io

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

LABEL_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_LABEL = {0: "A", 1: "B", 2: "C"}


# =========================================================
# IO and parsing
# =========================================================

def read_rows_from_files(data_files) -> List[Dict[str, Any]]:
    files = [data_files] if isinstance(data_files, str) else list(data_files)
    rows: List[Dict[str, Any]] = []
    for file_path in files:
        path = str(file_path)
        suffix = Path(path).suffix.lower()
        if suffix == ".parquet":
            rows.extend(pd.read_parquet(path).to_dict(orient="records"))
        elif suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if not isinstance(obj, list):
                    raise ValueError(f"JSON file must contain a list: {path}")
                rows.extend(obj)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    return rows


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out, seen = [], set()
    for v in values:
        s = str(v).strip()
        if not s or s.lower() in {"none", "null", "__none__"}:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _json_objects_from_text(text: str) -> List[Any]:
    """Parse a JSON object, JSON list, JSONL, or adjacent JSON objects."""
    text = str(text).strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        pass

    objs = []
    ok = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except Exception:
            ok = False
            break
    if ok and objs:
        return objs

    decoder = json.JSONDecoder()
    idx, n, objs = 0, len(text), []
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        objs.append(obj)
        idx = end
    return objs


def _extract_label_agents(obj: Any) -> Tuple[str, List[str]]:
    if not isinstance(obj, dict):
        raise ValueError(f"Response item must be a JSON object, got {type(obj)}")
    label = str(obj.get("label", "")).strip().upper()
    if label not in LABEL_TO_ID:
        raise ValueError(f"Invalid label: {obj}")

    agents: List[str] = []
    if "agents" in obj:
        v = obj.get("agents")
        if v is None:
            agents = []
        elif isinstance(v, str):
            agents = [v]
        elif isinstance(v, (list, tuple)):
            agents = [str(x).strip() for x in v if x is not None]
        else:
            raise ValueError(f"agents must be list/string/null: {obj}")
    elif "agent" in obj:
        v = obj.get("agent")
        if v is not None:
            agents = [str(v).strip()]
    return label, _dedupe_keep_order(agents)


def parse_direct_response(response_value: Any, allow_empty_agent_for_A: bool = True) -> Tuple[str, List[str], str]:
    """
    Canonicalize response to the JSON/JSONL text used as LM target.
    For A with multiple agents, output one JSON object per line.
    For B/C, output one JSON object with empty agents.
    """
    if isinstance(response_value, dict):
        objects = [response_value]
    else:
        text = str(response_value).strip()
        if text in {"A", "B", "C"}:
            canonical = json.dumps({"label": text, "agents": []}, ensure_ascii=False, separators=(",", ":"))
            return text, [], canonical
        objects = _json_objects_from_text(text)

    if not objects:
        raise ValueError(f"Empty response: {response_value}")

    labels, agents = [], []
    for obj in objects:
        label, obj_agents = _extract_label_agents(obj)
        labels.append(label)
        agents.extend(obj_agents)

    if len(set(labels)) != 1:
        raise ValueError(f"Inconsistent labels in response: {labels}")

    final_label = labels[0]
    agent_names = _dedupe_keep_order(agents)

    if final_label == "A":
        if not agent_names and not allow_empty_agent_for_A:
            raise ValueError(f"A response has no agent: {response_value}")
        if agent_names:
            canonical = "\n".join(
                json.dumps({"label": "A", "agents": [a]}, ensure_ascii=False, separators=(",", ":"))
                for a in agent_names
            )
        else:
            canonical = json.dumps({"label": "A", "agents": []}, ensure_ascii=False, separators=(",", ":"))
    else:
        agent_names = []
        canonical = json.dumps({"label": final_label, "agents": []}, ensure_ascii=False, separators=(",", ":"))

    return final_label, agent_names, canonical


def parse_candidate_agents_from_prompt(prompt: str) -> List[str]:
    marker = "Candidate agents:"
    idx = str(prompt).find(marker)
    if idx < 0:
        return []
    tail = str(prompt)[idx + len(marker):]
    agents = []
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            if agents:
                break
            continue
        if line.startswith("- "):
            a = line[2:].strip()
            if a and a.lower() != "none":
                agents.append(a)
        else:
            if agents:
                break
    return _dedupe_keep_order(agents)


def normalize_candidate_agents(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
            value = parsed if isinstance(parsed, list) else [text]
        except Exception:
            value = [x.strip() for x in text.split(",")]
    if not isinstance(value, (list, tuple)):
        return []
    return _dedupe_keep_order([str(x).strip() for x in value if x is not None])


# =========================================================
# Qwen3 no-thinking prompt helpers
# =========================================================

DEFAULT_SYSTEM_PROMPT_DIRECT_GENERATION = """You are a careful verifier for multi-agent trajectory failure attribution.

You will be given a trajectory, a failure hypothesis, and candidate agents.
Your task is to jointly predict:
- label: one of A, B, C
- agents: responsible agent(s) for the hypothesized failure

A = entail
B = neutral
C = contradict

Output JSON objects only. Do not output thinking, reasoning, analysis, markdown, or extra text.
"""


def _cfg_get(config, name: str, default: Any = None) -> Any:
    """Read from OmegaConf/DictConfig or normal object safely."""
    try:
        if hasattr(config, "get"):
            return config.get(name, default)
    except Exception:
        pass
    return getattr(config, name, default)


def _cfg_bool(config, name: str, default: bool = False) -> bool:
    v = _cfg_get(config, name, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(v)


def strip_thinking_blocks(text: str) -> str:
    """Remove existing generated thinking traces from prebuilt prompts, if any."""
    text = str(text)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Also remove broken/open fragments if bad data accidentally contains them.
    text = re.sub(r"<think>.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</think>\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def ensure_no_think_tag(prompt: str) -> str:
    """Append /no_think once near the end of the prompt."""
    prompt = str(prompt).rstrip()
    tail = prompt[-300:].lower()
    if "/no_think" in tail:
        return prompt
    return prompt + "\n\n/no_think"


def _parse_messages_value(value: Any) -> Optional[List[Dict[str, str]]]:
    """Parse a messages column if the dataset provides one."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, list):
        return None

    messages: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role", "user")).strip()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user"
        messages.append({"role": role, "content": strip_thinking_blocks(content)})
    return messages or None


def _looks_like_chat_template(prompt: str) -> bool:
    """Avoid wrapping an already formatted chat-template prompt as another user message."""
    p = str(prompt)
    chat_markers = [
        "<|im_start|>", "<|im_end|>",
        "<|start_header_id|>", "<|end_header_id|>",
        "[INST]", "[/INST]",
        "### Human:", "### Assistant:",
        "System:", "User:", "Assistant:",
    ]
    return any(m in p for m in chat_markers)


def build_no_thinking_prompt(row: Dict[str, Any], tokenizer, config) -> str:
    """
    Build or sanitize the training prompt for Qwen3 no-thinking SFT.

    Priority:
    1) If row has a messages column, rebuild with tokenizer.apply_chat_template(..., enable_thinking=False).
    2) Else use row[prompt_key].
       - strip any <think>...</think> traces
       - append /no_think by default
       - optionally rebuild raw non-chat prompts with enable_thinking=False

    Important: If your parquet/jsonl only stores an already-rendered prompt, the official hard
    switch can only be applied when that prompt was originally created. This function therefore
    keeps pre-rendered chat prompts intact and adds /no_think as a safe soft switch.
    """
    prompt_key = _cfg_get(config, "prompt_key", "prompt")
    messages_key = _cfg_get(config, "messages_key", "messages")
    force_no_thinking = _cfg_bool(config, "force_no_thinking", True)
    strip_blocks = _cfg_bool(config, "strip_think_blocks", True)
    append_tag = _cfg_bool(config, "append_no_think_tag", True)
    rebuild_raw = _cfg_bool(config, "rebuild_raw_prompt_with_chat_template", False)

    # Hard no-thinking path if messages are available.
    messages = _parse_messages_value(row.get(messages_key))
    if force_no_thinking and messages and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older transformers/tokenizers may not support enable_thinking.
            prompt = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
            return ensure_no_think_tag(prompt) if append_tag else prompt

    prompt = str(row.get(prompt_key, ""))
    if strip_blocks:
        prompt = strip_thinking_blocks(prompt)

    # Optional hard-switch rebuild for raw prompts only. Default is False to avoid damaging
    # datasets whose prompt column already contains a full chat template.
    if (
        force_no_thinking
        and rebuild_raw
        and hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
        and not _looks_like_chat_template(prompt)
    ):
        system_prompt = str(_cfg_get(config, "system_prompt", DEFAULT_SYSTEM_PROMPT_DIRECT_GENERATION)).strip()
        user_content = ensure_no_think_tag(prompt) if append_tag else prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return f"System: {system_prompt}\n\nUser: {user_content}\n\nAssistant:"

    if force_no_thinking and append_tag:
        prompt = ensure_no_think_tag(prompt)
    return prompt


# =========================================================
# Dataset
# =========================================================

class DirectJSONGenerationDataset(Dataset):
    def __init__(self, raw_rows, tokenizer, config):
        self.tokenizer = tokenizer
        self.prompt_key = getattr(config, "prompt_key", "prompt")
        self.response_key = getattr(config, "response_key", "response")
        self.candidate_agents_key = getattr(config, "candidate_agents_key", "candidate_agents")
        self.max_length = int(getattr(config, "max_length", 2048))
        self.max_prompt_length = int(getattr(config, "max_prompt_length", self.max_length))
        self.max_target_length = int(getattr(config, "max_target_length", 256))
        self.append_eos_to_target = bool(getattr(config, "append_eos_to_target", True))
        self.allow_empty_agent_for_A = bool(getattr(config, "allow_empty_agent_for_A", True))
        self.force_no_thinking = _cfg_bool(config, "force_no_thinking", True)
        self.strip_think_blocks = _cfg_bool(config, "strip_think_blocks", True)
        self.append_no_think_tag = _cfg_bool(config, "append_no_think_tag", True)
        self.rebuild_raw_prompt_with_chat_template = _cfg_bool(config, "rebuild_raw_prompt_with_chat_template", False)
        self.messages_key = _cfg_get(config, "messages_key", "messages")
        self.num_prompts_with_think = 0

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have eos_token_id")

        self.effective_max_prompt_length = min(self.max_prompt_length, self.max_length - self.max_target_length)
        if self.effective_max_prompt_length <= 0:
            raise ValueError("max_length must be larger than max_target_length")

        self.rows = self._validate_and_normalize_rows(raw_rows)
        if len(self.rows) == 0:
            raise ValueError("No valid rows loaded")

    def _validate_and_normalize_rows(self, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows, dropped = [], 0
        label_counter = {"A": 0, "B": 0, "C": 0}
        multi_agent_a = 0

        for row in raw_rows:
            if self.prompt_key not in row or self.response_key not in row:
                dropped += 1
                continue
            prompt = row.get(self.prompt_key)
            response = row.get(self.response_key)
            messages_value = row.get(self.messages_key)
            if response is None or (prompt is None and messages_value is None):
                dropped += 1
                continue
            try:
                prompt = build_no_thinking_prompt(row, self.tokenizer, self)
            except Exception as e:
                dropped += 1
                continue
            if "<think>" in prompt.lower() or "</think>" in prompt.lower():
                self.num_prompts_with_think += 1
            try:
                label, agent_names, canonical_response = parse_direct_response(
                    response, allow_empty_agent_for_A=self.allow_empty_agent_for_A
                )
            except Exception:
                dropped += 1
                continue

            if self.candidate_agents_key in row:
                candidate_agents = normalize_candidate_agents(row[self.candidate_agents_key])
            else:
                candidate_agents = parse_candidate_agents_from_prompt(prompt)
            for a in agent_names:
                if a not in candidate_agents:
                    candidate_agents.append(a)
            candidate_agents = _dedupe_keep_order(candidate_agents)

            label_counter[label] += 1
            if label == "A" and len(agent_names) > 1:
                multi_agent_a += 1

            rows.append({
                "prompt": prompt,
                "label": label,
                "agent_names": agent_names,
                "target_text": canonical_response,
                "candidate_agents": candidate_agents,
            })

        print(f"[DATA] loaded {len(rows)} valid rows, dropped {dropped} invalid rows")
        print(f"[DATA] label counts: A={label_counter['A']}, B={label_counter['B']}, C={label_counter['C']}, A_multi_agent={multi_agent_a}")
        print(
            "[DATA] no-thinking config: "
            f"force_no_thinking={self.force_no_thinking}, "
            f"strip_think_blocks={self.strip_think_blocks}, "
            f"append_no_think_tag={self.append_no_think_tag}, "
            f"rebuild_raw_prompt_with_chat_template={self.rebuild_raw_prompt_with_chat_template}"
        )
        if self.num_prompts_with_think > 0:
            print(f"[WARN] {self.num_prompts_with_think} prompts still contain <think> after sanitization")
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt_enc = self.tokenizer(
            row["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.effective_max_prompt_length,
            return_attention_mask=True,
        )
        target_enc = self.tokenizer(
            row["target_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_length,
            return_attention_mask=False,
        )
        prompt_ids = prompt_enc["input_ids"]
        target_ids = target_enc["input_ids"]
        if self.append_eos_to_target:
            target_ids = target_ids + [self.tokenizer.eos_token_id]

        if len(prompt_ids) + len(target_ids) > self.max_length:
            overflow = len(prompt_ids) + len(target_ids) - self.max_length
            prompt_ids = prompt_ids[:-overflow]

        if len(prompt_ids) < 1 or len(target_ids) < 1:
            raise ValueError(f"Empty tokenized prompt/target at idx={idx}")

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        lm_labels = [-100] * len(prompt_ids) + target_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lm_labels": lm_labels,
            "prompt_length": len(prompt_ids),
            "raw_label": row["label"],
            "gold_response_text": row["target_text"],
            "gold_agent_names": row["agent_names"],
            "candidate_texts": row["candidate_agents"],
        }


class DirectJSONGenerationCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id
        out = {
            "input_ids": [],
            "attention_mask": [],
            "lm_labels": [],
            "prompt_length": [],
            "raw_label": [],
            "gold_response_text": [],
            "gold_agent_names": [],
            "candidate_texts": [],
        }
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            out["input_ids"].append(item["input_ids"] + [pad_id] * pad_len)
            out["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
            out["lm_labels"].append(item["lm_labels"] + [-100] * pad_len)
            out["prompt_length"].append(item["prompt_length"])
            out["raw_label"].append(item["raw_label"])
            out["gold_response_text"].append(item["gold_response_text"])
            out["gold_agent_names"].append(item["gold_agent_names"])
            out["candidate_texts"].append(item["candidate_texts"])
        return {
            "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(out["attention_mask"], dtype=torch.long),
            "lm_labels": torch.tensor(out["lm_labels"], dtype=torch.long),
            "prompt_length": torch.tensor(out["prompt_length"], dtype=torch.long),
            "raw_label": out["raw_label"],
            "gold_response_text": out["gold_response_text"],
            "gold_agent_names": out["gold_agent_names"],
            "candidate_texts": out["candidate_texts"],
        }


# =========================================================
# Model: memory-optimized LM loss, no cls_head
# =========================================================

class DirectJSONGenerationLMModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        self.base_model = base_model
        self.lm_head = self.base_model.get_output_embeddings()
        if self.lm_head is None:
            raise ValueError("base_model.get_output_embeddings() returned None")

    def _get_lm_model(self):
        if hasattr(self.base_model, "get_base_model"):
            return self.base_model.get_base_model()
        return self.base_model

    def _get_backbone(self):
        lm_model = self._get_lm_model()
        prefix = getattr(lm_model, "base_model_prefix", None)
        if prefix is not None and hasattr(lm_model, prefix):
            return getattr(lm_model, prefix)
        if hasattr(lm_model, "model"):
            return lm_model.model
        if hasattr(lm_model, "base_model"):
            return lm_model.base_model
        raise ValueError("Cannot locate backbone model")

    def forward(self, input_ids, attention_mask, lm_labels=None, use_cache=False, return_token_loss=False):
        backbone = self._get_backbone()
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        result = {"active_logits": None, "active_labels": None, "token_loss": None, "num_active_tokens": None}

        if lm_labels is not None:
            shift_hidden = last_hidden[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            active_mask = shift_labels.ne(-100)
            result["num_active_tokens"] = active_mask.float().sum()
            if active_mask.any():
                active_hidden = shift_hidden[active_mask]
                active_labels = shift_labels[active_mask]
                active_logits = self.lm_head(active_hidden)
                result["active_logits"] = active_logits
                result["active_labels"] = active_labels
                if return_token_loss:
                    result["token_loss"] = F.cross_entropy(active_logits.float(), active_labels, reduction="none")
        return result


# =========================================================
# Trainer
# =========================================================

class FSDPDirectJSONGenerationSFTTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.device_name = get_device_name()
        self._normalize_config_bsz()
        self._build_dataloader(train_dataset, val_dataset)
        self._build_model_optimizer()
        if self.device_mesh.get_rank() == 0:
            print(self.config)
            print("[INFO] Direct generation SFT: LM loss only; no cls_head.")

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")
        assert self.config.data.train_batch_size % dp_size == 0
        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        collator = DirectJSONGenerationCollator(self.tokenizer)
        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        num_workers = int(getattr(self.config.data, "num_workers", 4))
        pin_memory = bool(getattr(self.config.data, "pin_memory", True))

        self.train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
        if self.config.model.get("external_lib", None) is not None:
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)
        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = PrecisionType.to_dtype(self.config.model.fsdp_config.get("model_dtype", "bf16"))
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            max_length = int(getattr(self.config.data, "max_length", 2048))
            self.model_config.max_position_embeddings = max(self.model_config.max_position_embeddings, max_length)

        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)
        with init_context():
            base_model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=base_model)
            if self.config.model.get("lora_rank", 0) > 0:
                base_model.enable_input_require_grads()
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                base_model = get_peft_model(base_model, LoraConfig(**lora_config))
            self.model = DirectJSONGenerationLMModel(base_model=base_model)

        if self.config.model.enable_gradient_checkpointing:
            self.model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        cpu_offload = None if not self.config.model.fsdp_config.cpu_offload else CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)
        fsdp_strategy = self.config.model.strategy

        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch >= 2.4 is required for FSDP2"
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, cast_forward_inputs=True)
            fsdp_kwargs = {"mesh": self.device_mesh, "mp_policy": mp_policy, "offload_policy": cpu_offload, "reshard_after_forward": True}
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)
        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(self.optimizer, num_warmup_steps, self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_lm_loss(self, model_outputs: Dict[str, Optional[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        active_logits = model_outputs["active_logits"]
        active_labels = model_outputs["active_labels"]
        num_active_tokens = model_outputs.get("num_active_tokens")
        if active_logits is None or active_labels is None:
            zero = torch.zeros([], device=torch.device(self.device_name), dtype=torch.float32, requires_grad=True)
            return zero, torch.zeros([], device=torch.device(self.device_name))
        loss = F.cross_entropy(active_logits.float(), active_labels)
        if num_active_tokens is None:
            num_active_tokens = torch.tensor(float(active_labels.numel()), device=active_labels.device)
        return loss, num_active_tokens.detach().float()

    def _forward_and_compute_core_metrics(self, batch):
        input_ids = batch["input_ids"].to(self.device_name, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device_name, non_blocking=True)
        lm_labels = batch["lm_labels"].to(self.device_name, non_blocking=True)
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            outputs = self.fsdp_model(input_ids=input_ids, attention_mask=attention_mask, lm_labels=lm_labels, use_cache=False)
            lm_loss, num_active_tokens = self._compute_lm_loss(outputs)
        return {
            "loss": lm_loss,
            "lm_loss": lm_loss.detach(),
            "num_active_tokens": num_active_tokens,
            "num_samples": torch.tensor(float(input_ids.size(0)), device=input_ids.device),
        }

    def _empty_count_stats(self) -> Dict[str, torch.Tensor]:
        device = torch.device(self.device_name)
        return {k: torch.zeros([], device=device, dtype=torch.float32) for k in ["loss", "lm_loss", "num_active_tokens", "num_samples"]}

    def _accumulate_count_stats(self, acc, out, loss_weight=1.0):
        for k in ["loss", "lm_loss"]:
            acc[k] += out[k].detach().float() * loss_weight
        for k in ["num_active_tokens", "num_samples"]:
            acc[k] += out[k].detach().float()

    def _metrics_from_count_stats(self, prefix: str, stats: Dict[str, torch.Tensor], lr: Optional[float] = None) -> Dict[str, float]:
        lm_loss = float(stats["lm_loss"].item())
        metrics = {
            f"{prefix}/loss": float(stats["loss"].item()),
            f"{prefix}/lm_loss": lm_loss,
            f"{prefix}/ppl": float(torch.exp(torch.tensor(min(lm_loss, 20.0))).item()),
            f"{prefix}/num_active_tokens": float(stats["num_active_tokens"].item()),
            f"{prefix}/num_samples": float(stats["num_samples"].item()),
        }
        if lr is not None:
            metrics[f"{prefix}/lr(1e-3)"] = lr * 1e3
        return metrics

    def training_step(self, batch):
        self.fsdp_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        bsz = batch["input_ids"].size(0)
        micro = self.config.data.micro_batch_size_per_gpu
        micro_batches = []
        for start in range(0, bsz, micro):
            end = start + micro
            micro_batches.append({
                "input_ids": batch["input_ids"][start:end],
                "attention_mask": batch["attention_mask"][start:end],
                "lm_labels": batch["lm_labels"][start:end],
            })
        stats = self._empty_count_stats()
        n_micro = len(micro_batches)
        for micro_batch in micro_batches:
            out = self._forward_and_compute_core_metrics(micro_batch)
            (out["loss"] / n_micro).backward()
            self._accumulate_count_stats(stats, out, loss_weight=1.0 / n_micro)

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.step()
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]

        for k in stats:
            op = torch.distributed.ReduceOp.AVG if k in {"loss", "lm_loss"} else torch.distributed.ReduceOp.SUM
            torch.distributed.all_reduce(stats[k], op=op)
        return self._metrics_from_count_stats("train", stats, lr=lr)

    @torch.no_grad()
    def validation_step(self, batch):
        self.fsdp_model.eval()
        out = self._forward_and_compute_core_metrics(batch)
        stats = self._empty_count_stats()
        self._accumulate_count_stats(stats, out, loss_weight=1.0)
        return stats

    def save_checkpoint(self, step):
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        strategy = self.config.model.strategy
        if strategy == "fsdp":
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()
        elif strategy == "fsdp2":
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)
        else:
            raise NotImplementedError

        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            torch.save(state_dict, os.path.join(path, "model.pt"))
            self.model_config.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            meta = {
                "checkpoint_type": "direct_full_json_generation_lm_memsave",
                "has_cls_head": False,
                "label_prediction": "generation",
                "agent_prediction": "generation",
                "lm_target": "full_json_response",
                "supports_multi_agent_jsonl_response": True,
                "qwen3_no_thinking_sft": True,
                "prompt_no_thinking_policy": {
                    "force_no_thinking": bool(getattr(self.config.data, "force_no_thinking", True)),
                    "strip_think_blocks": bool(getattr(self.config.data, "strip_think_blocks", True)),
                    "append_no_think_tag": bool(getattr(self.config.data, "append_no_think_tag", True)),
                    "rebuild_raw_prompt_with_chat_template": bool(getattr(self.config.data, "rebuild_raw_prompt_with_chat_template", False)),
                },
                "label_to_id": LABEL_TO_ID,
                "id_to_label": ID_TO_LABEL,
            }
            with open(os.path.join(path, "direct_generation_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )
        global_step = 0
        last_valid_metric = None
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in self.train_dataloader:
                global_step += 1
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                is_last_step = global_step >= self.total_training_steps
                is_valid_step = self.config.trainer.test_freq > 0 and global_step % self.config.trainer.test_freq == 0
                is_save_step = self.config.trainer.save_freq > 0 and global_step % self.config.trainer.save_freq == 0

                if is_last_step or is_valid_step:
                    val_stats, val_batches = None, 0
                    for val_data in self.val_dataloader:
                        out = self.validation_step(val_data)
                        if val_stats is None:
                            val_stats = {k: torch.zeros_like(v) for k, v in out.items()}
                        for k, v in out.items():
                            val_stats[k] += v.detach().float()
                        val_batches += 1
                    if val_stats is None:
                        val_stats = self._empty_count_stats()
                    for k in val_stats:
                        torch.distributed.all_reduce(val_stats[k], op=torch.distributed.ReduceOp.SUM)
                    denom = max(1, val_batches * self.device_mesh.size())
                    for k in ["loss", "lm_loss"]:
                        val_stats[k] /= denom
                    if rank == 0:
                        metric = self._metrics_from_count_stats("val", val_stats)
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or is_save_step:
                    self.save_checkpoint(global_step)
                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


# =========================================================
# Run
# =========================================================

def create_dataset_from_rows(raw_rows, data_config, tokenizer):
    return DirectJSONGenerationDataset(raw_rows=raw_rows, tokenizer=tokenizer, config=data_config)


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    if int(getattr(config, "ulysses_sequence_parallel_size", 1)) != 1:
        raise ValueError("Set ulysses_sequence_parallel_size=1 for this trainer.")
    if getattr(config, "use_remove_padding", False):
        raise ValueError("Set use_remove_padding=False for this trainer.")

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    from verl.utils import hf_tokenizer
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_rows_from_files(config.data.train_files)
    val_rows = read_rows_from_files(config.data.val_files)
    train_dataset = create_dataset_from_rows(train_rows, config.data, tokenizer)
    val_dataset = create_dataset_from_rows(val_rows, config.data, tokenizer)

    if rank == 0:
        print("===== DEBUG DATASET =====")
        print("train size:", len(train_dataset))
        for i in range(min(3, len(train_dataset))):
            item = train_dataset[i]
            print(f"--- sample {i} ---")
            print("label:", item["raw_label"])
            print("gold_response_text:", item["gold_response_text"])
            print("gold_agent_names:", item["gold_agent_names"])
            print("input len:", len(item["input_ids"]))
            print("prompt length:", item["prompt_length"])
            print("decoded prompt tail:", tokenizer.decode(item["input_ids"][:item["prompt_length"]][-120:]))
            print("decoded target:", tokenizer.decode([x for x in item["lm_labels"] if x != -100]))
            print("candidate_agents:", item["candidate_texts"][:10])
        print("===== END DEBUG =====")

    trainer = FSDPDirectJSONGenerationSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


if __name__ == "__main__":
    main()
