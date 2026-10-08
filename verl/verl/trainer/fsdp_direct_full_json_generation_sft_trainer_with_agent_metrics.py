# -*- coding: utf-8 -*-
# @Author  : qiaohezhe / ChatGPT
# @Date    : 2026-04-28
# @File    : fsdp_direct_full_json_generation_sft_trainer.py
# @Desc    : Direct generation SFT trainer with FSDP.
#            - No cls_head.
#            - No label CE loss.
#            - label and agent are both learned as JSON generation.
#            - LM loss is computed only on response JSON tokens.
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
            prompt = row[self.prompt_key]
            response = row[self.response_key]
            if prompt is None or response is None:
                dropped += 1
                continue
            prompt = str(prompt)
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
        result = {
            "active_logits": None,
            "active_labels": None,
            "active_pred_ids": None,
            "active_counts": None,
            "token_loss": None,
            "num_active_tokens": None,
        }

        if lm_labels is not None:
            shift_hidden = last_hidden[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            active_mask = shift_labels.ne(-100)
            result["active_counts"] = active_mask.long().sum(dim=1)
            result["num_active_tokens"] = active_mask.float().sum()
            if active_mask.any():
                active_hidden = shift_hidden[active_mask]
                active_labels = shift_labels[active_mask]
                active_logits = self.lm_head(active_hidden)
                result["active_logits"] = active_logits
                result["active_labels"] = active_labels
                # Teacher-forced greedy prediction at response-token positions.
                # This is useful for cheap monitoring of JSON / label / agent accuracy.
                result["active_pred_ids"] = torch.argmax(active_logits.detach().float(), dim=-1)
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

        # Text-level monitoring for direct JSON generation.
        # These metrics are teacher-forced argmax metrics, not free-generation metrics.
        # They are cheap enough for periodic validation and optional training logging.
        self.compute_train_text_metrics = bool(getattr(self.config.trainer, "compute_train_text_metrics", False))
        self.compute_val_text_metrics = bool(getattr(self.config.trainer, "compute_val_text_metrics", True))

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

    @staticmethod
    def _safe_parse_pred_response(text: str) -> Tuple[Optional[str], List[str], bool]:
        """
        Robustly parse model-predicted JSON/JSONL text.

        Returns:
          pred_label, pred_agents, valid_json
        """
        if text is None:
            return None, [], False
        text = str(text).strip().replace("\\n", "\n")
        if not text:
            return None, [], False

        decoder = json.JSONDecoder()
        objs: List[Any] = []
        idx, n = 0, len(text)

        # Scan for JSON objects. This is more robust than json.loads because
        # teacher-forced argmax text can contain malformed prefixes/suffixes.
        while idx < n:
            j = text.find("{", idx)
            if j < 0:
                break
            try:
                obj, end = decoder.raw_decode(text, j)
                objs.append(obj)
                idx = end
            except Exception:
                idx = j + 1

        if not objs:
            return None, [], False

        labels: List[str] = []
        agents: List[str] = []
        try:
            for obj in objs:
                label, obj_agents = _extract_label_agents(obj)
                labels.append(label)
                agents.extend(obj_agents)
        except Exception:
            return None, [], False

        if not labels:
            return None, [], False

        # If labels are inconsistent, mark invalid but still use the first label
        # for debugging-oriented metrics.
        valid = len(set(labels)) == 1
        pred_label = labels[0]
        pred_agents = _dedupe_keep_order(agents)
        if pred_label in {"B", "C"}:
            pred_agents = []
        return pred_label, pred_agents, valid

    def _empty_text_metric_stats(self) -> Dict[str, torch.Tensor]:
        device = torch.device(self.device_name)
        keys = [
            "tf_num_samples",
            "tf_num_A_samples",
            "tf_valid_json",
            "tf_label_correct",
            "tf_agent_exact",
            "tf_agent_exact_A",
            "tf_agent_tp",
            "tf_agent_fp",
            "tf_agent_fn",
            "tf_pred_agent_total",
            "tf_ooc_agent_total",
        ]
        return {k: torch.zeros([], device=device, dtype=torch.float32) for k in keys}

    def _compute_teacher_forced_text_metrics(self, batch, model_outputs: Dict[str, Optional[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Compute cheap text-level metrics from teacher-forced greedy predictions.

        Important:
        - This is NOT free-generation accuracy.
        - It predicts each response token with gold previous response tokens.
        - It is still useful for monitoring whether the model is learning the JSON label/agent target.
        """
        stats = self._empty_text_metric_stats()
        active_pred_ids = model_outputs.get("active_pred_ids", None)
        active_counts = model_outputs.get("active_counts", None)
        if active_pred_ids is None or active_counts is None:
            return stats

        pred_ids_flat = active_pred_ids.detach().cpu().tolist()
        counts = active_counts.detach().cpu().tolist()

        offset = 0
        pred_texts: List[str] = []
        for c in counts:
            c = int(c)
            ids = pred_ids_flat[offset: offset + c]
            offset += c
            pred_texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))

        raw_labels = batch.get("raw_label", [])
        gold_agent_names = batch.get("gold_agent_names", [])
        candidate_texts = batch.get("candidate_texts", [])

        for i, pred_text in enumerate(pred_texts):
            gold_label = raw_labels[i] if i < len(raw_labels) else None
            gold_agents = set(gold_agent_names[i]) if i < len(gold_agent_names) else set()
            candidates = set(candidate_texts[i]) if i < len(candidate_texts) else set()

            pred_label, pred_agents_list, valid = self._safe_parse_pred_response(pred_text)
            pred_agents = set(pred_agents_list)

            stats["tf_num_samples"] += 1.0
            if gold_label == "A":
                stats["tf_num_A_samples"] += 1.0
            if valid:
                stats["tf_valid_json"] += 1.0
            if pred_label == gold_label:
                stats["tf_label_correct"] += 1.0

            if pred_agents == gold_agents:
                stats["tf_agent_exact"] += 1.0
                if gold_label == "A":
                    stats["tf_agent_exact_A"] += 1.0

            tp = len(pred_agents & gold_agents)
            fp = len(pred_agents - gold_agents)
            fn = len(gold_agents - pred_agents)
            stats["tf_agent_tp"] += float(tp)
            stats["tf_agent_fp"] += float(fp)
            stats["tf_agent_fn"] += float(fn)
            stats["tf_pred_agent_total"] += float(len(pred_agents))
            stats["tf_ooc_agent_total"] += float(sum(1 for a in pred_agents if a not in candidates))

        return stats

    def _forward_and_compute_core_metrics(self, batch, compute_text_metrics: bool = False):
        input_ids = batch["input_ids"].to(self.device_name, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device_name, non_blocking=True)
        lm_labels = batch["lm_labels"].to(self.device_name, non_blocking=True)
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            outputs = self.fsdp_model(input_ids=input_ids, attention_mask=attention_mask, lm_labels=lm_labels, use_cache=False)
            lm_loss, num_active_tokens = self._compute_lm_loss(outputs)
        result = {
            "loss": lm_loss,
            "lm_loss": lm_loss.detach(),
            "num_active_tokens": num_active_tokens,
            "num_samples": torch.tensor(float(input_ids.size(0)), device=input_ids.device),
        }
        if compute_text_metrics:
            result.update(self._compute_teacher_forced_text_metrics(batch, outputs))
        return result

    def _empty_count_stats(self, include_text_metrics: bool = False) -> Dict[str, torch.Tensor]:
        device = torch.device(self.device_name)
        stats = {k: torch.zeros([], device=device, dtype=torch.float32) for k in ["loss", "lm_loss", "num_active_tokens", "num_samples"]}
        if include_text_metrics:
            stats.update(self._empty_text_metric_stats())
        return stats

    def _accumulate_count_stats(self, acc, out, loss_weight=1.0):
        for k in ["loss", "lm_loss"]:
            if k in out:
                acc[k] += out[k].detach().float() * loss_weight
        for k, v in out.items():
            if k in {"loss", "lm_loss"}:
                continue
            if k in acc:
                acc[k] += v.detach().float()

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

        # Optional teacher-forced text-level metrics.
        if "tf_num_samples" in stats:
            n = max(1.0, float(stats["tf_num_samples"].item()))
            n_A = max(1.0, float(stats["tf_num_A_samples"].item()))
            tp = float(stats["tf_agent_tp"].item())
            fp = float(stats["tf_agent_fp"].item())
            fn = float(stats["tf_agent_fn"].item())
            pred_total = float(stats["tf_pred_agent_total"].item())

            precision = tp / max(1e-12, tp + fp)
            recall = tp / max(1e-12, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)

            metrics.update({
                f"{prefix}/tf_valid_json_rate": float(stats["tf_valid_json"].item()) / n,
                f"{prefix}/tf_label_acc": float(stats["tf_label_correct"].item()) / n,
                f"{prefix}/tf_agent_exact_acc": float(stats["tf_agent_exact"].item()) / n,
                f"{prefix}/tf_agent_exact_acc_A": float(stats["tf_agent_exact_A"].item()) / n_A,
                f"{prefix}/tf_agent_micro_precision": precision,
                f"{prefix}/tf_agent_micro_recall": recall,
                f"{prefix}/tf_agent_micro_f1": f1,
                f"{prefix}/tf_ooc_agent_rate": float(stats["tf_ooc_agent_total"].item()) / max(1e-12, pred_total),
                f"{prefix}/tf_pred_agent_total": pred_total,
                f"{prefix}/tf_agent_tp": tp,
                f"{prefix}/tf_agent_fp": fp,
                f"{prefix}/tf_agent_fn": fn,
            })
        return metrics

    def _slice_micro_batch(self, batch, start: int, end: int) -> Dict[str, Any]:
        micro_batch = {
            "input_ids": batch["input_ids"][start:end],
            "attention_mask": batch["attention_mask"][start:end],
            "lm_labels": batch["lm_labels"][start:end],
        }
        # Keep metadata only when text metrics are enabled.
        if self.compute_train_text_metrics:
            for k in ["raw_label", "gold_response_text", "gold_agent_names", "candidate_texts"]:
                if k in batch:
                    micro_batch[k] = batch[k][start:end]
        return micro_batch

    def training_step(self, batch):
        self.fsdp_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        bsz = batch["input_ids"].size(0)
        micro = self.config.data.micro_batch_size_per_gpu
        micro_batches = []
        for start in range(0, bsz, micro):
            end = start + micro
            micro_batches.append(self._slice_micro_batch(batch, start, end))

        stats = self._empty_count_stats(include_text_metrics=self.compute_train_text_metrics)
        n_micro = len(micro_batches)
        for micro_batch in micro_batches:
            out = self._forward_and_compute_core_metrics(
                micro_batch,
                compute_text_metrics=self.compute_train_text_metrics,
            )
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
        out = self._forward_and_compute_core_metrics(
            batch,
            compute_text_metrics=self.compute_val_text_metrics,
        )
        stats = self._empty_count_stats(include_text_metrics=self.compute_val_text_metrics)
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
