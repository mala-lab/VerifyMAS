# -*- coding: utf-8 -*-
# @Author  : qiaohezhe / ChatGPT
# @Date    : 2026-04-24
# @File    : fsdp_abc_agent_joint_trainer.py
# @Desc    : FSDP trainer for joint A/B/C classification + agent attribution classification.
#
# Expected data format:
# {
#   "prompt": "...prompt text ending right before the answer...",
#   "response": "{\"label\":\"A\",\"agent\":\"Solver\"}"
# }
#
# Also supports:
#   "response": {"label":"A","agent":"Solver"}
#   "response": "A"        # only valid for B/C or label-only debugging; A without agent is dropped
#
# Optional candidate mask support:
#   If each row contains data.candidate_agents_key, default "candidate_agents",
#   e.g. {"candidate_agents": ["Solver", "Critic", "Planner"]},
#   agent loss / agent accuracy will be computed after masking invalid candidate agents.
#   If no candidate list is provided, all agents in the global agent vocabulary are valid.
#
# Training target:
#   - cls_head: A / B / C
#   - agent_head: responsible agent, computed only for gold label A
#   - total_loss = cls_loss + model.agent_loss_weight * agent_loss

import os
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import hydra
import pandas as pd
import torch
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
IGNORE_AGENT_ID = -100


# =========================================================
# Data helpers
# =========================================================

def read_rows_from_files(data_files) -> List[Dict[str, Any]]:
    if isinstance(data_files, str):
        files = [data_files]
    elif isinstance(data_files, (list, tuple)):
        files = list(data_files)
    else:
        raise ValueError(f"Unsupported data_files type: {type(data_files)}")

    rows: List[Dict[str, Any]] = []
    for file_path in files:
        path = str(file_path)
        suffix = Path(path).suffix.lower()
        print(f"[DATA] reading file: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file does not exist: {path}")

        if suffix == ".parquet":
            df = pd.read_parquet(path)
            rows.extend(df.to_dict(orient="records"))
            print(f"[DATA] loaded parquet rows={len(df)} from {path}")
        elif suffix == ".jsonl":
            n = 0
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                        n += 1
                    except Exception as e:
                        print(f"[WARN] skip invalid jsonl line {line_no}: {e}")
            print(f"[DATA] loaded jsonl rows={n} from {path}")
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, list):
                raise ValueError(f"JSON file must contain a list of objects: {path}")
            rows.extend(obj)
            print(f"[DATA] loaded json rows={len(obj)} from {path}")
        else:
            raise ValueError(f"Unsupported file format: {path}")

    print(f"[DATA] total loaded rows={len(rows)}")
    return rows


def _clean_agent(agent_value: Any) -> Optional[str]:
    if agent_value is None:
        return None
    agent = str(agent_value).strip()
    if not agent:
        return None
    if agent.lower() in {"none", "null", "nil", "n/a", "na", "__none__"}:
        return None
    return agent


def parse_label_agent_from_response(response_value: Any) -> Tuple[str, Optional[str]]:
    """
    Accept:
      - {"label":"A","agent":"Solver"}
      - "{\"label\":\"A\",\"agent\":\"Solver\"}"
      - "A" / "B" / "C"
    Returns:
      label in {"A","B","C"}, agent as str or None.
    """
    if isinstance(response_value, dict):
        obj = response_value
        label = str(obj.get("label", "")).strip().upper()
        agent = _clean_agent(obj.get("agent", None))
    else:
        text = str(response_value).strip()
        if text.upper() in LABEL_TO_ID:
            label = text.upper()
            agent = None
        else:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"Response JSON must be object, got {type(obj)}")
            label = str(obj.get("label", "")).strip().upper()
            agent = _clean_agent(obj.get("agent", None))

    if label not in LABEL_TO_ID:
        raise ValueError(f"Invalid label: {response_value}")

    return label, agent


def parse_candidate_agents(value: Any) -> Optional[List[str]]:
    """Return normalized candidate agent list, or None if not provided."""
    if value is None:
        return None

    obj = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
        except Exception:
            # Fallback: comma-separated string.
            obj = [x.strip() for x in text.split(",")]

    if isinstance(obj, dict):
        # Common possible schemas: {"agents": [...]}, {"candidate_agents": [...]}
        if "agents" in obj:
            obj = obj["agents"]
        elif "candidate_agents" in obj:
            obj = obj["candidate_agents"]
        else:
            return None

    if not isinstance(obj, (list, tuple, set)):
        return None

    agents = []
    seen = set()
    for x in obj:
        agent = _clean_agent(x)
        if agent is not None and agent not in seen:
            agents.append(agent)
            seen.add(agent)

    return agents if agents else None


def get_candidate_agents_from_row(row: Dict[str, Any], data_config) -> Optional[List[str]]:
    key = getattr(data_config, "candidate_agents_key", "candidate_agents")
    if key in row:
        return parse_candidate_agents(row.get(key))

    # Light fallback for common names. This will not change behavior if absent.
    for fallback_key in ["candidate_agents", "agents", "agent_candidates"]:
        if fallback_key in row:
            return parse_candidate_agents(row.get(fallback_key))
    return None


def get_hidden_size_from_config(config_obj) -> int:
    for key in ["hidden_size", "n_embd", "d_model"]:
        if hasattr(config_obj, key):
            return int(getattr(config_obj, key))
    if hasattr(config_obj, "text_config"):
        for key in ["hidden_size", "n_embd", "d_model"]:
            if hasattr(config_obj.text_config, key):
                return int(getattr(config_obj.text_config, key))
    raise ValueError("Cannot infer hidden size from model config.")


def _maybe_get_list(config_obj, key: str):
    value = getattr(config_obj, key, None)
    if value is None:
        return None
    return list(value)


def build_agent_vocab(raw_rows: List[Dict[str, Any]], data_config) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build global agent vocabulary.

    Priority:
      1. data.agent_vocab_path: JSON list, or JSON dict {agent: id}
      2. data.agent_list: explicit list from Hydra
      3. scan response.agent for label=A and optional candidate_agents fields
    """
    agent_vocab_path = getattr(data_config, "agent_vocab_path", None)
    if agent_vocab_path:
        with open(str(agent_vocab_path), "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            agent_to_id = {str(k): int(v) for k, v in obj.items()}
            id_to_agent = {v: k for k, v in agent_to_id.items()}
        elif isinstance(obj, list):
            agents = [_clean_agent(x) for x in obj]
            agents = [x for x in agents if x is not None]
            agent_to_id = {agent: i for i, agent in enumerate(agents)}
            id_to_agent = {i: agent for agent, i in agent_to_id.items()}
        else:
            raise ValueError("data.agent_vocab_path must contain a JSON list or dict.")
        if len(agent_to_id) == 0:
            raise ValueError("Empty agent vocabulary from data.agent_vocab_path.")
        return agent_to_id, id_to_agent

    explicit_agent_list = _maybe_get_list(data_config, "agent_list")
    if explicit_agent_list is not None:
        agents = []
        seen = set()
        for x in explicit_agent_list:
            agent = _clean_agent(x)
            if agent is not None and agent not in seen:
                agents.append(agent)
                seen.add(agent)
        if len(agents) == 0:
            raise ValueError("data.agent_list is provided but empty after normalization.")
        agent_to_id = {agent: i for i, agent in enumerate(agents)}
        id_to_agent = {i: agent for agent, i in agent_to_id.items()}
        return agent_to_id, id_to_agent

    response_key = getattr(data_config, "response_key", "response")
    agents = set()
    bad_response = 0
    for row in raw_rows:
        if response_key in row and row[response_key] is not None:
            try:
                label, agent = parse_label_agent_from_response(row[response_key])
                if label == "A" and agent is not None:
                    agents.add(agent)
            except Exception:
                bad_response += 1

        candidate_agents = get_candidate_agents_from_row(row, data_config)
        if candidate_agents:
            for agent in candidate_agents:
                agents.add(agent)

    if bad_response > 0:
        print(f"[WARN] build_agent_vocab skipped bad_response rows={bad_response}")

    agents = sorted(agents)
    if len(agents) == 0:
        raise ValueError(
            "Cannot build agent vocabulary. Please ensure label=A responses contain an agent, "
            "or set data.agent_list=[...] / data.agent_vocab_path=..."
        )

    agent_to_id = {agent: i for i, agent in enumerate(agents)}
    id_to_agent = {i: agent for agent, i in agent_to_id.items()}
    return agent_to_id, id_to_agent


# =========================================================
# Dataset / collator
# =========================================================

class ABCJointAgentDataset(Dataset):
    def __init__(self, raw_rows, tokenizer, config, agent_to_id: Dict[str, int]):
        self.tokenizer = tokenizer
        self.agent_to_id = agent_to_id
        self.num_agents = len(agent_to_id)
        self.prompt_key = getattr(config, "prompt_key", "prompt")
        self.response_key = getattr(config, "response_key", "response")
        self.max_prompt_length = int(getattr(config, "max_prompt_length", getattr(config, "max_length", 4096)))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.rows = self._validate_rows(raw_rows, config)
        if len(self.rows) == 0:
            raise ValueError("No valid rows loaded.")

        label_hist = {"A": 0, "B": 0, "C": 0}
        agent_hist = {}
        masked_rows = 0
        for row in self.rows:
            label_hist[row["label"]] += 1
            if row["agent"] is not None:
                agent_hist[row["agent"]] = agent_hist.get(row["agent"], 0) + 1
            if row["candidate_agents"] is not None:
                masked_rows += 1
        print(f"[DATASET] valid rows={len(self.rows)}, label_hist={label_hist}")
        print(f"[DATASET] num_agents={self.num_agents}, A-agent_hist={agent_hist}")
        print(f"[DATASET] rows_with_candidate_mask={masked_rows}/{len(self.rows)}")

    def _validate_rows(self, raw_rows: List[Dict[str, Any]], config) -> List[Dict[str, Any]]:
        valid_rows = []
        dropped = 0
        drop_reasons = {}

        def add_drop(reason: str):
            nonlocal dropped
            dropped += 1
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        for row in raw_rows:
            if self.prompt_key not in row or self.response_key not in row:
                add_drop("missing_prompt_or_response")
                continue

            prompt = row[self.prompt_key]
            response = row[self.response_key]
            if prompt is None or response is None:
                add_drop("none_prompt_or_response")
                continue

            try:
                label, agent = parse_label_agent_from_response(response)
            except Exception:
                add_drop("bad_response")
                continue

            if label == "A":
                if agent is None:
                    add_drop("missing_agent_for_A")
                    continue
                if agent not in self.agent_to_id:
                    add_drop("unknown_agent_for_A")
                    continue
            else:
                # Agent is not supervised for non-A samples.
                agent = None

            candidate_agents = get_candidate_agents_from_row(row, config)
            if candidate_agents is not None:
                candidate_agents = [x for x in candidate_agents if x in self.agent_to_id]
                if label == "A" and agent is not None and agent not in candidate_agents:
                    # Ensure the gold agent is valid even if candidate list is incomplete/noisy.
                    candidate_agents.append(agent)
                if len(candidate_agents) == 0:
                    candidate_agents = None

            valid_rows.append({
                "prompt": str(prompt),
                "label": label,
                "agent": agent,
                "candidate_agents": candidate_agents,
            })

        print(f"[DATASET] dropped={dropped}, reasons={drop_reasons}")
        return valid_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_attention_mask=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if len(input_ids) < 1:
            raise ValueError(f"Empty tokenized prompt at idx={idx}")

        label_id = LABEL_TO_ID[row["label"]]
        if row["label"] == "A":
            agent_id = self.agent_to_id[row["agent"]]
        else:
            agent_id = IGNORE_AGENT_ID

        candidate_mask = [True] * self.num_agents
        if row["candidate_agents"] is not None:
            candidate_mask = [False] * self.num_agents
            for agent in row["candidate_agents"]:
                candidate_mask[self.agent_to_id[agent]] = True
            if row["label"] == "A":
                candidate_mask[agent_id] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_id": label_id,
            "agent_id": agent_id,
            "candidate_mask": candidate_mask,
            "raw_label": row["label"],
            "raw_agent": row["agent"],
        }


class ABCJointAgentCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask = [], []
        label_id, agent_id, candidate_mask = [], [], []
        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            label_id.append(item["label_id"])
            agent_id.append(item["agent_id"])
            candidate_mask.append(item["candidate_mask"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label_id": torch.tensor(label_id, dtype=torch.long),
            "agent_id": torch.tensor(agent_id, dtype=torch.long),
            "candidate_mask": torch.tensor(candidate_mask, dtype=torch.bool),
        }


# =========================================================
# Model
# =========================================================

class ABCJointAgentModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel, hidden_size: int, num_agents: int):
        super().__init__()
        self.base_model = base_model
        self.cls_head = nn.Linear(hidden_size, 3)
        self.agent_head = nn.Linear(hidden_size, num_agents)

    def forward(self, input_ids, attention_mask, use_cache=False):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]
        last_pos = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_idx, last_pos, :]

        cls_logits = self.cls_head(pooled)
        agent_logits = self.agent_head(pooled)
        return cls_logits, agent_logits


# =========================================================
# Trainer
# =========================================================

class FSDPABCJointAgentTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        agent_to_id: Dict[str, int],
        id_to_agent: Dict[int, str],
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.device_name = get_device_name()
        self.agent_to_id = agent_to_id
        self.id_to_agent = id_to_agent
        self.num_agents = len(agent_to_id)
        self.agent_loss_weight = float(getattr(self.config.model, "agent_loss_weight", 1.0))

        self._normalize_config_bsz()
        self._build_dataloader(train_dataset, val_dataset)
        self._build_model_optimizer()

        # Optional class weights:
        # model.cls_class_weights=[0.5,1.0,1.0]
        cls_class_weights = getattr(self.config.model, "cls_class_weights", None)
        if cls_class_weights is not None:
            weight = torch.tensor(list(cls_class_weights), dtype=torch.float32, device=self.device_name)
            if weight.numel() != 3:
                raise ValueError("model.cls_class_weights must contain exactly 3 values for A/B/C.")
            self.cls_loss_fct = nn.CrossEntropyLoss(weight=weight)
            if self.device_mesh.get_rank() == 0:
                print(f"[INFO] using cls_class_weights={weight.detach().cpu().tolist()}")
        else:
            self.cls_loss_fct = nn.CrossEntropyLoss()

        agent_class_weights = getattr(self.config.model, "agent_class_weights", None)
        if agent_class_weights is not None:
            weight = torch.tensor(list(agent_class_weights), dtype=torch.float32, device=self.device_name)
            if weight.numel() != self.num_agents:
                raise ValueError(
                    f"model.agent_class_weights must contain num_agents={self.num_agents} values."
                )
            self.agent_loss_fct = nn.CrossEntropyLoss(weight=weight)
            if self.device_mesh.get_rank() == 0:
                print(f"[INFO] using agent_class_weights={weight.detach().cpu().tolist()}")
        else:
            self.agent_loss_fct = nn.CrossEntropyLoss()

        if self.device_mesh.get_rank() == 0:
            print(self.config)
            print(
                "[INFO] joint trainer: cls_head + agent_head; "
                "agent_loss is computed only on gold label A samples."
            )
            print(f"[INFO] agent_loss_weight={self.agent_loss_weight}")
            print(f"[INFO] num_agents={self.num_agents}, agent_to_id={self.agent_to_id}")

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"[INFO] Normalize batch size by dp={dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )
        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        collator = ABCJointAgentCollator(self.tokenizer)

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()

        # Stable default: avoid DataLoader multiprocessing hang.
        num_workers = int(getattr(self.config.data, "dataloader_num_workers", 0))
        pin_memory = bool(getattr(self.config.data, "pin_memory", False))

        if rank == 0:
            print(f"[INFO] FSDP rank={rank}, world_size={world_size}")
            print(f"[INFO] DataLoader num_workers={num_workers}, pin_memory={pin_memory}")

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=False,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank,
            drop_last=False,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=False,
        )

        if rank == 0:
            print(f"[INFO] train dataloader steps={len(self.train_dataloader)}")
            print(f"[INFO] val dataloader steps={len(self.val_dataloader)}")

    def _build_model_optimizer(self):
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "bf16")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        print(f"\n==== Using torch_dtype: {torch_dtype} ====")

        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config

        if hasattr(self.model_config, "max_position_embeddings"):
            max_prompt_length = int(getattr(self.config.data, "max_prompt_length", getattr(self.config.data, "max_length", 4096)))
            self.model_config.max_position_embeddings = max(self.model_config.max_position_embeddings, max_prompt_length)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings,
            mesh=self.device_mesh,
        )

        with init_context():
            base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
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

            hidden_size = get_hidden_size_from_config(config)
            self.model = ABCJointAgentModel(
                base_model=base_model,
                hidden_size=hidden_size,
                num_agents=self.num_agents,
            )

        if self.config.model.enable_gradient_checkpointing:
            self.model.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        cpu_offload = None
        if self.config.model.fsdp_config.cpu_offload:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

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
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                cast_forward_inputs=True,
            )
            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
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

        if self.device_mesh.get_rank() == 0:
            print(
                f"[INFO] steps/epoch={self.steps_per_epoch}, "
                f"epochs={self.config.trainer.total_epochs}, "
                f"total_steps={self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    @staticmethod
    def _mask_agent_logits(agent_logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        very_neg = torch.finfo(agent_logits.dtype).min
        return agent_logits.masked_fill(~candidate_mask, very_neg)

    def _compute_loss_and_metrics(self, batch):
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        label_id = batch["label_id"].to(self.device_name)
        agent_id = batch["agent_id"].to(self.device_name)
        candidate_mask = batch["candidate_mask"].to(self.device_name)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            cls_logits, agent_logits = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            masked_agent_logits = self._mask_agent_logits(agent_logits, candidate_mask)

            cls_loss = self.cls_loss_fct(cls_logits, label_id)
            a_mask = label_id == LABEL_TO_ID["A"]
            agent_count = a_mask.long().sum()

            if a_mask.any():
                agent_loss = self.agent_loss_fct(masked_agent_logits[a_mask], agent_id[a_mask])
            else:
                agent_loss = cls_logits.new_zeros(())

            loss = cls_loss + self.agent_loss_weight * agent_loss

            cls_pred = cls_logits.argmax(dim=-1)
            agent_pred = masked_agent_logits.argmax(dim=-1)

            cls_correct = (cls_pred == label_id).float().sum()
            cls_count = torch.tensor(label_id.numel(), dtype=torch.float32, device=label_id.device)

            if a_mask.any():
                agent_correct = (agent_pred[a_mask] == agent_id[a_mask]).float().sum()
            else:
                agent_correct = cls_logits.new_zeros(())
            agent_count_float = agent_count.to(dtype=torch.float32)

            # Joint correctness: B/C only require label correctness; A requires both label and agent correctness.
            agent_ok_or_not_needed = (~a_mask) | (agent_pred == agent_id)
            joint_correct = ((cls_pred == label_id) & agent_ok_or_not_needed).float().sum()
            joint_count = cls_count

            # Per-class accuracy for debugging label collapse.
            class_stats = {}
            for label_name, label_idx in LABEL_TO_ID.items():
                mask = label_id == label_idx
                class_stats[f"correct_{label_name}"] = (cls_pred[mask] == label_id[mask]).float().sum() if mask.any() else cls_logits.new_zeros(())
                class_stats[f"count_{label_name}"] = mask.float().sum()

        return {
            "loss": loss,
            "cls_loss": cls_loss.detach(),
            "agent_loss": agent_loss.detach(),
            "cls_correct": cls_correct.detach(),
            "cls_count": cls_count.detach(),
            "agent_correct": agent_correct.detach(),
            "agent_count": agent_count_float.detach(),
            "joint_correct": joint_correct.detach(),
            "joint_count": joint_count.detach(),
            "correct_A": class_stats["correct_A"].detach(),
            "count_A": class_stats["count_A"].detach(),
            "correct_B": class_stats["correct_B"].detach(),
            "count_B": class_stats["count_B"].detach(),
            "correct_C": class_stats["correct_C"].detach(),
            "count_C": class_stats["count_C"].detach(),
        }

    @staticmethod
    def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        return num / den.clamp_min(1.0)

    def training_step(self, batch):
        self.fsdp_model.train()
        self.optimizer.zero_grad()

        micro_batches = []
        bsz = batch["input_ids"].size(0)
        micro = self.config.data.micro_batch_size_per_gpu
        for start in range(0, bsz, micro):
            end = start + micro
            micro_batches.append({
                "input_ids": batch["input_ids"][start:end],
                "attention_mask": batch["attention_mask"][start:end],
                "label_id": batch["label_id"][start:end],
                "agent_id": batch["agent_id"][start:end],
                "candidate_mask": batch["candidate_mask"][start:end],
            })

        n_micro_batches = len(micro_batches)
        loss_avg = 0.0
        cls_loss_avg = 0.0
        agent_loss_weighted_sum = 0.0
        agent_loss_count = 0.0

        sums = {
            "cls_correct": 0.0,
            "cls_count": 0.0,
            "agent_correct": 0.0,
            "agent_count": 0.0,
            "joint_correct": 0.0,
            "joint_count": 0.0,
            "correct_A": 0.0,
            "count_A": 0.0,
            "correct_B": 0.0,
            "count_B": 0.0,
            "correct_C": 0.0,
            "count_C": 0.0,
        }

        for micro_batch in micro_batches:
            out = self._compute_loss_and_metrics(micro_batch)
            (out["loss"] / n_micro_batches).backward()

            loss_avg += out["loss"].detach().item() / n_micro_batches
            cls_loss_avg += out["cls_loss"].detach().item() / n_micro_batches

            acount = out["agent_count"].detach().item()
            if acount > 0:
                agent_loss_weighted_sum += out["agent_loss"].detach().item() * acount
                agent_loss_count += acount

            for key in sums:
                sums[key] += out[key].detach().item()

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]

        # Reduce average losses across ranks.
        loss_tensors = {
            "train/loss": torch.tensor(loss_avg, device=self.device_name),
            "train/cls_loss": torch.tensor(cls_loss_avg, device=self.device_name),
        }
        for k in loss_tensors:
            torch.distributed.all_reduce(loss_tensors[k], op=torch.distributed.ReduceOp.AVG)

        # Reduce weighted agent loss and counts across ranks.
        reduced = {k: torch.tensor(v, dtype=torch.float32, device=self.device_name) for k, v in sums.items()}
        reduced["agent_loss_weighted_sum"] = torch.tensor(agent_loss_weighted_sum, dtype=torch.float32, device=self.device_name)
        reduced["agent_loss_count_for_loss"] = torch.tensor(agent_loss_count, dtype=torch.float32, device=self.device_name)
        for k in reduced:
            torch.distributed.all_reduce(reduced[k], op=torch.distributed.ReduceOp.SUM)

        agent_loss_metric = self._safe_div(reduced["agent_loss_weighted_sum"], reduced["agent_loss_count_for_loss"])

        return {
            "train/loss": loss_tensors["train/loss"].item(),
            "train/cls_loss": loss_tensors["train/cls_loss"].item(),
            "train/agent_loss": agent_loss_metric.item(),
            "train/cls_acc": self._safe_div(reduced["cls_correct"], reduced["cls_count"]).item(),
            "train/agent_acc": self._safe_div(reduced["agent_correct"], reduced["agent_count"]).item(),
            "train/joint_acc": self._safe_div(reduced["joint_correct"], reduced["joint_count"]).item(),
            "train/acc_A": self._safe_div(reduced["correct_A"], reduced["count_A"]).item(),
            "train/acc_B": self._safe_div(reduced["correct_B"], reduced["count_B"]).item(),
            "train/acc_C": self._safe_div(reduced["correct_C"], reduced["count_C"]).item(),
            "train/agent_count": reduced["agent_count"].item(),
            "train/lr(1e-3)": lr * 1e3,
        }

    @torch.no_grad()
    def validation_step(self, batch):
        self.fsdp_model.eval()
        out = self._compute_loss_and_metrics(batch)

        # For validation, return globally reduced sums for exact count-weighted accuracies.
        keys_to_sum = [
            "cls_correct", "cls_count", "agent_correct", "agent_count", "joint_correct", "joint_count",
            "correct_A", "count_A", "correct_B", "count_B", "correct_C", "count_C",
        ]
        reduced = {}
        for key in keys_to_sum:
            val = out[key].detach().clone()
            torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)
            reduced[key] = val

        loss_keys = ["loss", "cls_loss", "agent_loss"]
        for key in loss_keys:
            val = out[key].detach().clone()
            torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.AVG)
            reduced[key] = val

        return reduced

    def save_checkpoint(self, step):
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        fsdp_strategy = self.config.model.strategy

        if fsdp_strategy == "fsdp":
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()
        elif fsdp_strategy == "fsdp2":
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            torch.save(state_dict, os.path.join(path, "model.pt"))
            self.model_config.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

            meta = {
                "label_to_id": LABEL_TO_ID,
                "id_to_label": ID_TO_LABEL,
                "agent_to_id": self.agent_to_id,
                "id_to_agent": {str(k): v for k, v in self.id_to_agent.items()},
                "num_agents": self.num_agents,
                "checkpoint_type": "abc_agent_joint_classifier",
                "agent_loss_scope": "gold_label_A_only",
                "agent_loss_weight": self.agent_loss_weight,
                "candidate_mask": "optional: uses row candidate_agents if provided, otherwise all agents are valid",
            }
            with open(os.path.join(path, "joint_head_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            with open(os.path.join(path, "agent_vocab.json"), "w", encoding="utf-8") as f:
                json.dump(self.agent_to_id, f, ensure_ascii=False, indent=2)

        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def _aggregate_val_metrics(self, val_metrics: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        if not val_metrics:
            return {}

        # Losses are averaged per validation step, same as your original implementation.
        loss = torch.mean(torch.stack([x["loss"] for x in val_metrics])).item()
        cls_loss = torch.mean(torch.stack([x["cls_loss"] for x in val_metrics])).item()
        agent_loss = torch.mean(torch.stack([x["agent_loss"] for x in val_metrics])).item()

        def sum_key(key):
            return torch.stack([x[key] for x in val_metrics]).sum()

        cls_correct = sum_key("cls_correct")
        cls_count = sum_key("cls_count")
        agent_correct = sum_key("agent_correct")
        agent_count = sum_key("agent_count")
        joint_correct = sum_key("joint_correct")
        joint_count = sum_key("joint_count")

        metric = {
            "val/loss": loss,
            "val/cls_loss": cls_loss,
            "val/agent_loss": agent_loss,
            "val/cls_acc": self._safe_div(cls_correct, cls_count).item(),
            "val/agent_acc": self._safe_div(agent_correct, agent_count).item(),
            "val/joint_acc": self._safe_div(joint_correct, joint_count).item(),
            "val/agent_count": agent_count.item(),
        }

        for label_name in ["A", "B", "C"]:
            correct = sum_key(f"correct_{label_name}")
            count = sum_key(f"count_{label_name}")
            metric[f"val/acc_{label_name}"] = self._safe_div(correct, count).item()
            metric[f"val/count_{label_name}"] = count.item()

        return metric

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
        print(f"[INFO] Total training steps: {self.total_training_steps}")

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in self.train_dataloader:
                global_step += 1
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                    if global_step <= 3 or global_step % 10 == 0:
                        print(f"[TRAIN] step={global_step} metric={metric}")

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = self.config.trainer.test_freq > 0 and global_step % self.config.trainer.test_freq == 0
                is_save_step = self.config.trainer.save_freq > 0 and global_step % self.config.trainer.save_freq == 0

                if is_last_step or is_valid_step:
                    val_metrics = []
                    for val_data in self.val_dataloader:
                        out = self.validation_step(val_data)
                        val_metrics.append(out)

                    if rank == 0:
                        metric = self._aggregate_val_metrics(val_metrics)
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                        print(f"[VAL] step={global_step} metric={metric}")
                    torch.distributed.barrier()

                if is_last_step or is_save_step:
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"[INFO] Final validation metrics: {last_valid_metric}")
                    return


# =========================================================
# Run
# =========================================================

def create_dataset_from_rows(raw_rows, data_config, tokenizer, agent_to_id):
    return ABCJointAgentDataset(
        raw_rows=raw_rows,
        tokenizer=tokenizer,
        config=data_config,
        agent_to_id=agent_to_id,
    )


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    ulysses_sp_size = int(getattr(config, "ulysses_sequence_parallel_size", 1))
    if ulysses_sp_size != 1:
        raise ValueError("This trainer does not support ulysses sequence parallel. Set ulysses_sequence_parallel_size=1.")
    if getattr(config, "use_remove_padding", False):
        raise ValueError("This trainer does not use remove_padding. Set use_remove_padding=False.")

    device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",),
    )

    from verl.utils import hf_tokenizer

    if rank == 0:
        print("[INFO] copying/loading tokenizer...")
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        print("[INFO] reading train/val rows...")
    train_rows = read_rows_from_files(config.data.train_files)
    val_rows = read_rows_from_files(config.data.val_files)

    # To avoid val unseen-agent crash, default builds vocab from train+val.
    # For stricter experiments, set data.build_agent_vocab_from_val=false and provide data.agent_list if needed.
    build_from_val = bool(getattr(config.data, "build_agent_vocab_from_val", True))
    vocab_rows = train_rows + val_rows if build_from_val else train_rows
    agent_to_id, id_to_agent = build_agent_vocab(vocab_rows, config.data)

    if rank == 0:
        print(f"[INFO] built agent vocab with num_agents={len(agent_to_id)}")
        print(f"[INFO] agent_to_id={agent_to_id}")
        print("[INFO] creating datasets...")
    train_dataset = create_dataset_from_rows(train_rows, config.data, tokenizer, agent_to_id)
    val_dataset = create_dataset_from_rows(val_rows, config.data, tokenizer, agent_to_id)

    if rank == 0:
        print("===== DEBUG DATASET =====")
        print("train size:", len(train_dataset))
        print("val size:", len(val_dataset))
        for i in range(min(2, len(train_dataset))):
            item = train_dataset[i]
            print(f"--- sample {i} ---")
            print("label_id:", item["label_id"])
            print("agent_id:", item["agent_id"])
            print("raw_label:", item["raw_label"])
            print("raw_agent:", item["raw_agent"])
            print("candidate_mask_true:", sum(item["candidate_mask"]))
            print("input len:", len(item["input_ids"]))
            print("decoded tail:", tokenizer.decode(item["input_ids"][-120:]))
        print("===== END DEBUG =====")

    trainer = FSDPABCJointAgentTrainer(
        config=config,
        device_mesh=device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_to_id=agent_to_id,
        id_to_agent=id_to_agent,
    )
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


if __name__ == "__main__":
    main()
