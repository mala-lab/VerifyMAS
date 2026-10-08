# -*- coding: utf-8 -*-
# @Author  : OpenAI
# @Date    : 2026-04-23
# @File    : fsdp_joint_abc_local_candidate_trainer_v2.py
# @Desc    : Safer/optimized FSDP trainer for joint A/B/C classification + true local-candidate agent classification.

import os
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
IGNORE_AGENT_INDEX = -100


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

        if suffix == ".parquet":
            df = pd.read_parquet(path)
            rows.extend(df.to_dict(orient="records"))
        elif suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    rows.extend(obj)
                else:
                    raise ValueError(f"JSON file must contain a list of objects: {path}")
        else:
            raise ValueError(f"Unsupported file format: {path}")
    return rows


def parse_joint_response(response_value: Any) -> Tuple[str, Optional[str]]:
    if isinstance(response_value, dict):
        obj = response_value
    else:
        text = str(response_value).strip()
        if text in {"A", "B", "C"}:
            return text, None
        try:
            obj = json.loads(text)
        except Exception as e:
            raise ValueError(f"Cannot parse response as joint JSON: {response_value}") from e

    if not isinstance(obj, dict):
        raise ValueError(f"Response must be a dict/JSON object, got: {type(obj)}")

    label = str(obj.get("label", "")).strip().upper()
    if label not in LABEL_TO_ID:
        raise ValueError(f"Invalid label in response: {obj}")

    agent = obj.get("agent", None)
    if agent is None:
        return label, None
    agent = str(agent).strip()
    if not agent or agent.lower() in {"none", "null"}:
        return label, None
    return label, agent


def parse_candidate_agents_from_prompt(prompt: str) -> List[str]:
    marker = "Candidate agents:"
    idx = prompt.find(marker)
    if idx < 0:
        return []

    tail = prompt[idx + len(marker):]
    agents = []
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            if agents:
                break
            continue
        if line.startswith("- "):
            agent = line[2:].strip()
            if agent and agent.lower() != "none":
                agents.append(agent)
        else:
            if agents:
                break

    deduped = []
    seen = set()
    for a in agents:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    return deduped


def get_hidden_size_from_config(config_obj) -> int:
    for key in ["hidden_size", "n_embd", "d_model"]:
        if hasattr(config_obj, key):
            return int(getattr(config_obj, key))
    if hasattr(config_obj, "text_config"):
        for key in ["hidden_size", "n_embd", "d_model"]:
            if hasattr(config_obj.text_config, key):
                return int(getattr(config_obj.text_config, key))
    raise ValueError("Cannot infer hidden size from model config.")


class JointPromptLocalCandidateDataset(Dataset):
    """Pretokenize prompts/candidates at init time to avoid worker-time tokenizer calls."""
    def __init__(self, raw_rows, tokenizer, config):
        self.tokenizer = tokenizer
        self.prompt_key = getattr(config, "prompt_key", "prompt")
        self.response_key = getattr(config, "response_key", "response")
        self.candidate_agents_key = getattr(config, "candidate_agents_key", "candidate_agents")
        self.max_prompt_length = int(getattr(config, "max_prompt_length", getattr(config, "max_length", 4096)))
        self.max_candidate_length = int(getattr(config, "max_candidate_length", 16))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have eos_token_id.")

        t0 = time.time()
        self.rows = self._build_rows(raw_rows)
        if len(self.rows) == 0:
            raise ValueError("No valid rows loaded.")
        print(f"[DATA] dataset pretokenized in {time.time() - t0:.2f}s")

    def _tokenize_candidate_name(self, agent_name: str) -> Tuple[List[int], List[int]]:
        enc = self.tokenizer(
            str(agent_name),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_candidate_length,
            return_attention_mask=True,
        )
        ids = list(enc["input_ids"])
        mask = list(enc["attention_mask"])
        if len(ids) == 0:
            ids = [self.tokenizer.eos_token_id]
            mask = [1]
        return ids, mask

    def _tokenize_prompt(self, prompt: str) -> Tuple[List[int], List[int]]:
        enc = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_attention_mask=True,
        )
        ids = list(enc["input_ids"])
        mask = list(enc["attention_mask"])
        if len(ids) == 0:
            raise ValueError("Empty tokenized prompt")
        return ids, mask

    def _build_rows(self, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_rows = []
        dropped = 0
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
                label, agent_name = parse_joint_response(response)
            except Exception:
                dropped += 1
                continue

            if self.candidate_agents_key in row and row[self.candidate_agents_key] is not None:
                candidate_agents = []
                for a in row[self.candidate_agents_key]:
                    if a is None:
                        continue
                    a = str(a).strip()
                    if a:
                        candidate_agents.append(a)
            else:
                candidate_agents = parse_candidate_agents_from_prompt(prompt)

            deduped = []
            seen = set()
            for a in candidate_agents:
                if a not in seen:
                    seen.add(a)
                    deduped.append(a)
            candidate_agents = deduped

            if len(candidate_agents) == 0:
                dropped += 1
                continue

            if label == "A":
                if agent_name is None:
                    dropped += 1
                    continue
                if agent_name not in candidate_agents:
                    candidate_agents.append(agent_name)
            else:
                agent_name = None

            try:
                prompt_ids, prompt_mask = self._tokenize_prompt(prompt)
            except Exception:
                dropped += 1
                continue

            candidate_token_ids = []
            candidate_attention_masks = []
            for a in candidate_agents:
                ids, mask = self._tokenize_candidate_name(a)
                candidate_token_ids.append(ids)
                candidate_attention_masks.append(mask)

            if label == "A":
                try:
                    agent_local_id = candidate_agents.index(agent_name)
                except ValueError:
                    dropped += 1
                    continue
            else:
                agent_local_id = IGNORE_AGENT_INDEX

            valid_rows.append(
                {
                    "input_ids": prompt_ids,
                    "attention_mask": prompt_mask,
                    "label_id": LABEL_TO_ID[label],
                    "agent_local_id": agent_local_id,
                    "candidate_agents": candidate_agents,
                    "candidate_token_ids": candidate_token_ids,
                    "candidate_attention_masks": candidate_attention_masks,
                    "raw_label": label,
                    "raw_agent": agent_name,
                }
            )

        print(f"[DATA] loaded {len(valid_rows)} valid rows, dropped {dropped} invalid rows")
        return valid_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class JointPromptLocalCandidateCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        prompt_max_len = max(len(x["input_ids"]) for x in batch)
        num_candidates_max = max(len(x["candidate_agents"]) for x in batch)
        cand_token_max_len = max(max(len(ids) for ids in x["candidate_token_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        label_id = []
        agent_local_id = []
        candidate_input_ids = []
        candidate_attention_mask = []
        candidate_mask = []
        candidate_agents = []

        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pad_len = prompt_max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            label_id.append(item["label_id"])
            agent_local_id.append(item["agent_local_id"])

            local_agent_names = list(item["candidate_agents"])
            local_ids = []
            local_masks = []
            local_valid_mask = []
            for ids_i, mask_i in zip(item["candidate_token_ids"], item["candidate_attention_masks"]):
                pad_c = cand_token_max_len - len(ids_i)
                local_ids.append(ids_i + [pad_id] * pad_c)
                local_masks.append(mask_i + [0] * pad_c)
                local_valid_mask.append(1)

            while len(local_ids) < num_candidates_max:
                local_ids.append([pad_id] * cand_token_max_len)
                local_masks.append([0] * cand_token_max_len)
                local_valid_mask.append(0)
                local_agent_names.append(None)

            candidate_input_ids.append(local_ids)
            candidate_attention_mask.append(local_masks)
            candidate_mask.append(local_valid_mask)
            candidate_agents.append(local_agent_names)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label_id": torch.tensor(label_id, dtype=torch.long),
            "agent_local_id": torch.tensor(agent_local_id, dtype=torch.long),
            "candidate_input_ids": torch.tensor(candidate_input_ids, dtype=torch.long),
            "candidate_attention_mask": torch.tensor(candidate_attention_mask, dtype=torch.long),
            "candidate_mask": torch.tensor(candidate_mask, dtype=torch.bool),
            "candidate_agents": candidate_agents,
        }


class JointABCLocalCandidateModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.cls_head = nn.Linear(hidden_size, 3)
        self.agent_prompt_proj = nn.Linear(hidden_size, hidden_size)
        self.agent_candidate_proj = nn.Linear(hidden_size, hidden_size)

    def _encode_prompt(self, input_ids, attention_mask, use_cache=False):
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
        return pooled

    def _encode_candidates(self, candidate_input_ids, candidate_attention_mask):
        emb_layer = self.base_model.get_input_embeddings()
        candidate_emb = emb_layer(candidate_input_ids)
        mask = candidate_attention_mask.unsqueeze(-1).to(candidate_emb.dtype)
        summed = (candidate_emb * mask).sum(dim=2)
        denom = mask.sum(dim=2).clamp(min=1.0)
        mean_emb = summed / denom
        return mean_emb

    def forward(self, input_ids, attention_mask, candidate_input_ids, candidate_attention_mask, candidate_mask, use_cache=False):
        pooled = self._encode_prompt(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
        cls_logits = self.cls_head(pooled)

        candidate_repr = self._encode_candidates(candidate_input_ids=candidate_input_ids, candidate_attention_mask=candidate_attention_mask)
        prompt_repr = self.agent_prompt_proj(pooled).unsqueeze(1)
        candidate_repr = self.agent_candidate_proj(candidate_repr)
        agent_logits = (prompt_repr * candidate_repr).sum(dim=-1)

        very_neg = torch.finfo(agent_logits.dtype).min
        agent_logits = agent_logits.masked_fill(~candidate_mask, very_neg)
        return cls_logits, agent_logits


class FSDPJointABCLocalCandidateTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.device_name = get_device_name()
        self.lambda_agent = float(getattr(self.config.model, "lambda_agent", 1.0))
        self.num_workers = int(getattr(self.config.data, "num_workers", 0))
        self.pin_memory = bool(getattr(self.config.data, "pin_memory", True))
        self.persistent_workers = bool(getattr(self.config.data, "persistent_workers", False)) and self.num_workers > 0

        self._normalize_config_bsz()
        self._build_dataloader(train_dataset, val_dataset)
        self._build_model_optimizer()
        self.cls_loss_fct = nn.CrossEntropyLoss()

        if self.device_mesh.get_rank() == 0:
            print(self.config)
            print("[INFO] local-candidate agent classification is enabled.")
            print(f"[INFO] dataloader num_workers={self.num_workers}, pin_memory={self.pin_memory}, persistent_workers={self.persistent_workers}")

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")
        assert self.config.data.train_batch_size % dp_size == 0
        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _make_dataloader(self, dataset, batch_size, sampler, collator, drop_last):
        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
        return DataLoader(**kwargs)

    def _build_dataloader(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        collator = JointPromptLocalCandidateCollator(self.tokenizer)

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        if rank == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)

        self.train_dataloader = self._make_dataloader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            collator=collator,
            drop_last=True,
        )
        self.val_dataloader = self._make_dataloader(
            dataset=self.val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            collator=collator,
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
        print(f"\n==== Using torch_dtype: {torch_dtype} ====")

        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            max_prompt_length = int(getattr(self.config.data, "max_prompt_length", getattr(self.config.data, "max_length", 4096)))
            self.model_config.max_position_embeddings = max(self.model_config.max_position_embeddings, max_prompt_length)

        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)
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
            self.model = JointABCLocalCandidateModel(base_model=base_model, hidden_size=hidden_size)

        if self.config.model.enable_gradient_checkpointing:
            self.model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

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
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(self.optimizer, num_warmup_steps, self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_metrics(self, batch):
        input_ids = batch["input_ids"].to(self.device_name, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device_name, non_blocking=True)
        label_id = batch["label_id"].to(self.device_name, non_blocking=True)
        agent_local_id = batch["agent_local_id"].to(self.device_name, non_blocking=True)
        candidate_input_ids = batch["candidate_input_ids"].to(self.device_name, non_blocking=True)
        candidate_attention_mask = batch["candidate_attention_mask"].to(self.device_name, non_blocking=True)
        candidate_mask = batch["candidate_mask"].to(self.device_name, non_blocking=True)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            cls_logits, agent_logits = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                candidate_input_ids=candidate_input_ids,
                candidate_attention_mask=candidate_attention_mask,
                candidate_mask=candidate_mask,
                use_cache=False,
            )
            cls_loss = self.cls_loss_fct(cls_logits, label_id)
            entail_mask = (label_id == LABEL_TO_ID["A"])
            if entail_mask.any():
                agent_loss = F.cross_entropy(agent_logits[entail_mask], agent_local_id[entail_mask])
            else:
                agent_loss = cls_loss.new_zeros([])
            loss = cls_loss + self.lambda_agent * agent_loss

            cls_pred = cls_logits.argmax(dim=-1)
            agent_pred = agent_logits.argmax(dim=-1)
            cls_acc = (cls_pred == label_id).float().mean()
            if entail_mask.any():
                entail_agent_acc = (agent_pred[entail_mask] == agent_local_id[entail_mask]).float().mean()
                entail_joint_acc = ((cls_pred[entail_mask] == label_id[entail_mask]) & (agent_pred[entail_mask] == agent_local_id[entail_mask])).float().mean()
            else:
                entail_agent_acc = torch.zeros([], device=cls_logits.device)
                entail_joint_acc = torch.zeros([], device=cls_logits.device)
            joint_correct = (cls_pred == label_id)
            if entail_mask.any():
                joint_correct = joint_correct.clone()
                joint_correct[entail_mask] = joint_correct[entail_mask] & (agent_pred[entail_mask] == agent_local_id[entail_mask])
            joint_acc = joint_correct.float().mean()

        return {
            "loss": loss,
            "cls_loss": cls_loss.detach(),
            "agent_loss": agent_loss.detach(),
            "cls_acc": cls_acc.detach(),
            "agent_acc": entail_agent_acc.detach(),
            "joint_acc": joint_acc.detach(),
            "entail_agent_acc": entail_agent_acc.detach(),
            "entail_joint_acc": entail_joint_acc.detach(),
        }

    def training_step(self, batch):
        self.fsdp_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        bsz = batch["input_ids"].size(0)
        micro = self.config.data.micro_batch_size_per_gpu
        n_micro_batches = (bsz + micro - 1) // micro

        step_loss = 0.0
        step_cls_loss = 0.0
        step_agent_loss = 0.0
        step_cls_acc = 0.0
        step_agent_acc = 0.0
        step_joint_acc = 0.0
        step_entail_agent_acc = 0.0
        step_entail_joint_acc = 0.0

        for start in range(0, bsz, micro):
            end = start + micro
            micro_batch = {k: (v[start:end] if torch.is_tensor(v) else v[start:end]) for k, v in batch.items()}
            out = self._compute_loss_and_metrics(micro_batch)
            (out["loss"] / n_micro_batches).backward()

            step_loss += out["loss"].item() / n_micro_batches
            step_cls_loss += out["cls_loss"].item() / n_micro_batches
            step_agent_loss += out["agent_loss"].item() / n_micro_batches
            step_cls_acc += out["cls_acc"].item() / n_micro_batches
            step_agent_acc += out["agent_acc"].item() / n_micro_batches
            step_joint_acc += out["joint_acc"].item() / n_micro_batches
            step_entail_agent_acc += out["entail_agent_acc"].item() / n_micro_batches
            step_entail_joint_acc += out["entail_joint_acc"].item() / n_micro_batches

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.step()

        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]

        metrics = {
            "train/loss": torch.tensor(step_loss, device=self.device_name),
            "train/cls_loss": torch.tensor(step_cls_loss, device=self.device_name),
            "train/agent_loss": torch.tensor(step_agent_loss, device=self.device_name),
            "train/cls_acc": torch.tensor(step_cls_acc, device=self.device_name),
            "train/agent_acc": torch.tensor(step_agent_acc, device=self.device_name),
            "train/joint_acc": torch.tensor(step_joint_acc, device=self.device_name),
            "train/entail_agent_acc": torch.tensor(step_entail_agent_acc, device=self.device_name),
            "train/entail_joint_acc": torch.tensor(step_entail_joint_acc, device=self.device_name),
        }
        for k in metrics:
            torch.distributed.all_reduce(metrics[k], op=torch.distributed.ReduceOp.AVG)

        return {
            "train/loss": metrics["train/loss"].item(),
            "train/cls_loss": metrics["train/cls_loss"].item(),
            "train/agent_loss": metrics["train/agent_loss"].item(),
            "train/cls_acc": metrics["train/cls_acc"].item(),
            "train/agent_acc": metrics["train/agent_acc"].item(),
            "train/joint_acc": metrics["train/joint_acc"].item(),
            "train/entail_agent_acc": metrics["train/entail_agent_acc"].item(),
            "train/entail_joint_acc": metrics["train/entail_joint_acc"].item(),
            "train/lr(1e-3)": lr * 1e3,
        }

    @torch.no_grad()
    def validation_step(self, batch):
        self.fsdp_model.eval()
        out = self._compute_loss_and_metrics(batch)
        reduced = {}
        for key in ["loss", "cls_loss", "agent_loss", "cls_acc", "agent_acc", "joint_acc", "entail_agent_acc", "entail_joint_acc"]:
            val = out[key]
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
                "checkpoint_type": "joint_abc_local_candidate_classifier",
                "agent_mode": "true_local_candidate",
                "max_candidate_length": int(getattr(self.config.data, "max_candidate_length", 16)),
                "lambda_agent": self.lambda_agent,
            }
            with open(os.path.join(path, "joint_head_meta.json"), "w", encoding="utf-8") as f:
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
            epoch_start = time.time()
            first_batch_time = None
            for data in self.train_dataloader:
                if first_batch_time is None:
                    first_batch_time = time.time() - epoch_start
                    if rank == 0:
                        print(f"[INFO] first batch ready after {first_batch_time:.2f}s")
                global_step += 1
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = self.config.trainer.test_freq > 0 and global_step % self.config.trainer.test_freq == 0
                is_save_step = self.config.trainer.save_freq > 0 and global_step % self.config.trainer.save_freq == 0

                if is_last_step or is_valid_step:
                    val_metrics = []
                    for val_data in self.val_dataloader:
                        out = self.validation_step(val_data)
                        val_metrics.append(out)
                    if rank == 0:
                        keys = ["loss", "cls_loss", "agent_loss", "cls_acc", "agent_acc", "joint_acc", "entail_agent_acc", "entail_joint_acc"]
                        metric = {f"val/{key}": torch.mean(torch.stack([x[key] for x in val_metrics])).item() for key in keys}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or is_save_step:
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def create_dataset_from_rows(raw_rows, data_config, tokenizer):
    return JointPromptLocalCandidateDataset(raw_rows=raw_rows, tokenizer=tokenizer, config=data_config)


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    ulysses_sp_size = int(getattr(config, "ulysses_sequence_parallel_size", 1))
    if ulysses_sp_size != 1:
        raise ValueError("This simplified joint trainer does not support ulysses sequence parallel. Set ulysses_sequence_parallel_size=1.")
    if getattr(config, "use_remove_padding", False):
        raise ValueError("This simplified joint trainer does not use remove_padding. Set use_remove_padding=False.")

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
            print("agent:", item["raw_agent"])
            print("num candidates:", len(item["candidate_agents"]))
            print("candidates:", item["candidate_agents"])
            print("input len:", len(item["input_ids"]))
            print("decoded tail:", tokenizer.decode(item["input_ids"][-120:]))
        print("===== END DEBUG =====")

    trainer = FSDPJointABCLocalCandidateTrainer(
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
