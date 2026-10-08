# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @Date    : 2026-04-23
# @File    : fsdp_joint_abc_cls_agent_lm_trainer_memsave.py
# @Desc    : Memory-optimized FSDP trainer:
#            - label (A/B/C) uses 3-class classification loss
#            - agent uses LM loss (generate agent text only)
#            - avoids output_hidden_states=True on all layers
#            - avoids materializing full [B, S, V] logits for LM loss
#
# Expected data format (jsonl / parquet / json):
# {
#   "prompt": "...prompt text ending right before the answer...",
#   "response": "{\"label\":\"A\",\"agent\":\"Solver\"}"
# }
# or
# {
#   "prompt": "...",
#   "response": {"label":"A","agent":"Solver"}
# }
#
# Candidate agents are optional:
# {
#   "candidate_agents": ["RoleAssigner", "Solver", "Evaluator"]
# }
#
# If candidate_agents is missing, the dataset will try to parse them from:
#   Candidate agents:
#   - RoleAssigner
#   - Solver
#   - Evaluator

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
# General helpers
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
    """
    Accept:
      - {"label":"A","agent":"Solver"}
      - "{\"label\":\"A\",\"agent\":\"Solver\"}"
      - "A"  (legacy fallback -> agent=None)
    Returns:
      (label, agent_name_or_none)
    """
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


# =========================================================
# Dataset
# =========================================================

class JointClsAgentLMDataset(Dataset):
    """
    For each sample:
      - prompt is model input prefix
      - cls target = A/B/C
      - lm target = agent text only
          * label=A => gold agent name
          * label=B/C => none_agent_text
    """

    def __init__(self, raw_rows, tokenizer, config):
        self.tokenizer = tokenizer
        self.prompt_key = getattr(config, "prompt_key", "prompt")
        self.response_key = getattr(config, "response_key", "response")
        self.candidate_agents_key = getattr(config, "candidate_agents_key", "candidate_agents")

        self.max_length = int(getattr(config, "max_length", 2048))
        self.max_prompt_length = int(getattr(config, "max_prompt_length", self.max_length))
        self.max_target_length = int(getattr(config, "max_target_length", 16))

        self.none_agent_text = str(
            getattr(config, "none_agent_text", getattr(config, "none_agent_name", "__NONE__"))
        )
        self.append_eos_to_target = bool(getattr(config, "append_eos_to_target", True))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have eos_token_id.")

        self.effective_max_prompt_length = min(
            self.max_prompt_length,
            self.max_length - self.max_target_length,
        )
        if self.effective_max_prompt_length <= 0:
            raise ValueError(
                f"max_length ({self.max_length}) must be larger than max_target_length ({self.max_target_length})."
            )

        self.rows = self._validate_and_normalize_rows(raw_rows)
        if len(self.rows) == 0:
            raise ValueError("No valid rows loaded.")

    def _validate_and_normalize_rows(self, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

            if label == "A":
                if agent_name is None:
                    dropped += 1
                    continue
                if agent_name not in candidate_agents:
                    candidate_agents.append(agent_name)
                target_text = agent_name
            else:
                agent_name = None
                target_text = self.none_agent_text

            valid_rows.append(
                {
                    "prompt": prompt,
                    "label": label,
                    "label_id": LABEL_TO_ID[label],
                    "agent_name": agent_name,
                    "target_text": target_text,
                    "candidate_agents": candidate_agents,
                }
            )

        print(f"[DATA] loaded {len(valid_rows)} valid rows, dropped {dropped} invalid rows")
        return valid_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        prompt = row["prompt"]
        label_id = row["label_id"]
        target_text = row["target_text"]

        prompt_enc = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.effective_max_prompt_length,
            return_attention_mask=True,
        )

        target_enc = self.tokenizer(
            target_text,
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

        if len(prompt_ids) < 1:
            raise ValueError(f"Empty tokenized prompt at idx={idx}")
        if len(target_ids) < 1:
            raise ValueError(f"Empty tokenized target at idx={idx}: target_text={target_text}")

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        lm_labels = [-100] * len(prompt_ids) + target_ids

        candidate_texts = list(row["candidate_agents"])
        if self.none_agent_text not in candidate_texts:
            candidate_texts = [self.none_agent_text] + candidate_texts
        else:
            candidate_texts = [self.none_agent_text] + [x for x in candidate_texts if x != self.none_agent_text]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lm_labels": lm_labels,
            "prompt_length": len(prompt_ids),
            "label_id": label_id,
            "raw_label": row["label"],
            "gold_agent_text": target_text,
            "candidate_texts": candidate_texts,
        }


class JointClsAgentLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        lm_labels = []
        prompt_length = []
        label_id = []
        raw_label = []
        gold_agent_text = []
        candidate_texts = []

        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            labs = item["lm_labels"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            lm_labels.append(labs + [-100] * pad_len)

            prompt_length.append(item["prompt_length"])
            label_id.append(item["label_id"])
            raw_label.append(item["raw_label"])
            gold_agent_text.append(item["gold_agent_text"])
            candidate_texts.append(item["candidate_texts"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "lm_labels": torch.tensor(lm_labels, dtype=torch.long),
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
            "label_id": torch.tensor(label_id, dtype=torch.long),
            "raw_label": raw_label,
            "gold_agent_text": gold_agent_text,
            "candidate_texts": candidate_texts,
        }


# =========================================================
# Model
# =========================================================

class JointABCClsAgentLMModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.cls_head = nn.Linear(hidden_size, 3)

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

        raise ValueError("Cannot locate backbone model from base_model")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: torch.Tensor,
        lm_labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_token_loss: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        backbone = self._get_backbone()

        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_hidden_states=False,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state  # [B, S, H]

        last_prompt_pos = prompt_length - 1
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_idx, last_prompt_pos, :]  # [B, H]
        cls_logits = self.cls_head(pooled)  # [B, 3]

        result: Dict[str, Optional[torch.Tensor]] = {
            "cls_logits": cls_logits,
            "active_logits": None,
            "active_labels": None,
            "token_loss": None,
        }

        if lm_labels is not None:
            shift_hidden = last_hidden[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            active_mask = shift_labels.ne(-100)

            if active_mask.any():
                active_hidden = shift_hidden[active_mask]     # [N_active, H]
                active_labels = shift_labels[active_mask]     # [N_active]
                active_logits = self.lm_head(active_hidden)   # [N_active, V]

                result["active_logits"] = active_logits
                result["active_labels"] = active_labels

                if return_token_loss:
                    result["token_loss"] = F.cross_entropy(
                        active_logits,
                        active_labels,
                        reduction="none",
                    )

        return result


# =========================================================
# Trainer
# =========================================================

class FSDPJointABCClsAgentLMTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.device_name = get_device_name()

        self.none_agent_text = str(
            getattr(self.config.data, "none_agent_text", getattr(self.config.data, "none_agent_name", "__NONE__"))
        )
        self.lambda_lm = float(getattr(self.config.model, "lambda_lm", 1.0))
        # Agent accuracy is computed by candidate-text loss scoring.
        # This is relatively expensive, so it is enabled for validation by default only when requested.
        self.eval_agent_on_val = bool(getattr(self.config.trainer, "eval_agent_on_val", False))

        self._normalize_config_bsz()
        self._build_dataloader(train_dataset, val_dataset)
        self._build_model_optimizer()

        self.cls_loss_fct = nn.CrossEntropyLoss()

        if self.device_mesh.get_rank() == 0:
            print(self.config)
            print(f"[INFO] none_agent_text={self.none_agent_text}, lambda_lm={self.lambda_lm}")

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )
        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        collator = JointClsAgentLMCollator(self.tokenizer)

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        num_workers = int(getattr(self.config.data, "num_workers", 4))
        pin_memory = bool(getattr(self.config.data, "pin_memory", True))

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
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
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
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "bf16")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        print(f"\n==== Using torch_dtype: {torch_dtype} ====")

        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config

        if hasattr(self.model_config, "max_position_embeddings"):
            max_length = int(getattr(self.config.data, "max_length", 2048))
            self.model_config.max_position_embeddings = max(self.model_config.max_position_embeddings, max_length)

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
            self.model = JointABCClsAgentLMModel(
                base_model=base_model,
                hidden_size=hidden_size,
            )

        if self.config.model.enable_gradient_checkpointing:
            self.model.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
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
                f"Number of steps/epoch {self.steps_per_epoch}, "
                f"number of epochs {self.config.trainer.total_epochs}, "
                f"total number of steps {self.total_steps}"
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

    def _compute_lm_loss_from_outputs(self, model_outputs: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        active_logits = model_outputs["active_logits"]
        active_labels = model_outputs["active_labels"]

        if active_logits is None or active_labels is None:
            cls_logits = model_outputs["cls_logits"]
            return cls_logits.sum() * 0.0

        return F.cross_entropy(active_logits, active_labels)

    def _compute_cls_count_stats(self, label_id: torch.Tensor, cls_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return count-based ABC classification stats.

        Count-based aggregation is more accurate than averaging per-batch accuracies,
        especially when A/B/C are imbalanced or absent in some batches.
        """
        device = label_id.device
        correct = cls_pred.eq(label_id)

        stats = {
            "cls_total": torch.tensor(float(label_id.numel()), device=device),
            "cls_correct": correct.float().sum(),
        }
        for label_name, label_idx in LABEL_TO_ID.items():
            mask = label_id.eq(label_idx)
            stats[f"cls_total_{label_name}"] = mask.float().sum()
            stats[f"cls_correct_{label_name}"] = (correct & mask).float().sum()
        return stats

    def _forward_and_compute_core_metrics(self, batch):
        input_ids = batch["input_ids"].to(self.device_name, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device_name, non_blocking=True)
        lm_labels = batch["lm_labels"].to(self.device_name, non_blocking=True)
        prompt_length = batch["prompt_length"].to(self.device_name, non_blocking=True)
        label_id = batch["label_id"].to(self.device_name, non_blocking=True)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            outputs = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_length=prompt_length,
                lm_labels=lm_labels,
                use_cache=False,
                return_token_loss=False,
            )

            cls_logits = outputs["cls_logits"]
            cls_loss = self.cls_loss_fct(cls_logits, label_id)
            lm_loss = self._compute_lm_loss_from_outputs(outputs)
            loss = cls_loss + self.lambda_lm * lm_loss

            cls_pred = cls_logits.argmax(dim=-1)

        cls_stats = self._compute_cls_count_stats(label_id=label_id, cls_pred=cls_pred)

        result = {
            "loss": loss,
            "cls_loss": cls_loss.detach(),
            "lm_loss": lm_loss.detach(),
            "cls_pred": cls_pred.detach(),
            "label_id": label_id.detach(),
        }
        result.update({k: v.detach() for k, v in cls_stats.items()})
        return result

    def _safe_div(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        if float(denominator.detach().item()) <= 0:
            return torch.zeros([], device=numerator.device, dtype=torch.float32)
        return numerator.float() / denominator.float()

    def _metrics_from_count_stats(self, prefix: str, stats: Dict[str, torch.Tensor], lr: Optional[float] = None) -> Dict[str, float]:
        metrics = {
            f"{prefix}/loss": float(stats["loss"].item()),
            f"{prefix}/cls_loss": float(stats["cls_loss"].item()),
            f"{prefix}/lm_loss": float(stats["lm_loss"].item()),
            f"{prefix}/cls_acc": float(self._safe_div(stats["cls_correct"], stats["cls_total"]).item()),
            f"{prefix}/acc_A": float(self._safe_div(stats["cls_correct_A"], stats["cls_total_A"]).item()),
            f"{prefix}/acc_B": float(self._safe_div(stats["cls_correct_B"], stats["cls_total_B"]).item()),
            f"{prefix}/acc_C": float(self._safe_div(stats["cls_correct_C"], stats["cls_total_C"]).item()),
            f"{prefix}/label_total_A": float(stats["cls_total_A"].item()),
            f"{prefix}/label_total_B": float(stats["cls_total_B"].item()),
            f"{prefix}/label_total_C": float(stats["cls_total_C"].item()),
        }

        # Optional agent-generation metrics. agent_acc_A is usually the most meaningful one,
        # because B/C gold agent is the artificial none_agent_text.
        if "agent_total_all" in stats:
            metrics.update({
                f"{prefix}/agent_acc_all": float(self._safe_div(stats["agent_correct_all"], stats["agent_total_all"]).item()),
                f"{prefix}/agent_acc_A": float(self._safe_div(stats["agent_correct_A"], stats["agent_total_A"]).item()),
                f"{prefix}/joint_acc_all": float(self._safe_div(stats["joint_correct_all"], stats["joint_total_all"]).item()),
                f"{prefix}/joint_acc_A": float(self._safe_div(stats["joint_correct_A"], stats["joint_total_A"]).item()),
                f"{prefix}/agent_total_all": float(stats["agent_total_all"].item()),
                f"{prefix}/agent_total_A": float(stats["agent_total_A"].item()),
            })

        if lr is not None:
            metrics[f"{prefix}/lr(1e-3)"] = lr * 1e3
        return metrics

    def _empty_count_stats(self) -> Dict[str, torch.Tensor]:
        device = torch.device(self.device_name)
        keys = [
            "loss", "cls_loss", "lm_loss",
            "cls_total", "cls_correct",
            "cls_total_A", "cls_correct_A",
            "cls_total_B", "cls_correct_B",
            "cls_total_C", "cls_correct_C",
        ]
        return {k: torch.zeros([], device=device, dtype=torch.float32) for k in keys}

    def _accumulate_count_stats(self, acc: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], loss_weight: float = 1.0):
        # Losses are averaged over micro/validation batches; counts are summed.
        for key in ["loss", "cls_loss", "lm_loss"]:
            acc[key] += out[key].detach().float() * loss_weight
        for key in [
            "cls_total", "cls_correct",
            "cls_total_A", "cls_correct_A",
            "cls_total_B", "cls_correct_B",
            "cls_total_C", "cls_correct_C",
        ]:
            acc[key] += out[key].detach().float()

    @torch.no_grad()
    def _compute_agent_count_stats(self, batch, cls_pred: torch.Tensor, label_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute agent-generation accuracy by candidate text loss scoring.

        The model does not have an agent classification head. It is trained to generate
        only the agent text, so the most faithful accuracy is obtained by scoring the
        current sample's candidate texts and choosing the lowest-NLL text.
        """
        pred_agent_texts = self._predict_agent_texts_from_candidates(batch)
        gold_agent_texts = batch["gold_agent_text"]

        device = torch.device(self.device_name)
        stats = {
            "agent_total_all": torch.tensor(float(len(pred_agent_texts)), device=device),
            "agent_correct_all": torch.zeros([], device=device, dtype=torch.float32),
            "agent_total_A": torch.zeros([], device=device, dtype=torch.float32),
            "agent_correct_A": torch.zeros([], device=device, dtype=torch.float32),
            "joint_total_all": torch.tensor(float(len(pred_agent_texts)), device=device),
            "joint_correct_all": torch.zeros([], device=device, dtype=torch.float32),
            "joint_total_A": torch.zeros([], device=device, dtype=torch.float32),
            "joint_correct_A": torch.zeros([], device=device, dtype=torch.float32),
        }

        for i, (pred_agent, gold_agent) in enumerate(zip(pred_agent_texts, gold_agent_texts)):
            gold_label_id = int(label_id[i].item())
            pred_label_id = int(cls_pred[i].item())
            agent_ok = (pred_agent == gold_agent)
            label_ok = (pred_label_id == gold_label_id)
            joint_ok = label_ok and agent_ok

            if agent_ok:
                stats["agent_correct_all"] += 1.0
            if joint_ok:
                stats["joint_correct_all"] += 1.0

            if gold_label_id == LABEL_TO_ID["A"]:
                stats["agent_total_A"] += 1.0
                stats["joint_total_A"] += 1.0
                if agent_ok:
                    stats["agent_correct_A"] += 1.0
                if joint_ok:
                    stats["joint_correct_A"] += 1.0

        return stats

    @torch.no_grad()
    def _score_candidate_text(self, prompt_ids: List[int], candidate_text: str) -> float:
        """
        Score candidate by average token NLL.
        Lower is better.
        Only computes logits on valid target positions.
        """
        cand_ids = self.tokenizer(
            candidate_text,
            add_special_tokens=False,
            truncation=True,
            max_length=int(getattr(self.config.data, "max_target_length", 16)),
        )["input_ids"]

        if bool(getattr(self.config.data, "append_eos_to_target", True)):
            cand_ids = cand_ids + [self.tokenizer.eos_token_id]

        full_ids = prompt_ids + cand_ids
        full_mask = [1] * len(full_ids)
        labels = [-100] * len(prompt_ids) + cand_ids

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device_name)
        attention_mask = torch.tensor([full_mask], dtype=torch.long, device=self.device_name)
        prompt_length = torch.tensor([len(prompt_ids)], dtype=torch.long, device=self.device_name)
        labels = torch.tensor([labels], dtype=torch.long, device=self.device_name)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            outputs = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_length=prompt_length,
                lm_labels=labels,
                use_cache=False,
                return_token_loss=True,
            )

        token_loss = outputs["token_loss"]
        if token_loss is None or token_loss.numel() == 0:
            return 0.0
        return token_loss.mean().item()

    @torch.no_grad()
    def _predict_agent_texts_from_candidates(self, batch) -> List[str]:
        self.fsdp_model.eval()

        input_ids = batch["input_ids"]
        prompt_lengths = batch["prompt_length"].tolist()
        candidate_texts = batch["candidate_texts"]

        pred_agent_texts = []
        for i in range(len(prompt_lengths)):
            p_len = prompt_lengths[i]
            prompt_ids = input_ids[i][:p_len].tolist()

            candidates = candidate_texts[i]
            best_text = None
            best_score = None

            for cand in candidates:
                score = self._score_candidate_text(prompt_ids, cand)
                if best_score is None or score < best_score:
                    best_score = score
                    best_text = cand

            pred_agent_texts.append(best_text)

        return pred_agent_texts

    def training_step(self, batch):
        self.fsdp_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        micro_batches = []
        bsz = batch["input_ids"].size(0)
        micro = self.config.data.micro_batch_size_per_gpu
        for start in range(0, bsz, micro):
            end = start + micro
            micro_batches.append({
                "input_ids": batch["input_ids"][start:end],
                "attention_mask": batch["attention_mask"][start:end],
                "lm_labels": batch["lm_labels"][start:end],
                "prompt_length": batch["prompt_length"][start:end],
                "label_id": batch["label_id"][start:end],
            })

        n_micro_batches = len(micro_batches)
        stats = self._empty_count_stats()

        for micro_batch in micro_batches:
            out = self._forward_and_compute_core_metrics(micro_batch)
            (out["loss"] / n_micro_batches).backward()
            self._accumulate_count_stats(stats, out, loss_weight=1.0 / n_micro_batches)

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

        # Losses are already averaged within rank; counts should be summed across ranks.
        for key in stats:
            if key in {"loss", "cls_loss", "lm_loss"}:
                torch.distributed.all_reduce(stats[key], op=torch.distributed.ReduceOp.AVG)
            else:
                torch.distributed.all_reduce(stats[key], op=torch.distributed.ReduceOp.SUM)

        return self._metrics_from_count_stats("train", stats, lr=lr)

    @torch.no_grad()
    def validation_step(self, batch):
        self.fsdp_model.eval()
        out = self._forward_and_compute_core_metrics(batch)

        stats = self._empty_count_stats()
        self._accumulate_count_stats(stats, out, loss_weight=1.0)

        if self.eval_agent_on_val:
            agent_stats = self._compute_agent_count_stats(
                batch=batch,
                cls_pred=out["cls_pred"],
                label_id=out["label_id"],
            )
            stats.update(agent_stats)

        return stats

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
                "none_agent_text": self.none_agent_text,
                "lambda_lm": self.lambda_lm,
                "label_to_id": LABEL_TO_ID,
                "id_to_label": ID_TO_LABEL,
                "checkpoint_type": "joint_abc_cls_plus_agent_lm_memsave",
                "memory_optimized": True,
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
            for data in self.train_dataloader:
                global_step += 1
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = self.config.trainer.test_freq > 0 and global_step % self.config.trainer.test_freq == 0
                is_save_step = self.config.trainer.save_freq > 0 and global_step % self.config.trainer.save_freq == 0

                if is_last_step or is_valid_step:
                    val_stats = None
                    val_batches = 0
                    for val_data in self.val_dataloader:
                        out = self.validation_step(val_data)
                        if val_stats is None:
                            val_stats = {k: torch.zeros_like(v) for k, v in out.items()}
                        for key, val in out.items():
                            val_stats[key] += val.detach().float()
                        val_batches += 1

                    if val_stats is None:
                        val_stats = self._empty_count_stats()

                    for key in val_stats:
                        torch.distributed.all_reduce(val_stats[key], op=torch.distributed.ReduceOp.SUM)

                    # Losses were summed over local validation batches and then over ranks.
                    # Counts remain summed; losses should be averaged by number of val batches * world size.
                    denom = max(1, val_batches * self.device_mesh.size())
                    for key in ["loss", "cls_loss", "lm_loss"]:
                        val_stats[key] /= denom

                    if rank == 0:
                        metric = self._metrics_from_count_stats("val", val_stats)
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric

                    torch.distributed.barrier()

                if is_last_step or is_save_step:
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


# =========================================================
# Run
# =========================================================

def create_dataset_from_rows(raw_rows, data_config, tokenizer):
    return JointClsAgentLMDataset(
        raw_rows=raw_rows,
        tokenizer=tokenizer,
        config=data_config,
    )


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    ulysses_sp_size = int(getattr(config, "ulysses_sequence_parallel_size", 1))
    if ulysses_sp_size != 1:
        raise ValueError("This simplified joint trainer does not support ulysses sequence parallel. Set ulysses_sequence_parallel_size=1.")
    if getattr(config, "use_remove_padding", False):
        raise ValueError("This simplified joint trainer does not use remove_padding. Set use_remove_padding=False.")

    device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",),
    )

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
            print("gold_agent_text:", item["gold_agent_text"])
            print("input len:", len(item["input_ids"]))
            print("decoded prompt tail:", tokenizer.decode(item["input_ids"][:item["prompt_length"]][-120:]))
            print("candidate_texts:", item["candidate_texts"][:10])
        print("===== END DEBUG =====")

    trainer = FSDPJointABCClsAgentLMTrainer(
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
