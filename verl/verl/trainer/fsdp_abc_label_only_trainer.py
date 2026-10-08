# -*- coding: utf-8 -*-
# @Author  : qiaohezhe / ChatGPT
# @Date    : 2026-04-24
# @File    : fsdp_abc_label_only_trainer.py
# @Desc    : FSDP trainer for A/B/C label-only classification.
#
# Expected data format:
# {
#   "prompt": "...prompt text ending right before the answer...",
#   "response": "{\"label\":\"A\",\"agent\":\"Solver\"}"
# }
#
# Also supports:
#   "response": {"label":"A","agent":"Solver"}
#   "response": "A"
#
# Training target:
#   - cls_head only: A / B / C
#   - NO agent_head
#   - NO agent_loss
#   - NO candidate_mask

import os
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

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


def parse_label_from_response(response_value: Any) -> str:
    """
    Accept:
      - {"label":"A","agent":"Solver"}
      - "{\"label\":\"A\",\"agent\":\"Solver\"}"
      - "A"
    Returns:
      label in {"A","B","C"}
    """
    if isinstance(response_value, dict):
        obj = response_value
        label = str(obj.get("label", "")).strip().upper()
    else:
        text = str(response_value).strip()
        if text.upper() in LABEL_TO_ID:
            label = text.upper()
        else:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"Response JSON must be object, got {type(obj)}")
            label = str(obj.get("label", "")).strip().upper()

    if label not in LABEL_TO_ID:
        raise ValueError(f"Invalid label: {response_value}")

    return label


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
# Dataset / collator
# =========================================================

class ABCPromptOnlyDataset(Dataset):
    def __init__(self, raw_rows, tokenizer, config):
        self.tokenizer = tokenizer
        self.prompt_key = getattr(config, "prompt_key", "prompt")
        self.response_key = getattr(config, "response_key", "response")
        self.max_prompt_length = int(getattr(config, "max_prompt_length", getattr(config, "max_length", 4096)))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.rows = self._validate_rows(raw_rows)
        if len(self.rows) == 0:
            raise ValueError("No valid rows loaded.")

        label_hist = {"A": 0, "B": 0, "C": 0}
        for row in self.rows:
            label_hist[row["label"]] += 1
        print(f"[DATASET] valid rows={len(self.rows)}, label_hist={label_hist}")

    def _validate_rows(self, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                label = parse_label_from_response(response)
            except Exception:
                add_drop("bad_response")
                continue

            valid_rows.append({
                "prompt": str(prompt),
                "label": label,
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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_id": LABEL_TO_ID[row["label"]],
            "raw_label": row["label"],
        }


class ABCPromptOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, label_id = [], [], []
        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            label_id.append(item["label_id"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label_id": torch.tensor(label_id, dtype=torch.long),
        }


# =========================================================
# Model
# =========================================================

class ABCLabelOnlyModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.cls_head = nn.Linear(hidden_size, 3)

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
        return cls_logits


# =========================================================
# Trainer
# =========================================================

class FSDPABCLabelOnlyTrainer:
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

        if self.device_mesh.get_rank() == 0:
            print(self.config)
            print("[INFO] label-only trainer: cls_head only, no agent_head, no agent_loss")

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
        collator = ABCPromptOnlyCollator(self.tokenizer)

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
            self.model = ABCLabelOnlyModel(
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

    def _compute_loss_and_metrics(self, batch):
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        label_id = batch["label_id"].to(self.device_name)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            cls_logits = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            loss = self.cls_loss_fct(cls_logits, label_id)

            cls_pred = cls_logits.argmax(dim=-1)
            cls_acc = (cls_pred == label_id).float().mean()

            # Per-class accuracy for debugging collapse.
            metrics = {}
            for label_name, label_idx in LABEL_TO_ID.items():
                mask = label_id == label_idx
                if mask.any():
                    metrics[f"acc_{label_name}"] = (cls_pred[mask] == label_id[mask]).float().mean()
                else:
                    metrics[f"acc_{label_name}"] = torch.zeros([], device=cls_logits.device)

        return {
            "loss": loss,
            "cls_loss": loss.detach(),
            "cls_acc": cls_acc.detach(),
            "acc_A": metrics["acc_A"].detach(),
            "acc_B": metrics["acc_B"].detach(),
            "acc_C": metrics["acc_C"].detach(),
        }

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
            })

        n_micro_batches = len(micro_batches)
        step_loss = step_cls_loss = step_cls_acc = 0.0
        step_acc_A = step_acc_B = step_acc_C = 0.0

        for micro_batch in micro_batches:
            out = self._compute_loss_and_metrics(micro_batch)
            (out["loss"] / n_micro_batches).backward()

            step_loss += out["loss"].detach().item() / n_micro_batches
            step_cls_loss += out["cls_loss"].detach().item() / n_micro_batches
            step_cls_acc += out["cls_acc"].detach().item() / n_micro_batches
            step_acc_A += out["acc_A"].detach().item() / n_micro_batches
            step_acc_B += out["acc_B"].detach().item() / n_micro_batches
            step_acc_C += out["acc_C"].detach().item() / n_micro_batches

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

        metrics = {
            "train/loss": torch.tensor(step_loss, device=self.device_name),
            "train/cls_loss": torch.tensor(step_cls_loss, device=self.device_name),
            "train/cls_acc": torch.tensor(step_cls_acc, device=self.device_name),
            "train/acc_A": torch.tensor(step_acc_A, device=self.device_name),
            "train/acc_B": torch.tensor(step_acc_B, device=self.device_name),
            "train/acc_C": torch.tensor(step_acc_C, device=self.device_name),
        }

        for k in metrics:
            torch.distributed.all_reduce(metrics[k], op=torch.distributed.ReduceOp.AVG)

        return {
            "train/loss": metrics["train/loss"].item(),
            "train/cls_loss": metrics["train/cls_loss"].item(),
            "train/cls_acc": metrics["train/cls_acc"].item(),
            "train/acc_A": metrics["train/acc_A"].item(),
            "train/acc_B": metrics["train/acc_B"].item(),
            "train/acc_C": metrics["train/acc_C"].item(),
            "train/lr(1e-3)": lr * 1e3,
        }

    @torch.no_grad()
    def validation_step(self, batch):
        self.fsdp_model.eval()
        out = self._compute_loss_and_metrics(batch)

        reduced = {}
        for key in ["loss", "cls_loss", "cls_acc", "acc_A", "acc_B", "acc_C"]:
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
                "checkpoint_type": "abc_label_only_classifier",
            }
            with open(os.path.join(path, "label_head_meta.json"), "w", encoding="utf-8") as f:
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
                        keys = ["loss", "cls_loss", "cls_acc", "acc_A", "acc_B", "acc_C"]
                        metric = {}
                        if val_metrics:
                            for key in keys:
                                metric[f"val/{key}"] = torch.mean(torch.stack([x[key] for x in val_metrics])).item()
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

def create_dataset_from_rows(raw_rows, data_config, tokenizer):
    return ABCPromptOnlyDataset(raw_rows=raw_rows, tokenizer=tokenizer, config=data_config)


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

    if rank == 0:
        print("[INFO] creating datasets...")
    train_dataset = create_dataset_from_rows(train_rows, config.data, tokenizer)
    val_dataset = create_dataset_from_rows(val_rows, config.data, tokenizer)

    if rank == 0:
        print("===== DEBUG DATASET =====")
        print("train size:", len(train_dataset))
        print("val size:", len(val_dataset))
        for i in range(min(2, len(train_dataset))):
            item = train_dataset[i]
            print(f"--- sample {i} ---")
            print("label_id:", item["label_id"])
            print("raw_label:", item["raw_label"])
            print("input len:", len(item["input_ids"]))
            print("decoded tail:", tokenizer.decode(item["input_ids"][-120:]))
        print("===== END DEBUG =====")

    trainer = FSDPABCLabelOnlyTrainer(
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
