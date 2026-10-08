# -*- coding: utf-8 -*-
# @Author  : qiaohezhe / ChatGPT
# @Date    : 2026-04-25
# @File    : fsdp_generation_json_sft_trainer.py
# @Desc    : Pure generation SFT trainer for JSON outputs:
#            prompt -> {"label":"A","agent":"Planner"}
#
# Key design:
#   - NO cls_head
#   - NO agent_head
#   - NO classification loss
#   - Only LM loss on response JSON tokens
#   - Prompt tokens are masked with labels=-100
#
# Expected data format:
# {
#   "prompt": "... prompt ending right before the answer ...",
#   "response": "{\"label\":\"A\",\"agent\":\"Solver\"}"
# }
#
# Also supports:
#   "response": {"label":"A","agent":"Solver"}
#   "response": "{\"label\":\"B\",\"agent\":null}"
#   "response": "A"  # converted to {"label":"A","agent":null}; not recommended
#
# Recommended inference after this trainer:
#   1) Free generation, or preferably
#   2) Candidate JSON loss scoring:
#      candidates = [
#        {"label":"A","agent":"<candidate_agent_1>"},
#        ...,
#        {"label":"B","agent":null},
#        {"label":"C","agent":null},
#      ]
#      choose the candidate with lowest conditional LM loss.

import os
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import hydra
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import optim
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

LABELS = {"A", "B", "C"}


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


def _maybe_parse_json_text(text: str) -> Optional[Any]:
    text = str(text).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def canonicalize_generation_response(response_value: Any) -> str:
    """
    Return a compact canonical JSON string:
      {"label":"A","agent":"Planner"}
      {"label":"B","agent":null}
      {"label":"C","agent":null}

    This is important because pure generation SFT is sensitive to output format.
    """
    if isinstance(response_value, dict):
        obj = response_value
    else:
        text = str(response_value).strip()
        # Legacy support: "A" / "B" / "C"
        if text.upper() in LABELS:
            obj = {"label": text.upper(), "agent": None}
        else:
            parsed = _maybe_parse_json_text(text)
            if not isinstance(parsed, dict):
                raise ValueError(f"Response must be a JSON object or A/B/C string, got: {response_value}")
            obj = parsed

    label = str(obj.get("label", "")).strip().upper()
    if label not in LABELS:
        raise ValueError(f"Invalid label in response: {response_value}")

    agent = obj.get("agent", None)

    # For B/C, force agent=null to avoid inconsistent targets.
    if label in {"B", "C"}:
        agent = None
    else:
        # For A, agent should normally be a candidate agent name.
        # We keep None only for robustness, but this is not recommended.
        if agent is not None:
            agent = str(agent).strip()
            if agent == "" or agent.lower() in {"none", "null", "__none__"}:
                agent = None

    out = {"label": label, "agent": agent}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def raw_or_canonical_response(response_value: Any, canonicalize: bool = True) -> str:
    if canonicalize:
        return canonicalize_generation_response(response_value)
    if isinstance(response_value, dict):
        return json.dumps(response_value, ensure_ascii=False, separators=(",", ":"))
    return str(response_value).strip()


# =========================================================
# Dataset / collator
# =========================================================

class GenerationJsonSFTDataset(Dataset):
    def __init__(self, raw_rows, tokenizer, config):
        self.tokenizer = tokenizer
        self.prompt_key = getattr(config, "prompt_key", "prompt")
        self.response_key = getattr(config, "response_key", "response")
        self.max_length = int(getattr(config, "max_length", 4096))
        self.max_prompt_length = int(getattr(config, "max_prompt_length", self.max_length))
        self.add_eos = bool(getattr(config, "add_eos", True))
        self.canonicalize_response = bool(getattr(config, "canonicalize_response", True))

        # When prompt+response exceeds max_length, truncate prompt.
        # "left" means drop beginning of prompt and keep tail, which usually contains hypothesis/candidates/assistant marker.
        # "right" means keep beginning of prompt and drop tail.
        self.prompt_truncation_side = str(getattr(config, "prompt_truncation_side", "left")).lower()
        if self.prompt_truncation_side not in {"left", "right"}:
            raise ValueError("data.prompt_truncation_side must be 'left' or 'right'.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.rows = self._validate_rows(raw_rows)
        if len(self.rows) == 0:
            raise ValueError("No valid rows loaded.")

        label_hist = {"A": 0, "B": 0, "C": 0}
        for row in self.rows:
            try:
                obj = json.loads(row["response"])
                label_hist[obj["label"]] += 1
            except Exception:
                pass
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
                response_text = raw_or_canonical_response(
                    response,
                    canonicalize=self.canonicalize_response,
                )
            except Exception:
                add_drop("bad_response")
                continue

            if not str(prompt).strip() or not response_text.strip():
                add_drop("empty_prompt_or_response")
                continue

            valid_rows.append({
                "prompt": str(prompt),
                "response": response_text,
            })

        print(f"[DATASET] dropped={dropped}, reasons={drop_reasons}")
        return valid_rows

    def __len__(self):
        return len(self.rows)

    def _truncate_prompt_to_fit(self, prompt_ids: List[int], response_ids: List[int]) -> List[int]:
        # Always keep all response tokens; truncate prompt if needed.
        max_prompt_allowed = self.max_length - len(response_ids)
        if max_prompt_allowed <= 0:
            # Response alone is too long. Keep the tail of response elsewhere in __getitem__.
            return []

        if len(prompt_ids) <= max_prompt_allowed:
            return prompt_ids

        if self.prompt_truncation_side == "left":
            return prompt_ids[-max_prompt_allowed:]
        return prompt_ids[:max_prompt_allowed]

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = row["prompt"]
        response = row["response"]

        prompt_enc = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_attention_mask=False,
        )
        response_enc = self.tokenizer(
            response,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )

        prompt_ids = list(prompt_enc["input_ids"])
        response_ids = list(response_enc["input_ids"])

        if self.add_eos and self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]

        if len(response_ids) == 0:
            raise ValueError(f"Empty response tokens at idx={idx}")

        # If response itself is too long, keep its tail to avoid empty labels.
        if len(response_ids) >= self.max_length:
            response_ids = response_ids[-(self.max_length - 1):]

        prompt_ids = self._truncate_prompt_to_fit(prompt_ids, response_ids)

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_length:
            # Safety guard. This should not happen because prompt is truncated above.
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]
            attention_mask = attention_mask[-self.max_length:]

            # Ensure we still ignore prompt-like tokens if truncation damaged alignment.
            # In normal path this is unnecessary.
            has_label = any(x != -100 for x in labels)
            if not has_label:
                labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "response_text": response,
        }


class GenerationJsonSFTCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            lab = item["labels"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =========================================================
# Trainer
# =========================================================

class FSDPGenerationJsonSFTTrainer:
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

        self.compute_token_acc = bool(getattr(self.config.trainer, "compute_token_acc", True))

        if self.device_mesh.get_rank() == 0:
            print(self.config)
            print("[INFO] pure generation SFT trainer: LM loss only, no cls_head, no agent_head")

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
        collator = GenerationJsonSFTCollator(self.tokenizer)

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
            max_length = int(getattr(self.config.data, "max_length", 4096))
            self.model_config.max_position_embeddings = max(self.model_config.max_position_embeddings, max_length)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings,
            mesh=self.device_mesh,
        )

        with init_context():
            model_kwargs = dict(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            attn_impl = self.config.model.get("attn_implementation", "flash_attention_2")
            if attn_impl is not None and str(attn_impl).lower() not in {"none", "null", ""}:
                model_kwargs["attn_implementation"] = attn_impl

            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                **model_kwargs,
            )

            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

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
        labels = batch["labels"].to(self.device_name)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            outputs = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            loss = outputs.loss

        # Token accuracy on response tokens only.
        # This is optional because argmax over full vocab costs extra memory/time.
        token_acc = torch.zeros([], device=input_ids.device)
        exact_match = torch.zeros([], device=input_ids.device)

        if self.compute_token_acc:
            with torch.no_grad():
                logits = outputs.logits
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                valid = shift_labels != -100

                if valid.any():
                    pred = torch.argmax(shift_logits, dim=-1)
                    correct = (pred == shift_labels) & valid
                    token_acc = correct.sum().float() / valid.sum().float()

                    # Sample-level exact match on supervised response tokens.
                    valid_count = valid.sum(dim=1)
                    per_token_ok = (pred == shift_labels) | (~valid)
                    sample_ok = per_token_ok.all(dim=1) & (valid_count > 0)
                    exact_match = sample_ok.float().mean()

        return {
            "loss": loss,
            "lm_loss": loss.detach(),
            "token_acc": token_acc.detach(),
            "exact_match": exact_match.detach(),
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
                "labels": batch["labels"][start:end],
            })

        n_micro_batches = len(micro_batches)
        step_loss = step_lm_loss = step_token_acc = step_exact_match = 0.0

        for micro_batch in micro_batches:
            out = self._compute_loss_and_metrics(micro_batch)
            (out["loss"] / n_micro_batches).backward()

            step_loss += out["loss"].detach().item() / n_micro_batches
            step_lm_loss += out["lm_loss"].detach().item() / n_micro_batches
            step_token_acc += out["token_acc"].detach().item() / n_micro_batches
            step_exact_match += out["exact_match"].detach().item() / n_micro_batches

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
            "train/lm_loss": torch.tensor(step_lm_loss, device=self.device_name),
            "train/token_acc": torch.tensor(step_token_acc, device=self.device_name),
            "train/exact_match": torch.tensor(step_exact_match, device=self.device_name),
        }

        for k in metrics:
            torch.distributed.all_reduce(metrics[k], op=torch.distributed.ReduceOp.AVG)

        return {
            "train/loss": metrics["train/loss"].item(),
            "train/lm_loss": metrics["train/lm_loss"].item(),
            "train/token_acc": metrics["train/token_acc"].item(),
            "train/exact_match": metrics["train/exact_match"].item(),
            "train/lr(1e-3)": lr * 1e3,
        }

    @torch.no_grad()
    def validation_step(self, batch):
        self.fsdp_model.eval()
        out = self._compute_loss_and_metrics(batch)

        reduced = {}
        for key in ["loss", "lm_loss", "token_acc", "exact_match"]:
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
                "checkpoint_type": "pure_generation_json_sft",
                "loss": "causal_lm_loss_on_response_tokens_only",
                "target_format": "{\"label\":\"A\",\"agent\":\"Planner\"}",
                "non_entail_format": "{\"label\":\"B\",\"agent\":null} or {\"label\":\"C\",\"agent\":null}",
                "add_eos": bool(getattr(self.config.data, "add_eos", True)),
                "canonicalize_response": bool(getattr(self.config.data, "canonicalize_response", True)),
                "prompt_truncation_side": str(getattr(self.config.data, "prompt_truncation_side", "left")),
            }
            with open(os.path.join(path, "generation_sft_meta.json"), "w", encoding="utf-8") as f:
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
                        keys = ["loss", "lm_loss", "token_acc", "exact_match"]
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
    return GenerationJsonSFTDataset(raw_rows=raw_rows, tokenizer=tokenizer, config=data_config)


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

    # Training LM loss with right padding is fine because labels for padding are -100.
    tokenizer.padding_side = "right"

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
            label_positions = sum(1 for x in item["labels"] if x != -100)
            print(f"--- sample {i} ---")
            print("input len:", len(item["input_ids"]))
            print("supervised response tokens:", label_positions)
            print("response_text:", item["response_text"])
            print("decoded input tail:", tokenizer.decode(item["input_ids"][-160:]))
            supervised_ids = [tid for tid, lab in zip(item["input_ids"], item["labels"]) if lab != -100]
            print("decoded supervised response:", tokenizer.decode(supervised_ids))
        print("===== END DEBUG =====")

    trainer = FSDPGenerationJsonSFTTrainer(
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
