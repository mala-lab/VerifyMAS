# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A simplified FSDP trainer for 3-way next-token classification on labels A/B/C.

This version is designed for datasets where the ground truth is exactly one letter:
    A / B / C

Key idea:
- Feed ONLY the prompt to the model.
- Use the last valid prompt position to predict the next token.
- Compute CE loss over only the three label tokens: A, B, C.

This avoids SFTDataset/loss_mask completely and is much more stable for single-letter labels.

Expected data format (parquet or jsonl):
    {
      "prompt": "...prompt text ending right before the answer...",
      "response": "A"
    }

Typical prompt ending:
    "Output only one character: A or B or C.\nAssistant:"
or when using a chat template, the prompt column can already be fully formatted.
"""

import os
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

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


# =========================================================
# Dataset
# =========================================================

class ABCPromptOnlyDataset(Dataset):
    def __init__(self, data_files, tokenizer, config):
        self.tokenizer = tokenizer
        self.prompt_key = getattr(config, "prompt_key", "prompt")
        self.response_key = getattr(config, "response_key", "response")
        self.max_prompt_length = int(getattr(config, "max_prompt_length", getattr(config, "max_length", 4096)))
        self.label_texts = list(getattr(config, "label_texts", ["A", "B", "C"]))
        if self.label_texts != ["A", "B", "C"]:
            raise ValueError(f"This trainer currently expects label_texts=['A','B','C'], got {self.label_texts}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.label_to_idx = {"A": 0, "B": 1, "C": 2}
        self.rows = self._load_rows(data_files)
        if len(self.rows) == 0:
            raise ValueError(f"No valid rows loaded from {data_files}")

        # quick validation on a few rows
        bad_preview = []
        for i, row in enumerate(self.rows[:20]):
            resp = str(row[self.response_key]).strip().upper()
            if resp not in self.label_to_idx:
                bad_preview.append((i, resp))
        if bad_preview:
            raise ValueError(
                f"Found invalid labels in the first rows: {bad_preview[:5]}. "
                "Each response must be exactly one of A/B/C."
            )

    def _normalize_files(self, data_files) -> List[str]:
        if isinstance(data_files, str):
            return [data_files]
        if isinstance(data_files, (list, tuple)):
            return list(data_files)
        raise ValueError(f"Unsupported data_files type: {type(data_files)}")

    def _load_rows(self, data_files) -> List[Dict[str, Any]]:
        files = self._normalize_files(data_files)
        rows: List[Dict[str, Any]] = []

        for file_path in files:
            path = str(file_path)
            suffix = Path(path).suffix.lower()

            if suffix == ".parquet":
                df = pd.read_parquet(path)
                for rec in df.to_dict(orient="records"):
                    rows.append(rec)
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

        valid_rows = []
        dropped = 0
        for row in rows:
            if self.prompt_key not in row or self.response_key not in row:
                dropped += 1
                continue

            prompt = row[self.prompt_key]
            response = row[self.response_key]
            if prompt is None or response is None:
                dropped += 1
                continue

            prompt = str(prompt)
            response = str(response).strip().upper()
            if response not in {"A", "B", "C"}:
                dropped += 1
                continue

            valid_rows.append({
                self.prompt_key: prompt,
                self.response_key: response,
            })

        print(f"[DATA] loaded {len(valid_rows)} valid rows, dropped {dropped} invalid rows")
        return valid_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = str(row[self.prompt_key])
        response = str(row[self.response_key]).strip().upper()
        label_id = self.label_to_idx[response]

        enc = self.tokenizer(
            prompt,
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
            "label_id": label_id,
            "raw_label": response,
        }


class ABCPromptOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        label_id = []

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
# Trainer
# =========================================================

class FSDPABCTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.device_name = get_device_name()

        self.label_texts = list(getattr(self.config.data, "label_texts", ["A", "B", "C"]))
        if len(self.label_texts) != 3:
            raise ValueError(f"Expected exactly 3 labels, got {self.label_texts}")

        self._normalize_config_bsz()
        self._build_dataloader(train_dataset, val_dataset)
        self._build_model_optimizer()

        self.cls_loss_fct = nn.CrossEntropyLoss()
        self.class_token_ids = self._build_class_token_ids()

        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _build_class_token_ids(self) -> torch.Tensor:
        ids = []
        for label in self.label_texts:
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Label {label!r} is tokenized into {token_ids}, not a single token."
                )
            ids.append(token_ids[0])

        if self.device_mesh.get_rank() == 0:
            print("[INFO] label token ids:")
            for label, tok in zip(self.label_texts, ids):
                print(f"  {label} -> {tok}, decoded={self.tokenizer.decode([tok])!r}")

        return torch.tensor(ids, dtype=torch.long)

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
        collator = ABCPromptOnlyCollator(self.tokenizer)

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

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
            num_workers=4,
            pin_memory=True,
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
            num_workers=4,
            pin_memory=True,
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
            max_prompt_length = int(getattr(self.config.data, "max_prompt_length", getattr(self.config.data, "max_length", 4096)))
            self.model_config.max_position_embeddings = max(self.model_config.max_position_embeddings, max_prompt_length)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings,
            mesh=self.device_mesh,
        )

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
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
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

    def _compute_loss_and_accuracy(self, batch):
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        label_id = batch["label_id"].to(self.device_name)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            outputs = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits  # [B, S, V]

            last_pos = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(logits.size(0), device=logits.device)
            next_token_logits = logits[batch_idx, last_pos, :]  # [B, V]

            class_token_ids = self.class_token_ids.to(logits.device)
            cls_logits = next_token_logits.index_select(dim=-1, index=class_token_ids)  # [B, 3]

            loss = self.cls_loss_fct(cls_logits, label_id)
            preds = cls_logits.argmax(dim=-1)
            acc = (preds == label_id).float().mean()

        return loss, acc

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
        step_loss = 0.0
        step_acc = 0.0

        for micro_batch in micro_batches:
            loss, acc = self._compute_loss_and_accuracy(micro_batch)
            (loss / n_micro_batches).backward()
            step_loss += loss.detach().item() / n_micro_batches
            step_acc += acc.detach().item() / n_micro_batches

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

        step_loss = torch.tensor(step_loss, device=self.device_name)
        step_acc = torch.tensor(step_acc, device=self.device_name)
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(step_acc, op=torch.distributed.ReduceOp.AVG)

        return {
            "train/loss": step_loss.item(),
            "train/acc": step_acc.item(),
            "train/lr(1e-3)": lr * 1e3,
        }

    @torch.no_grad()
    def validation_step(self, batch):
        self.fsdp_model.eval()
        loss, acc = self._compute_loss_and_accuracy(batch)
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.AVG)
        return loss, acc

    def save_checkpoint(self, step):
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        fsdp_strategy = self.config.model.strategy

        if fsdp_strategy == "fsdp":
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
        elif fsdp_strategy == "fsdp2":
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

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
                    val_losses = []
                    val_accs = []
                    for val_data in self.val_dataloader:
                        val_loss, val_acc = self.validation_step(val_data)
                        val_losses.append(val_loss)
                        val_accs.append(val_acc)

                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        val_acc = torch.mean(torch.stack(val_accs))
                        metric = {
                            "val/loss": val_loss.item(),
                            "val/acc": val_acc.item(),
                        }
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

def create_dataset(data_paths, data_config, tokenizer):
    return ABCPromptOnlyDataset(data_files=data_paths, tokenizer=tokenizer, config=data_config)


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    ulysses_sp_size = int(getattr(config, "ulysses_sequence_parallel_size", 1))
    if ulysses_sp_size != 1:
        raise ValueError("This simplified ABC trainer does not support ulysses sequence parallel. Set ulysses_sequence_parallel_size=1.")
    if getattr(config, "use_remove_padding", False):
        raise ValueError("This simplified ABC trainer does not use remove_padding. Set use_remove_padding=False.")

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

    train_dataset = create_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_dataset(config.data.val_files, config.data, tokenizer)

    if rank == 0:
        print("===== DEBUG DATASET =====")
        print("train size:", len(train_dataset))
        for i in range(min(3, len(train_dataset))):
            item = train_dataset[i]
            print(f"--- sample {i} ---")
            print("label:", item["raw_label"])
            print("input len:", len(item["input_ids"]))
            print("decoded tail:", tokenizer.decode(item["input_ids"][-80:]))
        print("===== END DEBUG =====")

    trainer = FSDPABCTrainer(
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
