# -*- coding: utf-8 -*-
# @Author  : qiaohezhe / ChatGPT
# @Date    : 2026-04-25
# @File    : fsdp_candidate_json_scoring_trainer_ultra_mem.py
# @Desc    : Ultra memory-efficient FSDP trainer for candidate JSON loss scoring.
#
# Key idea:
#   For each sample, enumerate legal JSON responses:
#       {"label":"A","agent":"<candidate_agent>"}
#       {"label":"B","agent":null}
#       {"label":"C","agent":null}
#
#   Instead of keeping all candidate forward graphs in GPU memory, this trainer uses
#   a two-pass score-gradient trick:
#     1) no_grad pass: score every candidate and compute d CE / d score.
#     2) grad pass: re-forward one candidate at a time and immediately backward
#        weighted_score = score_i * grad_i.
#
# This makes peak activation memory roughly one candidate sequence at a time.
# It is slower than the normal scoring trainer, but much more memory efficient.
#
# Expected row format:
# {
#   "prompt": "... prompt ending before answer ...",
#   "response": "{\"label\":\"A\",\"agent\":\"Solver\"}",
#   "candidate_agents": ["Planner", "Solver", "Critic"]
# }
#
# Also supports response as dict and candidate agents from:
#   candidate_agents / agents / agent_list / candidate_agent_list
# or fallback parsing from a prompt section beginning with "Candidate agents:".

import os
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
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


def _maybe_parse_json_text(text: str) -> Optional[Any]:
    text = str(text).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_response_obj(response_value: Any) -> Dict[str, Any]:
    """Parse response into {'label': A/B/C, 'agent': str or None}."""
    if isinstance(response_value, dict):
        obj = response_value
    else:
        text = str(response_value).strip()
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
    if label in {"B", "C"}:
        agent = None
    else:
        if agent is not None:
            agent = str(agent).strip()
            if agent == "" or agent.lower() in {"none", "null", "__none__", "n/a"}:
                agent = None

    return {"label": label, "agent": agent}


def make_json_response(label: str, agent: Optional[str]) -> str:
    label = str(label).strip().upper()
    if label not in LABELS:
        raise ValueError(f"Invalid label: {label}")
    if label in {"B", "C"}:
        agent = None
    elif agent is not None:
        agent = str(agent).strip()
    return json.dumps({"label": label, "agent": agent}, ensure_ascii=False, separators=(",", ":"))


def normalize_agent_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        parsed = _maybe_parse_json_text(value)
        if isinstance(parsed, list):
            value = parsed
        else:
            text = value.strip().replace("\r", "\n")
            chunks = []
            for part in re.split(r"[\n,;]+", text):
                part = part.strip()
                part = re.sub(r"^[-*\d\.\)\s]+", "", part).strip()
                if part:
                    chunks.append(part)
            value = chunks

    if not isinstance(value, (list, tuple)):
        return []

    agents = []
    seen = set()
    for x in value:
        if x is None:
            continue
        s = str(x).strip()
        if not s or s.lower() in {"none", "null", "__none__"}:
            continue
        if s not in seen:
            seen.add(s)
            agents.append(s)
    return agents


def parse_candidate_agents_from_prompt(prompt: str) -> List[str]:
    prompt = str(prompt)
    m = re.search(r"Candidate agents\s*:\s*(.*)", prompt, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []

    block = m.group(1)
    stop_patterns = [
        r"\n\s*Return\b",
        r"\n\s*Output\b",
        r"\n\s*Answer\b",
        r"\n\s*Label meanings\b",
        r"\n\s*Decision rules\b",
    ]
    stop = len(block)
    for pat in stop_patterns:
        sm = re.search(pat, block, flags=re.IGNORECASE)
        if sm:
            stop = min(stop, sm.start())
    block = block[:stop]
    return normalize_agent_list(block)


def extract_candidate_agents(row: Dict[str, Any], prompt_key: str, candidate_agents_key: str, gold_agent: Optional[str]) -> List[str]:
    agents = []
    for key in [candidate_agents_key, "candidate_agents", "agents", "agent_list", "candidate_agent_list"]:
        if key in row:
            agents = normalize_agent_list(row.get(key))
            if agents:
                break

    if not agents and isinstance(row.get("input"), dict):
        inp = row["input"]
        for key in [candidate_agents_key, "candidate_agents", "agents", "agent_list", "candidate_agent_list"]:
            if key in inp:
                agents = normalize_agent_list(inp.get(key))
                if agents:
                    break

    if not agents:
        agents = parse_candidate_agents_from_prompt(str(row.get(prompt_key, "")))

    # Safety fallback: never drop a positive row only because candidate list is missing.
    # This gives no agent negative for that row, but keeps the gold candidate valid.
    if gold_agent and gold_agent not in agents:
        agents = agents + [gold_agent]

    out = []
    seen = set()
    for a in agents:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def maybe_cap_agents(candidate_agents: List[str], gold_agent: Optional[str], max_a_candidates: int) -> List[str]:
    """Cap A-agent candidates while preserving gold_agent if present.

    max_a_candidates counts only A-agent candidates, not B/C null candidates.
    A deterministic subset is used for reproducibility. For stronger training,
    regenerate the dataset with shuffled candidate order or increase this value.
    """
    if max_a_candidates <= 0 or len(candidate_agents) <= max_a_candidates:
        return candidate_agents

    kept = []
    seen = set()
    if gold_agent and gold_agent in candidate_agents:
        kept.append(gold_agent)
        seen.add(gold_agent)

    for a in candidate_agents:
        if len(kept) >= max_a_candidates:
            break
        if a not in seen:
            kept.append(a)
            seen.add(a)
    return kept


def enumerate_candidate_jsons(candidate_agents: List[str]) -> List[str]:
    candidates = []
    for agent in candidate_agents:
        candidates.append(make_json_response("A", agent))
    candidates.append(make_json_response("B", None))
    candidates.append(make_json_response("C", None))
    return candidates


def find_gold_index(candidate_jsons: List[str], gold_obj: Dict[str, Any]) -> int:
    gold_json = make_json_response(gold_obj["label"], gold_obj.get("agent"))
    try:
        return candidate_jsons.index(gold_json)
    except ValueError:
        return -1


# =========================================================
# Dataset / collator
# =========================================================

class CandidateJSONScoringDataset(Dataset):
    def __init__(self, raw_rows, tokenizer, config):
        self.tokenizer = tokenizer
        self.prompt_key = getattr(config, "prompt_key", "prompt")
        self.response_key = getattr(config, "response_key", "response")
        self.candidate_agents_key = getattr(config, "candidate_agents_key", "candidate_agents")
        self.max_length = int(getattr(config, "max_length", 4096))
        self.max_prompt_length = int(getattr(config, "max_prompt_length", self.max_length))
        self.max_response_length = int(getattr(config, "max_response_length", 96))
        self.add_eos = bool(getattr(config, "add_eos", True))
        self.prompt_truncation_side = str(getattr(config, "prompt_truncation_side", "left")).lower()
        self.max_a_candidates_per_sample = int(getattr(config, "max_a_candidates_per_sample", 0))

        if self.prompt_truncation_side not in {"left", "right"}:
            raise ValueError("data.prompt_truncation_side must be 'left' or 'right'.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.rows = self._validate_and_preprocess(raw_rows)
        if len(self.rows) == 0:
            raise ValueError("No valid rows loaded.")

        label_hist = {"A": 0, "B": 0, "C": 0}
        cand_counts = []
        for row in self.rows:
            label_hist[row["gold_label"]] += 1
            cand_counts.append(len(row["candidate_jsons"]))
        print(f"[DATASET] valid rows={len(self.rows)}, label_hist={label_hist}")
        print(
            f"[DATASET] candidate_count min={min(cand_counts)}, max={max(cand_counts)}, "
            f"avg={sum(cand_counts) / max(1, len(cand_counts)):.2f}"
        )

    def _validate_and_preprocess(self, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_rows = []
        dropped = 0
        drop_reasons: Dict[str, int] = {}

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
                gold_obj = parse_response_obj(response)
            except Exception:
                add_drop("bad_response")
                continue

            if gold_obj["label"] == "A" and not gold_obj.get("agent"):
                add_drop("A_without_gold_agent")
                continue

            candidate_agents = extract_candidate_agents(
                row=row,
                prompt_key=self.prompt_key,
                candidate_agents_key=self.candidate_agents_key,
                gold_agent=gold_obj.get("agent"),
            )

            if gold_obj["label"] == "A" and gold_obj.get("agent") not in candidate_agents:
                candidate_agents.append(gold_obj["agent"])

            candidate_agents = maybe_cap_agents(
                candidate_agents=candidate_agents,
                gold_agent=gold_obj.get("agent"),
                max_a_candidates=self.max_a_candidates_per_sample,
            )

            candidate_jsons = enumerate_candidate_jsons(candidate_agents)
            gold_index = find_gold_index(candidate_jsons, gold_obj)
            if gold_index < 0:
                add_drop("gold_not_in_candidates")
                continue

            valid_rows.append(
                {
                    "prompt": str(prompt),
                    "candidate_agents": candidate_agents,
                    "candidate_jsons": candidate_jsons,
                    "gold_index": int(gold_index),
                    "gold_label": gold_obj["label"],
                    "gold_agent": gold_obj.get("agent"),
                    "gold_json": make_json_response(gold_obj["label"], gold_obj.get("agent")),
                }
            )

        print(f"[DATASET] dropped={dropped}, reasons={drop_reasons}")
        return valid_rows

    def __len__(self):
        return len(self.rows)

    def _truncate_prompt_ids(self, prompt_ids: List[int], max_prompt_len: int) -> List[int]:
        if len(prompt_ids) <= max_prompt_len:
            return prompt_ids
        if self.prompt_truncation_side == "left":
            return prompt_ids[-max_prompt_len:]
        return prompt_ids[:max_prompt_len]

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt_ids = self.tokenizer(
            row["prompt"],
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

        max_prompt_len = max(1, min(self.max_prompt_length, self.max_length - self.max_response_length))
        prompt_ids = self._truncate_prompt_ids(prompt_ids, max_prompt_len)

        candidate_response_ids = []
        for cand in row["candidate_jsons"]:
            ids = self.tokenizer(
                cand,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_response_length - (1 if self.add_eos else 0),
                return_attention_mask=False,
            )["input_ids"]
            if self.add_eos and self.tokenizer.eos_token_id is not None:
                ids = ids + [self.tokenizer.eos_token_id]
            if len(ids) < 1:
                raise ValueError(f"Empty response ids for candidate={cand}")
            candidate_response_ids.append(ids)

        return {
            "prompt_ids": prompt_ids,
            "candidate_response_ids": candidate_response_ids,
            "candidate_jsons": row["candidate_jsons"],
            "gold_index": row["gold_index"],
            "gold_label": row["gold_label"],
            "gold_agent": row["gold_agent"],
            "gold_json": row["gold_json"],
            "candidate_agents": row["candidate_agents"],
        }


class CandidateJSONScoringCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "prompt_ids": [x["prompt_ids"] for x in batch],
            "candidate_response_ids": [x["candidate_response_ids"] for x in batch],
            "candidate_jsons": [x["candidate_jsons"] for x in batch],
            "gold_index": torch.tensor([x["gold_index"] for x in batch], dtype=torch.long),
            "gold_label": [x["gold_label"] for x in batch],
            "gold_agent": [x["gold_agent"] for x in batch],
            "gold_json": [x["gold_json"] for x in batch],
            "candidate_agents": [x["candidate_agents"] for x in batch],
        }


# =========================================================
# Trainer
# =========================================================

class FSDPCandidateJSONScoringUltraMemTrainer:
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
        self.rank = self.device_mesh.get_rank()

        self.normalize_by = str(getattr(self.config.model, "score_normalize_by", "avg")).lower()
        if self.normalize_by not in {"avg", "sum"}:
            raise ValueError("model.score_normalize_by must be 'avg' or 'sum'.")
        self.gold_lm_loss_weight = float(getattr(self.config.model, "gold_lm_loss_weight", 0.0))
        self.no_grad_candidate_batch_size = int(getattr(self.config.data, "no_grad_candidate_batch_size", 1))
        self.empty_cache_every = int(getattr(self.config.trainer, "empty_cache_every", 0))

        self._normalize_config_bsz()
        self._build_dataloader(train_dataset, val_dataset)
        self._build_model_optimizer()

        if self.rank == 0:
            print(self.config)
            print("[INFO] ultra-memory candidate JSON scoring trainer")
            print("[INFO] no cls_head, no agent_head, no free generation")
            print("[INFO] training uses two-pass score-gradient trick")
            print(f"[INFO] score_normalize_by={self.normalize_by}, gold_lm_loss_weight={self.gold_lm_loss_weight}")

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0)
        if self.rank == 0:
            print(f"[INFO] Normalize batch size by dp={dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )
        self.config.data.train_batch_size //= dp_size
        # In this ultra-mem trainer, micro_batch_size_per_gpu is not used for tensor batching.
        # We still require it to exist for compatibility with existing Hydra configs.

    def _build_dataloader(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        collator = CandidateJSONScoringCollator()

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()

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
            batch_size=1,  # validation is also sample-by-sample for memory safety
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
            self.model.gradient_checkpointing_enable(
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
        if self.rank == 0:
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

        if self.rank == 0:
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

    def _make_candidate_tensors(self, prompt_ids: List[int], resp_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_length = int(getattr(self.config.data, "max_length", 4096))
        pad_id = self.tokenizer.pad_token_id

        max_prompt = max(1, max_length - len(resp_ids))
        pids = prompt_ids[-max_prompt:]
        ids = pids + resp_ids
        labels = [-100] * len(pids) + resp_ids

        if len(ids) > max_length:
            ids = ids[:max_length]
            labels = labels[:max_length]

        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        label_ids = torch.tensor([labels], dtype=torch.long)

        # No padding is needed because candidate_batch_size is one in the grad pass.
        # Keep pad_id reference so tokenizer config errors are surfaced early.
        if pad_id is None:
            raise ValueError("Tokenizer must have pad_token_id")
        return input_ids, attention_mask, label_ids

    def _candidate_nll_from_tensors(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = torch.device(self.device_name)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = self.fsdp_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid_mask = shift_labels.ne(-100)

        vocab_size = shift_logits.size(-1)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size).float(),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.size())

        seq_sum = (token_loss * valid_mask.float()).sum(dim=1)
        seq_count = valid_mask.float().sum(dim=1).clamp_min(1.0)
        if self.normalize_by == "sum":
            return seq_sum.squeeze(0)
        return (seq_sum / seq_count).squeeze(0)

    def _score_one_candidate(self, prompt_ids: List[int], resp_ids: List[int]) -> torch.Tensor:
        input_ids, attention_mask, labels = self._make_candidate_tensors(prompt_ids, resp_ids)
        nll = self._candidate_nll_from_tensors(input_ids, attention_mask, labels)
        return -nll

    @torch.no_grad()
    def _no_grad_scores_for_sample(self, prompt_ids: List[int], candidate_response_ids: List[List[int]]) -> torch.Tensor:
        scores = []
        # One candidate at a time is the safest. Batching no_grad candidates can be enabled
        # by increasing data.no_grad_candidate_batch_size in a future optimization.
        for resp_ids in candidate_response_ids:
            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                score = self._score_one_candidate(prompt_ids, resp_ids)
            scores.append(score.detach().float().cpu())
        return torch.stack(scores, dim=0)

    def _backward_one_sample(self, sample: Dict[str, Any], grad_scale: float) -> Dict[str, torch.Tensor]:
        """Two-pass memory-efficient backward for a single original sample."""
        prompt_ids = sample["prompt_ids"]
        candidate_response_ids = sample["candidate_response_ids"]
        gold_idx = int(sample["gold_index"])

        # Pass 1: compute scores without graph, then CE gradient wrt scores.
        scores_cpu = self._no_grad_scores_for_sample(prompt_ids, candidate_response_ids)
        target = torch.tensor([gold_idx], dtype=torch.long)
        ranking_loss_cpu = F.cross_entropy(scores_cpu.unsqueeze(0), target)
        probs = torch.softmax(scores_cpu, dim=0)
        grad_scores_cpu = probs
        grad_scores_cpu[gold_idx] -= 1.0

        pred_idx = int(torch.argmax(scores_cpu).item())
        gold_nll_cpu = -scores_cpu[gold_idx]

        # Pass 2: recompute one candidate with graph and immediately backward.
        for cand_idx, resp_ids in enumerate(candidate_response_ids):
            grad_coef = float(grad_scores_cpu[cand_idx].item())
            if cand_idx == gold_idx and self.gold_lm_loss_weight != 0.0:
                # loss += w * gold_nll = -w * score_gold
                grad_coef -= self.gold_lm_loss_weight

            # If coefficient is exactly zero, skip to save time.
            if abs(grad_coef) < 1e-12:
                continue

            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                score = self._score_one_candidate(prompt_ids, resp_ids)
                surrogate = score * (grad_coef * grad_scale)
            surrogate.backward()

            # Drop refs early.
            del score, surrogate

        device = torch.device(self.device_name)
        return {
            "ranking_loss": torch.tensor(float(ranking_loss_cpu.item()), device=device),
            "gold_lm_loss": torch.tensor(float(gold_nll_cpu.item()), device=device),
            "pred_index": pred_idx,
            "loss": torch.tensor(
                float(ranking_loss_cpu.item()) + self.gold_lm_loss_weight * float(gold_nll_cpu.item()),
                device=device,
            ),
        }

    @torch.no_grad()
    def _eval_one_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompt_ids = sample["prompt_ids"]
        candidate_response_ids = sample["candidate_response_ids"]
        gold_idx = int(sample["gold_index"])
        scores_cpu = self._no_grad_scores_for_sample(prompt_ids, candidate_response_ids)
        target = torch.tensor([gold_idx], dtype=torch.long)
        ranking_loss_cpu = F.cross_entropy(scores_cpu.unsqueeze(0), target)
        pred_idx = int(torch.argmax(scores_cpu).item())
        gold_nll_cpu = -scores_cpu[gold_idx]
        device = torch.device(self.device_name)
        return {
            "ranking_loss": torch.tensor(float(ranking_loss_cpu.item()), device=device),
            "gold_lm_loss": torch.tensor(float(gold_nll_cpu.item()), device=device),
            "pred_index": pred_idx,
            "loss": torch.tensor(
                float(ranking_loss_cpu.item()) + self.gold_lm_loss_weight * float(gold_nll_cpu.item()),
                device=device,
            ),
        }

    def _sample_from_batch(self, batch: Dict[str, Any], idx: int) -> Dict[str, Any]:
        return {
            "prompt_ids": batch["prompt_ids"][idx],
            "candidate_response_ids": batch["candidate_response_ids"][idx],
            "candidate_jsons": batch["candidate_jsons"][idx],
            "gold_index": int(batch["gold_index"][idx].item()),
            "gold_label": batch["gold_label"][idx],
            "gold_agent": batch["gold_agent"][idx],
            "gold_json": batch["gold_json"][idx],
            "candidate_agents": batch["candidate_agents"][idx],
        }

    def _empty_stats(self) -> Dict[str, torch.Tensor]:
        device = torch.device(self.device_name)
        keys = [
            "loss", "ranking_loss", "gold_lm_loss",
            "total", "label_correct", "joint_correct", "agent_total", "agent_correct",
            "label_total_A", "label_total_B", "label_total_C",
            "label_correct_A", "label_correct_B", "label_correct_C",
        ]
        return {k: torch.zeros([], device=device, dtype=torch.float32) for k in keys}

    def _add_prediction_stats(self, stats: Dict[str, torch.Tensor], sample: Dict[str, Any], pred_idx: int, loss_info: Dict[str, torch.Tensor]):
        stats["loss"] += loss_info["loss"].detach()
        stats["ranking_loss"] += loss_info["ranking_loss"].detach()
        stats["gold_lm_loss"] += loss_info["gold_lm_loss"].detach()

        pred_json = sample["candidate_jsons"][pred_idx]
        pred_obj = parse_response_obj(pred_json)
        pred_label = pred_obj["label"]
        pred_agent = pred_obj.get("agent")

        gold_label = sample["gold_label"]
        gold_agent = sample["gold_agent"]

        stats["total"] += 1.0
        stats[f"label_total_{gold_label}"] += 1.0

        if pred_label == gold_label:
            stats["label_correct"] += 1.0
            stats[f"label_correct_{gold_label}"] += 1.0

        if gold_label == "A":
            stats["agent_total"] += 1.0
            if pred_label == "A" and pred_agent == gold_agent:
                stats["agent_correct"] += 1.0
                stats["joint_correct"] += 1.0
        else:
            if pred_label == gold_label:
                stats["joint_correct"] += 1.0

    def _stats_to_metrics(self, prefix: str, stats: Dict[str, torch.Tensor], lr: Optional[float] = None) -> Dict[str, float]:
        total = max(float(stats["total"].item()), 1.0)
        agent_total = max(float(stats["agent_total"].item()), 1.0)
        total_A = max(float(stats["label_total_A"].item()), 1.0)
        total_B = max(float(stats["label_total_B"].item()), 1.0)
        total_C = max(float(stats["label_total_C"].item()), 1.0)

        metrics = {
            f"{prefix}/loss": float(stats["loss"].item()) / total,
            f"{prefix}/ranking_loss": float(stats["ranking_loss"].item()) / total,
            f"{prefix}/gold_lm_loss": float(stats["gold_lm_loss"].item()) / total,
            f"{prefix}/label_acc": float(stats["label_correct"].item()) / total,
            f"{prefix}/agent_acc": float(stats["agent_correct"].item()) / agent_total,
            f"{prefix}/joint_acc": float(stats["joint_correct"].item()) / total,
            f"{prefix}/acc_A": float(stats["label_correct_A"].item()) / total_A,
            f"{prefix}/acc_B": float(stats["label_correct_B"].item()) / total_B,
            f"{prefix}/acc_C": float(stats["label_correct_C"].item()) / total_C,
            f"{prefix}/label_total_A": float(stats["label_total_A"].item()),
            f"{prefix}/label_total_B": float(stats["label_total_B"].item()),
            f"{prefix}/label_total_C": float(stats["label_total_C"].item()),
            f"{prefix}/agent_total": float(stats["agent_total"].item()),
        }
        if lr is not None:
            metrics[f"{prefix}/lr(1e-3)"] = lr * 1e3
        return metrics

    def training_step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, float]:
        self.fsdp_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        bsz = len(batch["prompt_ids"])
        grad_scale = 1.0 / max(1, bsz)
        stats = self._empty_stats()

        for i in range(bsz):
            sample = self._sample_from_batch(batch, i)
            out = self._backward_one_sample(sample, grad_scale=grad_scale)
            self._add_prediction_stats(stats, sample, int(out["pred_index"]), out)

            if self.empty_cache_every > 0 and (i + 1) % self.empty_cache_every == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        # Sum counts and losses across ranks. Metrics divide by total samples afterward.
        for key in stats:
            torch.distributed.all_reduce(stats[key], op=torch.distributed.ReduceOp.SUM)

        if self.empty_cache_every > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self._stats_to_metrics("train", stats, lr=lr)

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        self.fsdp_model.eval()
        stats = self._empty_stats()
        bsz = len(batch["prompt_ids"])
        for i in range(bsz):
            sample = self._sample_from_batch(batch, i)
            out = self._eval_one_sample(sample)
            self._add_prediction_stats(stats, sample, int(out["pred_index"]), out)
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

        if self.rank == 0:
            os.makedirs(path, exist_ok=True)
            torch.save(state_dict, os.path.join(path, "model.pt"))
            self.model_config.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

            meta = {
                "checkpoint_type": "candidate_json_scoring_sft_ultra_mem",
                "label_to_id": LABEL_TO_ID,
                "id_to_label": ID_TO_LABEL,
                "score_normalize_by": self.normalize_by,
                "gold_lm_loss_weight": self.gold_lm_loss_weight,
                "response_format": {"label": "A/B/C", "agent": "agent_name_or_null"},
                "recommended_inference": "candidate_json_loss_scoring",
                "note": "trained with two-pass score-gradient trick to reduce activation memory",
            }
            with open(os.path.join(path, "candidate_json_scoring_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        if self.rank == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def fit(self):
        rank = self.rank
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
                metric = self.training_step(data, global_step=global_step)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                    if global_step <= 3 or global_step % 10 == 0:
                        print(f"[TRAIN] step={global_step} metric={metric}")

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = self.config.trainer.test_freq > 0 and global_step % self.config.trainer.test_freq == 0
                is_save_step = self.config.trainer.save_freq > 0 and global_step % self.config.trainer.save_freq == 0

                if is_last_step or is_valid_step:
                    val_stats = self._empty_stats()
                    for val_data in self.val_dataloader:
                        out = self.validation_step(val_data)
                        for key in out:
                            val_stats[key] += out[key]

                    for key in val_stats:
                        torch.distributed.all_reduce(val_stats[key], op=torch.distributed.ReduceOp.SUM)

                    if rank == 0:
                        metric = self._stats_to_metrics("val", val_stats)
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
    return CandidateJSONScoringDataset(raw_rows=raw_rows, tokenizer=tokenizer, config=data_config)


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
            print("gold_label:", item["gold_label"])
            print("gold_agent:", item["gold_agent"])
            print("gold_json:", item["gold_json"])
            print("gold_index:", item["gold_index"])
            print("candidate_agents:", item["candidate_agents"])
            print("candidate_jsons:", item["candidate_jsons"][:10])
            print("num_candidates:", len(item["candidate_jsons"]))
            print("prompt len:", len(item["prompt_ids"]))
            print("decoded prompt tail:", tokenizer.decode(item["prompt_ids"][-120:]))
        print("===== END DEBUG =====")

    trainer = FSDPCandidateJSONScoringUltraMemTrainer(
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
