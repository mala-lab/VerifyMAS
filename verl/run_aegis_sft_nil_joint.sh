#!/bin/bash
# Qwen2.5-7B-Instruct + Verl FSDP SFT
# 用法: ./run_detector_sft_qwen25_7b.sh <gpu数量> <保存路径>

set -x

if [ "$#" -lt 2 ]; then
    echo "用法: $0 <nproc_per_node> <save_path>"
    echo "示例: $0 4 ./detector_qwen25_7b_sft_output"
    exit 1
fi

nproc_per_node=$1
save_path=$2

TRAIN_DATA="./data/maserror/data/train_aegis_sft_no_agent_multi_agent_jsonl_balanced_masking.parquet"
VAL_DATA="./data/maserror/data/validation_aegis_sft_no_agent_multi_agent_jsonl_balanced_masking.parquet"

mkdir -p "$save_path"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
  -m verl.trainer.fsdp_direct_full_json_generation_sft_trainer \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.prompt_key=prompt \
  data.response_key=response \
 optim.lr=1e-4 \
  data.micro_batch_size_per_gpu=2 \
  data.train_batch_size=64 \
  data.max_length=8192 \
  data.truncation=right \
  model.partial_pretrain=Qwen/Qwen2.5-7B \
  model.trust_remote_code=True \
  model.lora_rank=64 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  model.fsdp_config.cpu_offload=False \
  model.fsdp_config.offload_params=False \
  model.fsdp_config.model_dtype=bf16 \
  model.enable_gradient_checkpointing=True \
  model.strategy=fsdp \
  trainer.default_local_dir=$save_path \
  trainer.project_name=mas_error_attribution \
  trainer.experiment_name=qwen25-7b-instruct-sft \
  trainer.logger='[console]' \
  trainer.total_epochs=5 \
  trainer.save_freq=100 \
  trainer.test_freq=-1 \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$nproc_per_node