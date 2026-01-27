#!/bin/bash
#PBS -P GROUP
#PBS -q RESOURCE_TYPE
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -N JOB_NAME
#PBS -k oed

cd ${PBS_O_WORKDIR}

source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1 cudnn/9.10/9.10.2 nccl/2.25/2.25.1-1

source .venv/bin/activate

deepspeed --num_gpus 8 src/train/train_qwen2_5vl.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --output_dir ./checkpoints/Qwen2.5-VL-7B-Instruct-HTML-Synth-train \
    --deepspeed ./scripts/train/zero2.json \
    --seed 42 \
    --learning_rate 2e-05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --bf16 True \
    --optim adamw_torch \
    --save_strategy epoch \
    --push_to_hub False \
    --report_to wandb \
    --logging_steps 1
