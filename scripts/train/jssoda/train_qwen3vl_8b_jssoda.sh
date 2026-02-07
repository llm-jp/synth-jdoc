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

deepspeed --num_gpus 8 src/train/jssoda/train_qwen3vl_jssoda.py \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --output_dir ./checkpoints/Qwen3-VL-8B-Instruct-JSSODa-train \
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
