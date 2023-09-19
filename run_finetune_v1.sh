#!/bin/bash#
CUDA_VISIBLE_DEVICES=3 python3 finetune.py --data_path "data/vi_all_v1.0.jsonl" --base_model "../llm/models/bloom-7b1" --model_family "bloom" \
  --finetune_method "qlora" --lora_r 64 --lora_alpha 16 --output_dir "bloom-7b1-instruct-v1.0" \
    --batch_size 128 --micro_batch_size 8 --cutoff_len 512 --num_epochs 3 --kbit "4bit" \
