#!/bin/bash#
CUDA_VISIBLE_DEVICES=0 python3 finetune.py --data_path "data/vi_merged.jsonl" --base_model "../llm/models/bloom-7b1" --model_family "bloom" \
  --finetune_method "qlora" --lora_r 16 --lora_alpha 16 --output_dir "chat-bloom-7b1-3e" \
    --batch_size 128 --micro_batch_size 4 --cutoff_len 512 --num_epochs 2 --kbit "4bit" \
