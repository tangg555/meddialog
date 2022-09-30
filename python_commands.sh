#!/bin/bash
set -e

# =============================== train ====================
python tasks/med-dialog/train.py --model_name bart --experiment_name=med-dialog\
 --learning_rate=1e-4 --train_batch_size=40 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --val_check_interval=0.5 --limit_val_batches=5 --max_epochs=10 --accum_batches_args=16  --num_sanity_val_steps=1 \
 --save_top_k 3 --eval_beams 10 --data_dir=datasets/med-dialog/english-dialog

# =============================== test ====================
python tasks/med-dialog//test.py\
  --eval_batch_size=10 --model_name_or_path=output/med-dialog/bart/best_tfmr \
  --output_dir=output/summarisation --model_name bart --experiment_name=med-dialog --eval_beams 10