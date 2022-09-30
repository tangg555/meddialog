#!/bin/bash
set -e

# =============================== train ====================
python tasks/med-dialog/train.py --model_name bart --experiment_name=english-dialog-bart\
 --learning_rate=3e-5 --train_batch_size=10 --eval_batch_size=5 --model_name_or_path=resources/external_models/bart-base \
 --val_check_interval=1.0 --limit_val_batches=2 --max_epochs=10 --accum_batches_args=16  --num_sanity_val_steps=1 \
 --save_top_k 3 --eval_beams 5 --data_dir=datasets/med-dialog/english-dialog

# =============================== test ====================
python tasks/med-dialog/test.py\
  --eval_batch_size=5 --model_name_or_path=output/med-dialog/english-dialog-bart/best_tfmr \
  --output_dir=output/summarisation --model_name bart --experiment_name=med-dialog --eval_beams 5