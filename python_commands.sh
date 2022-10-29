# =============================== data preprocess ====================
python tasks/med-dialog/preprocess_for_corpus.py
python tasks/med-dialog/enhance_dataset.py
# =============================== train ====================
# without terms
#python tasks/med-dialog/train.py --model_name gpt2 --experiment_name=dialgpt-large-meddialog\
# --learning_rate=1e-4 --train_batch_size=3 --eval_batch_size=3 --model_name_or_path=microsoft/DialoGPT-large \
# --val_check_interval=1.0 --max_epochs=5 --accum_batches_args=12  --num_sanity_val_steps=1 \
# --save_top_k 3 --eval_beams 2 --data_dir=datasets/med-dialog/large-english-dialog-corpus --max_target_length=400 \
# --limit_val_batches=20

#python tasks/med-dialog/train.py --model_name t5 --experiment_name=t5-large-meddialog\
# --learning_rate=1e-4 --train_batch_size=6 --eval_batch_size=6 --model_name_or_path=t5-large \
# --val_check_interval=0.5 --max_epochs=5 --accum_batches_args=12  --num_sanity_val_steps=1 \
# --save_top_k 3 --eval_beams 2 --data_dir=datasets/med-dialog/large-english-dialog-corpus --max_target_length=400 \
# --limit_val_batches=20
#
#python tasks/med-dialog/train.py --model_name bart --experiment_name=bart-large-meddialog\
# --learning_rate=1e-4 --train_batch_size=6 --eval_batch_size=6 --model_name_or_path=facebook/bart-large \
# --val_check_interval=1.0 --max_epochs=5 --accum_batches_args=12  --num_sanity_val_steps=1 \
# --save_top_k 3 --eval_beams 2 --data_dir=datasets/med-dialog/large-english-dialog-corpus --max_target_length=400 \
# --limit_val_batches=20
#
#python tasks/med-dialog/train.py --model_name t5 --experiment_name=t5-base-norl-meddialog\
# --learning_rate=1e-4 --train_batch_size=6 --eval_batch_size=6 --model_name_or_path=t5-base \
# --val_check_interval=1.0 --max_epochs=5 --accum_batches_args=12  --num_sanity_val_steps=1 \
# --save_top_k 3 --eval_beams 2 --data_dir=datasets/med-dialog/dialog-with-term --max_target_length=400 \
# --limit_val_batches=20

# with terms
#python tasks/med-dialog/train.py --model_name terms_bart --experiment_name=term_bart-large2-meddialog\
# --learning_rate=2e-5 --train_batch_size=6 --eval_batch_size=6 --model_name_or_path=facebook/bart-large \
# --val_check_interval=0.5 --max_epochs=6 --accum_batches_args=12  --num_sanity_val_steps=1 \
# --save_top_k 3 --eval_beams 2 --data_dir=datasets/med-dialog/dialog-with-term \
# --limit_val_batches=20

###
## =============================== test ====================
##without terms
python tasks/med-dialog/test.py\
  --eval_batch_size=16 --model_name_or_path=facebook/bart-large \
  --output_dir=output/med-dialog/ --model_name bart --experiment_name=bart-large-meddialog --eval_beams 2 \
  --max_target_length=424
#
#python tasks/med-dialog/test.py\
#  --eval_batch_size=16 --model_name_or_path=t5-large \
#  --output_dir=output/med-dialog/ --model_name t5 --experiment_name=t5-large-meddialog --eval_beams 2 \
#  --max_target_length=424
#
### with terms
#python tasks/med-dialog/test.py\
#  --eval_batch_size=32 --model_name_or_path=facebook/bart-base \
#  --output_dir=output/med-dialog/ --model_name terms_bart --experiment_name=term_bart-base-meddialog --eval_beams 2 \
#  --max_target_length=400
#
#python tasks/med-dialog/test.py\
#  --eval_batch_size=32 --model_name_or_path=t5-base \
#  --output_dir=output/med-dialog/ --model_name terms_t5 --experiment_name=term_t5-base-meddialog --eval_beams 2 \
#  --max_target_length=400