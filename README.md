# Terminology-aware Medical Dialogue Generation
Code and dataset for paper _Terminology-aware Medical Dialogue Generation_

## Dataset
The dataset and its corresponding citation relationship can be downloaded through this link.

The structure of `datasets`should be like this:
```markdown
├── datasets
   └── dialog-with-term		# dialogs with term tags
          └── `train.source.txt`    # leading context of dialog
          └── `train.target.txt`       # responses to the leading context
          └── `val.source.txt` 
          └── `val.target.txt` 
          └── `test.source.txt` 
          └── `test.target.txt` 
    └── dialog-with-term		# raw dialog datasets
          └── `train.source.txt`    # leading context of dialog
          └── `train.target.txt`       # responses to the leading context
          └── `val.source.txt` 
          └── `val.target.txt` 
          └── `test.source.txt` 
          └── `test.target.txt` 
```
train, val, test are split by the ratio of 0.90, 0.05, 0.05

the example of `test.source.txt` (leading context):

`patient: i have been [TERM] diagnosed [TERM] with bppv [TERM] and sjogrens syndrome. sometimes i experience [TERM] horizontal [TERM] rolling [TERM] like old tv s [TERM] used to do. this usually occurs when in a semi-recline [TERM] position [TERM] and lasts [TERM] for about 10-15 seconds. i [TERM] can t find [TERM] any information. [TERM] can you help?`

the example of `test.target.txt` (story):

`bppv is due to [TERM] ear problem, [TERM] and there is n [TERM] relation to sjogren's. it [TERM] can be treated by a good [TERM] ent specialist.`


## Instructions
This project is based on [pytorch-lightning](https://www.pytorchlightning.ai/) framework, and all pretrained models can be downloadeded from [Hugginface](https://huggingface.co).

So if you want to run this code, you must have following preliminaries:
- Python 3 or Anaconda (mine is 3.8)
- [Pytorch](https://pytorch.org/) 
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base))
- [pytorch-lightning (a package)](https://www.pytorchlightning.ai/)

## Quick Start

#### 1. Install packages
```shell
python -r requirements.txt
```
#### 2. Fine-tuning BART on Our datasets
I have set all essential parameters, so you can directly run 

`python ./tasks/med-dialog/train.py`

**Or** 

If you want to modify parameters, you can run
```shell
python tasks/med-dialog/train.py --model_name terms_bart --experiment_name=term_bart-large2-meddialog\
 --learning_rate=2e-5 --train_batch_size=6 --eval_batch_size=6 --model_name_or_path=facebook/bart-large \
 --val_check_interval=0.5 --max_epochs=6 --accum_batches_args=12  --num_sanity_val_steps=1 \
 --save_top_k 3 --eval_beams 2 --data_dir=datasets/med-dialog/dialog-with-term \
 --limit_val_batches=20
```

#### 4. Generating Responses and Evaluation
Same to training. Directly run 

`python ./tasks/med-dialog/test.py`

**Or** 

```shell
python tasks/med-dialog/test.py\
  --eval_batch_size=32 --model_name_or_path=facebook/bart-large \
  --output_dir=output/med-dialog/ --model_name terms_bart --experiment_name=term_bart-base-meddialog --eval_beams 2 \
  --max_target_length=400
```

## Notation
Some notes for this project.
#### 1 - Complete Prject Structure
```markdown
├── datasets 
├── output  # this will be automatically created to put all the output stuff including checkpoints and generated text
├── resources # put some resources used by the model e.g. the pretrained model.
├── tasks # excute programs e.g. training, tesing, generating stories
├── .gitignore # used by git
├── requirement.txt # the checklist of essential python packages 
```
#### 2 - Scripts for Downloading huggingface models
I wrote two scripts to download models from huggingface website.
One is `tasks/download_hf_models.sh`, and another is `src/utils/huggingface_helper.py`
