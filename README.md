# Terminology-aware Medical Dialogue Generation
This repository is the code and resources for the paper [Terminology-aware Medical Dialogue Generation](https://arxiv.org/pdf/2210.15551.pdf) 

## Instructions

This project is implemented with **Pytorch**.

This project is based on [pytorch-lightning](https://www.pytorchlightning.ai/) framework, and all pretrained models can be downloadeded from [Hugginface](https://huggingface.co).

So if you want to run this code, you must have following preliminaries:
- Python 3 or Anaconda (mine is 3.8)
- [Pytorch](https://pytorch.org/) 
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base))
- [pytorch-lightning (a package)](https://www.pytorchlightning.ai/)

## Datasets and Resources

### Directly Download Dataset and Resources
To reproduce our work you need to download following files:

- Processed data (put them to `datasets/med-dialog` directory) [med-dialog](https://www.dropbox.com/s/roewcfiw2u08g5w/med-dialog.zip?dl=0)

- Medicial Terminology List (put it to `resources/med-dialog` directory) [med_term_list.txt](https://www.dropbox.com/s/cpl5mbw2sy73dcn/med_term_list.txt?dl=0)

### Preprocess Dataset From Scratch

The raw dialogue corpus is downloaded from the work of [Medical-Dialogue-System](https://github.com/UCSD-AI4H/Medical-Dialogue-System), 
or you can download it from [here](https://www.dropbox.com/s/bmuoxzi587pz4v3/large-english-dialog-corpus.zip?dl=0).

Put it to `resources/med-dialog` directory.

### Put Files To Correct Destinations 

Unzip these files, and your `datasets` and `resources` should be as follows.

The structure of `datasets`should be like this:
```markdown
├── datasets/med-dialog
   └── dialog-with-term		# dialogs with term tags
          └── `train.source.txt`    
          └── `train.target.txt`       
          └── `val.source.txt` 
          └── `val.target.txt` 
          └── `test.source.txt` 
          └── `test.target.txt` 
    └── large-english-dialog-corpus		# dialog datasets without terms
          └── `train.source.txt`    # input: dialogue history
          └── `train.target.txt`    # reference output: response from the doctor 
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

The structure of `datasets` should be like this:
```markdown
├── resources/med-dialog
   └── large-english-dialog-corpus		# the raw dialogue corpus
          └── `train.source.txt`    
          └── `train.target.txt`       
          └── `val.source.txt` 
          └── `val.target.txt` 
          └── `test.source.txt` 
          └── `test.target.txt` 
    └── med_term_list.txt		# the terminology list
```

The terminology list is acquired from the word of [wordlist-medicalterms-en](https://github.com/glutanimate/wordlist-medicalterms-en/blob/master/wordlist.txt)

## Quick Start

### 1. Install packages
```shell
python -r requirements.txt
```
### 2. Collect Datasets and Resources

As mentioned above.

### 3. Run the code for training or testing

Train bart -w terms AL:

```shell
python tasks/med-dialog/train.py --model_name terms_bart --experiment_name=term_bart-base-meddialog\
 --learning_rate=2e-5 --train_batch_size=6 --eval_batch_size=6 --model_name_or_path=facebook/bart-base \
 --val_check_interval=0.5 --max_epochs=6 --accum_batches_args=12  --num_sanity_val_steps=1 \
 --save_top_k 3 --eval_beams 2 --data_dir=datasets/med-dialog/dialog-with-term \
 --limit_val_batches=20
```

Test bart -w terms AL:

```shell
python tasks/med-dialog/test.py\
  --eval_batch_size=32 --model_name_or_path=facebook/bart-base \
  --output_dir=output/med-dialog/ --model_name terms_bart --experiment_name=term_bart-base-meddialog --eval_beams 2 \
  --max_target_length=400
```

If you also want to try baselines, please read the code of
`tasks/med-dialog/train.py` and `tasks/med-dialog/test.py`. I believe you will understand what to do.


## Notation
Some notes for this project.
#### 1 - Complete Project Structure
```markdown
├── src # source code
├── tasks # code for running programs
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

## Citation
If you found this repository or paper is helpful to you, please cite our paper. 
Currently we only have arxiv citation listed as follows:

This is the arxiv citation:
```angular2
@misc{https://doi.org/10.48550/arxiv.2210.15551,
  doi = {10.48550/ARXIV.2210.15551},
  url = {https://arxiv.org/abs/2210.15551},
  author = {Tang, Chen and Zhang, Hongbo and Loakman, Tyler and Lin, Chenghua and Guerin, Frank},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Terminology-aware Medical Dialogue Generation},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


