"""
@Desc:
@Reference:
- t5
https://huggingface.co/docs/transformers/model_doc/t5
@Notes:
"""

import logging
import torch

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Config
from transformers import T5Tokenizer
from src.modules.med_dialog.bart_modules import T5WithTermsForCG

from src.modules.med_dialog.datasets import (
    EnglishDialogDataset, EnglishDialogDatasetWithTerms
)
from src.utils.med_dialog import model_utils
from src.models.med_dialog.bart import MyBart, MyBartWithTermClassification

logger = logging.getLogger(__name__)

class MyT5(MyBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # config
        self.config: T5Config = T5Config.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model: T5ForConditionalGeneration = \
            self._load_model(self.hparams.model_name_or_path, T5ForConditionalGeneration, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
        self.dataset_class = EnglishDialogDataset

    def _step(self, batch: dict):
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]

        outputs = self(src_ids,
                       attention_mask=src_mask,
                       labels=tgt_ids,
                       use_cache=False,
                       output_attentions=True, output_hidden_states=True)

        loss = outputs['loss']
        return loss

class MyT5WithTermClassification(MyBartWithTermClassification):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # config
        self.config: T5Config = T5Config.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model: T5ForConditionalGeneration = \
            self._load_model(self.hparams.model_name_or_path, T5WithTermsForCG, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
        self.dataset_class = EnglishDialogDatasetWithTerms
        self.loss_names_update_flag = True
