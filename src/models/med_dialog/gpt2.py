"""
@Desc:
@Reference:
- GPT2LMHeadModel
https://huggingface.co/docs/transformers/model_doc/gpt2
- GPT2 for text generation
https://github.com/dredwardhyde/nlp-models-examples/blob/main/gpt2_openai_text_generator.py
@Notes:
"""

import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Config
from transformers import GPT2Tokenizer

from src.utils import nlg_eval_utils
from src.modules.med_dialog.datasets import (
    GPT2EnglishDialogDataset,
)
from src.utils.med_dialog import model_utils
from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.models.med_dialog.bart import MyBart

logger = logging.getLogger(__name__)


class MyGPT2(MyBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # config
        self.config: GPT2Config = GPT2Config.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, GPT2LMHeadModel, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
        self.dataset_class = GPT2EnglishDialogDataset

    def _step(self, batch: dict):
        outputs = self(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       token_type_ids=batch["token_type_ids"],
                       labels=batch["input_ids"],
                       return_dict=True)

        lm_logits = outputs["logits"]

        if self.hparams.label_smoothing == 0:
            assert lm_logits.shape[-1] == self.vocab_size
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
            losses_ = outputs["loss"]
            loss = torch.mean(losses_)
        else:
            lprobs = torch.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = model_utils.label_smoothed_nll_loss(
                lprobs, batch["input_ids"][..., 1:].contiguous(),
                self.hparams.label_smoothing, ignore_index=self.pad_token_id
            )
        return loss

    def training_step(self, batch, batch_idx) -> Dict:
        loss = self._step(batch)
        logs = {"loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        # tokens per batch
        logs["tokens_per_batch"] = batch["input_ids"].ne(self.pad_token_id).sum()
        logs["batch_size"] = batch["input_ids"].shape[0]
        logs["source_pad_tokens_num"] = batch["input_ids"].eq(self.pad_token_id).sum()
        logs["source_pad_tokens_ratio"] = batch["input_ids"].eq(self.pad_token_id).float().mean()
        return {"loss": loss, "log": logs}

    @torch.no_grad()
    def sample_sequence(self, batch, use_top_p=False, top_p=0.9):
        # eval_batch_size must be 1, otherwise the output will be wrong.
        # pad = endoftext, the generated text will be endoftext.
        batch_size = len(batch["ids"])
        generated_ids = None
        input_ids = batch["src_ids"]
        attention_mask = batch["src_attention_mask"]
        eos_counter = torch.zeros([batch_size]).to(self.device)
        past_key_values = None
        temperature = 1
        context = input_ids

        for _ in range(self.hparams.max_target_length):
            outputs = self(input_ids=context,
                           attention_mask=attention_mask,
                           past_key_values=past_key_values,
                           return_dict=True)
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]
            if past_key_values is None:
                next_token_logits = logits[:, -1, :] / temperature
            else:
                #next_token_logits = logits / temperature
                next_token_logits = logits[:, -1, :] / temperature
            #next_token_logits = logits[:, -1, :] / temperature

            if use_top_p:
                next_token_logits = top_p_logits(next_token_logits, p=top_p, device=self.device)
                probs = torch.softmax(next_token_logits, dim=-1)
                preds = torch.multinomial(probs, 1)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                preds = torch.topk(input=probs, k=1).indices

            if generated_ids is None:
                generated_ids = preds
            else:
                generated_ids = torch.cat([generated_ids, preds], dim=1)
            context = preds.unsqueeze(0)

            # # early stop
            # eos_counter += preds[:, 0].eq(self.tokenizer.eos_token_id)
            # if eos_counter.ge(1).sum() == batch_size:
            #     break
        return generated_ids

    @torch.no_grad()
    def _generative_step(self, batch: dict, fast_generate=False) -> dict:
        tik = datetime.now()

        #generated_ids = self.sample_sequence(batch, use_top_p=self.use_top_p, top_p=self.top_p)
        input_ids = batch["src_ids"]
        attention_mask = batch["src_attention_mask"]
        cut_pos = [input_ids.shape[1] for _ in range(len(input_ids))]
        for aid,att in enumerate(attention_mask):
            for p,a in enumerate(att):
                if a == 0:
                    cut_pos[aid] = p
                    break
        generated_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            pad_token_id=self.decoder_start_token_id,
            min_length = 100+input_ids.shape[1],
            max_length=self.hparams.max_target_length+input_ids.shape[1],
            top_p=self.top_p if self.use_top_p else None,
        )
        preds = []
        for id,g_ids in enumerate(generated_ids):
            preds.extend(self.gen_ids_to_clean_text([generated_ids[id,cut_pos[id]:]]))
        #generated_ids = self.sample_sequence(batch, use_top_p=self.use_top_p, top_p=self.top_p)
        tok = datetime.now()
        batch_gen_time = tok - tik
        #preds: List[str] = self.gen_ids_to_clean_text(generated_ids)
        print(preds)
        targets: List[str] = self.gen_ids_to_clean_text(batch["tgt_ids"])
        source: List[str] = self.gen_ids_to_clean_text(batch["input_ids"])
        loss = self._step(batch)

        base_metrics = {"loss": loss.item()}
        rouge_metrics: Dict = nlg_eval_utils.calculate_rouge(pred_lines=preds, tgt_lines=targets)
        base_metrics.update(**rouge_metrics)
        bleu_metrics: Dict = nlg_eval_utils.calculate_bleu(ref_lines=[self.tokenizer.tokenize(l) for l in targets],
                                                           gen_lines=[self.tokenizer.tokenize(l) for l in preds])
        base_metrics.update(**bleu_metrics)
        summ_len = np.mean(list(map(len, generated_ids)))

        # update metric_names
        self.update_metric_names(base_metrics, update_flag=self.metric_names_update_flag)
        self.metric_names_update_flag = False
        base_metrics.update(batch_gen_time=batch_gen_time, gen_len=summ_len,
                            preds=preds, targets=targets,source=source)
        return base_metrics
