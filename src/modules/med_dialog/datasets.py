"""
@Desc:
@Reference:
- transformers examples for using BART model
https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization
https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904/2
- add_special_tokens
https://huggingface.co/docs/transformers/v4.17.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
- linecache
https://blog.csdn.net/my2010Sam/article/details/38022041
- torch Dataset
https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset
@Notes:
- add_special_tokens
special_tokens_dict (dictionary str to str or tokenizers.AddedToken) —
Keys should be in the list of predefined special attributes:
[bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, additional_special_tokens].
Tokens are only added if they are not already in the vocabulary (tested by checking
if the tokenizer assign the index of the unk_token to them).
- collate_fn
A custom collate_fn can be used to customize collation, e.g., padding sequential data to max length of a batch.
See this section on more about collate_fn.
"""
from typing import List
import torch
from transformers import GPT2Tokenizer, BartTokenizer

from src.modules.datasets_base import BaseDataset

TERM_TOKEN = "[TERM]"
SEP_TOKEN = "[SEP]"

class EnglishDialogDataset(BaseDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)
        self.tokenizer.sep_token = SEP_TOKEN
        self.tokenizer.add_special_tokens({'additional_special_tokens':
                                               [TERM_TOKEN, SEP_TOKEN],
                                           })

    def __getitem__(self, index):
        source_line = self.src_data[index]
        target_line = self.tgt_data[index]
        assert source_line, f"empty source line for index {index}"
        assert target_line, f"empty tgt line for index {index}"
        return {"src_text": source_line, "tgt_text": target_line, "data_id": index}

    def collate_fn(self, batch):
        batch_encoding = self.tokenizer(
            [x["src_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [x["tgt_text"] for x in batch],
                add_special_tokens=True,
                truncation=True,
                padding="longest",
                max_length=self.max_target_length,
                return_tensors="pt",
            ).data
        batch_encoding["labels"] = labels["input_ids"]
        batch_encoding["ids"] = [x["data_id"] for x in batch]
        return batch_encoding

    def __len__(self):
        return len(self.src_data)

class GPT2EnglishDialogDataset(EnglishDialogDataset):
    def __init__(self, tokenizer: GPT2Tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)

    def collate_fn(self, batch):
        batch_encoding = self.tokenizer(
            [x["src_text"] for x in batch],
            [x["tgt_text"] + f" {self.tokenizer.eos_token}" for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        src_batch = self.tokenizer(
            [x["src_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data

        tgt_batch = self.tokenizer(
            [x["tgt_text"] + f" {self.tokenizer.eos_token}" for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        batch_encoding["src_ids"] = src_batch["input_ids"]
        batch_encoding["src_attention_mask"] = src_batch["attention_mask"]
        batch_encoding["tgt_ids"] = tgt_batch["input_ids"]
        token_type_ids = batch_encoding["attention_mask"].clone()
        token_type_ids[:, :batch_encoding["src_attention_mask"].shape[1]] -= batch_encoding["src_attention_mask"]
        batch_encoding["token_type_ids"] = token_type_ids
        batch_encoding["ids"] = [x["data_id"] for x in batch]
        return batch_encoding

class EnglishDialogDatasetWithTerms(EnglishDialogDataset):
    def __init__(self, tokenizer: BartTokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)

    def collate_fn(self, batch):
        """
        let bart to do classification of whether the word is a term.
        """
        batch_encoding = self.tokenizer(
            [x["src_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [x["tgt_text"] for x in batch],
                add_special_tokens=True,
                truncation=True,
                padding="longest",
                max_length=self.max_target_length,
                return_tensors="pt",
            ).data
        batch_encoding["labels"] = labels["input_ids"]
        batch_encoding["ids"] = [x["data_id"] for x in batch]

        # add word labels
        input_texts = [x["src_text"] for x in batch]
        batch_encoding["token_labels"] = torch.zeros(batch_encoding["input_ids"].shape)
        for idx in range(len(input_texts)):
            sent = input_texts[idx]
            terms_list = self.get_terms_list(sent)
            tokenized_tokens = self.tokenizer.tokenize(sent)
            labels_for_encoded_words = self.get_labels_for_encoded_words(tokenized_tokens, terms_list)
            assert len(tokenized_tokens) == len(labels_for_encoded_words)
            if len(labels_for_encoded_words)>batch_encoding['token_labels'].shape[1]:
                labels_for_encoded_words = labels_for_encoded_words[:batch_encoding['token_labels'].shape[1]]
            batch_encoding["token_labels"][idx,:len(labels_for_encoded_words)] = torch.tensor(labels_for_encoded_words)

        return batch_encoding

    def are_two_list_equal(self, obj1, obj2):
        obj1 = list(obj1)
        obj2 = list(obj2)
        return obj1 == obj2

    def get_terms_list(self, sent: str):
        terms_list = []
        sent_words = sent.strip().split()
        for idx in range(len(sent_words)):
            if sent_words[idx] == TERM_TOKEN:
                terms_list.append(f"{sent_words[idx]} {sent_words[idx+1]}")
        return terms_list

    def get_labels_for_encoded_words(self, tokenized_tokens: List[str], terms_list):
        tokenized_terms = []
        for term in terms_list:
            tokenized_terms.append(self.tokenizer.tokenize(term))
        labels_for_encoded_words = [0] * len(tokenized_tokens)
        for idx in range(len(tokenized_tokens)):
            # for tokenized_term in tokenized_terms:
            if len(tokenized_terms) > 0:
                tokenized_term = tokenized_terms[0]
                et_len = len(tokenized_term)
                if self.are_two_list_equal(tokenized_tokens[idx: idx + et_len], tokenized_term):
                    for temp_id in range(idx, idx+et_len):
                        labels_for_encoded_words[temp_id] = 1
                    tokenized_terms.pop(0)
        assert len(tokenized_terms) == 0
        return labels_for_encoded_words


