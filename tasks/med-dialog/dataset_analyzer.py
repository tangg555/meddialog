"""
@Desc:
@Reference:
clinical terminology dict
https://github.com/glutanimate/wordlist-medicalterms-en/blob/master/wordlist.txt
@Notes:
"""

import os
import sys
from pathlib import Path
import json
from typing import List
from preprocess import load_dataset, make_dataset, split_data

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.modules.med_dialog.datasets import TERM_TOKEN

def load_term_set(term_file: Path):
    term_set = set()
    with term_file.open("r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            term_set.add(line)
    print(f"terms is loaded from {term_file}, total {len(term_set)}")
    return term_set


if __name__ == '__main__':
    term_path = Path(f"{BASE_DIR}/resources/med_term_list.txt")
    origin_data_dir = Path(f"{BASE_DIR}/datasets/med-dialog/large-english-dialog-corpus")
    output_dir = Path(f"{BASE_DIR}/datasets/med-dialog/dialog-with-term")
    os.makedirs(output_dir, exist_ok=True)

    term_set = load_term_set(term_path)
    data_stat = {"lines": 0, "utterances": 0, "words": 0, "terms": 0}
    for file_name in os.listdir(origin_data_dir):
        with origin_data_dir.joinpath(file_name).open("r", encoding="utf-8") as fr, \
                output_dir.joinpath(file_name).open("w", encoding="utf-8") as fw:
            added_term_count = 0
            line_count = 0
            for line in fr:
                line_count += 1
                words = []
                for origin_w in line.strip().split():
                    if origin_w not in term_set:
                        words.append(origin_w)
                    else:
                        words.append(f"{TERM_TOKEN} {origin_w}")
                        added_term_count += 1
                new_line = f"{' '.join(words).strip()}\n"
                fw.write(new_line)
                # add data stat
                data_stat["lines"] += 1
                data_stat["utterances"] += len(new_line.strip().split("[SEP]"))
                data_stat["words"] += len(new_line.strip().split()) # including special tokens
                data_stat["terms"] += len(new_line.strip().split("[TERM]"))

            print(f"{added_term_count} terms have been added to {output_dir.joinpath(file_name)}, "
                  f"which has {line_count} lines.")
            data_stat["avg. utterances"] = round(data_stat["utterances"] / data_stat["lines"], 2)
            data_stat["avg. words"] = round(data_stat["words"] / data_stat["lines"], 2)
            data_stat["avg. terms"] = round(data_stat["terms"] / data_stat["lines"], 2)
            print(f"data_stat: {data_stat}")
