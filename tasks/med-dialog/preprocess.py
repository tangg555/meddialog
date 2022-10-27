"""
@Desc:
@Reference:
Our preprocess code refers to :
https://github.com/UCSD-AI4H/COVID-Dialogue/blob/master/src/preprocess.py
@Notes:
"""
import os
import sys
from pathlib import Path
import json
from typing import List
import numpy as np

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

np.random.seed(22)

from src.modules.med_dialog.datasets import (TERM_TOKEN, SEP_TOKEN)

def return_if_is_id(line):
    line = line.strip()
    keyword = "id="
    kw_len = len(keyword)
    if line[:kw_len].lower() == "id=":
        return int(line[kw_len:])
    else:
        return None

def return_if_is_kw(line):
    kw_list = ["id=", "description", "dialogue", "patient:", "doctor:"]
    for kw in kw_list:
        if line.lower() == kw.lower():
            return line
    return None

def load_dataset(dataset_file=f"{BASE_DIR}/resources/med-dialog/covid_english.txt"):
    '''
    对 dataset_file 文件进行清洗处理并储存为 json 文件
    '''
    dataset_file = Path(dataset_file)

    data = {"ids": [], "data_frames":[]}
    pre_id = None
    with dataset_file.open("r", encoding="utf-8") as fr:
        id_content_records = {}
        for line in fr:
            # all words should be lower case
            line = line.strip().replace("﻿", "").lower() # ﻿ in healthcaremagic_dialogue_1.txt
            if not line:
                continue
            id = return_if_is_id(line)
            if id is not None:
                if id in id_content_records:
                    id_content_records[id] = []
                    continue # overwrite the former one
                id_content_records[id] = []
                # ---- validation -------
                if pre_id is not None:
                    assert pre_id == id - 1
                pre_id = id
            elif pre_id is None:
                raise ValueError(f"line:{line}, id:{id}, pre_id={pre_id}")
            else:
                id_content_records[pre_id].append(line)

        is_dirty_sample = False
        dirty_sample_count = 0
        for id, id_content_lines in id_content_records.items():
            assert id not in data["ids"]
            data_frame = {}
            # ---- parse content ----
            kw = None
            for line in id_content_lines:
                if return_if_is_kw(line):
                    kw = return_if_is_kw(line)
                    if kw == "description":
                        if "description" in data_frame:
                            is_dirty_sample = True
                            print(f"description kw already in data_frame, data_frame:{data_frame}")
                            continue
                        data_frame[kw] = ""
                    elif kw == "dialogue":
                        if "dialogue" in data_frame:
                            is_dirty_sample = True
                            print(f"dialogue kw already in data_frame, data_frame:{data_frame}")
                            continue
                        data_frame[kw] = []
                    else:
                        if "dialogue" not in data_frame:
                            is_dirty_sample = True
                            print(f"dialogue kw already in data_frame, data_frame:{data_frame}")
                            continue
                        data_frame["dialogue"].append([kw, ""])
                else:
                    if kw is None: # id
                        continue
                    elif kw == "description":
                        data_frame[kw] += line if not data_frame[kw] else f" {line}"
                    elif kw == "dialogue":
                        raise ValueError(f"This is not expected, line: {line}.")
                    else:
                        data_frame["dialogue"][-1][1] += line \
                            if len(data_frame["dialogue"][-1][1]) == 0 else f' {SEP_TOKEN} {line}'
            if is_dirty_sample:
                dirty_sample_count += 1
                continue
            data["ids"].append(id)
            data_frame["id"] = id
            data["data_frames"].append(data_frame)
        print(f"There are {dirty_sample_count} dirty data from {len(id_content_records)} in total")
        return data

def split_data(data_frames):
    np.random.shuffle(data_frames)
    # ratio: 9/0.5/0.5
    total_id_num = len(data_frames)
    validate_idx = int(float(total_id_num) * 9 / 10)
    test_idx = int(float(total_id_num) * 9.5 / 10)

    train_data = data_frames[:validate_idx]
    val_data = data_frames[validate_idx:test_idx]
    test_data = data_frames[test_idx:]
    return train_data, val_data, test_data

def rm_extra_spaces(line: str):
    return " ".join(line.strip().split())

def make_dataset(data_frames: List[dict], out_dir: Path, mode:str= "train",
                 max_src_len:int=500, max_tgt_len:int=200):
    # data_frames:[{id:xxx, Description:xxx, Dialogue:xxx}]
    src_file = out_dir / f"{mode}.source.txt"
    tgt_file = out_dir / f"{mode}.target.txt"

    with src_file.open("w", encoding="utf-8") as src_fw, \
        tgt_file.open("w", encoding="utf-8") as tgt_fw:
        for one in data_frames:
            if 'dialogue' not in one:
                continue
            diag_list = one['dialogue']
            # rm_extra_spaces
            for idx in range(len(diag_list)):
                diag_list[idx][1] = diag_list[idx][1].replace("in brief:", "")
                diag_list[idx][1] = rm_extra_spaces(diag_list[idx][1])
            if len(diag_list) % 2 != 0:
                # dirty data
                diag_list = diag_list[:-1]

            # process data ----
            if len(diag_list) == 2:
                assert diag_list[0][0] == 'patient:'
                assert diag_list[1][0] == 'doctor:'
                input_str = f"patient: {diag_list[0][1]}\n"
                output_str = f"{diag_list[1][1]}\n"
            elif len(diag_list) > 2:
                for doctor_idx in range(1, len(diag_list), 2):
                    # 防止对话过长
                    diag_history = " ".join([f"{diag[0]}: {diag[1]}" for diag in diag_list[:doctor_idx]])
                    input_str = f"patient: {diag_history[-1000:].strip()}\n"
                    output_str = f"{diag_list[doctor_idx][1].strip()}\n"
            else:
                raise ValueError("Unexpected.")
            if len(input_str.strip()) == 0 or len(output_str.strip()) == 0:
                continue
            # rm too long dialogues
            if len(input_str.strip().split()) > max_src_len or len(output_str.strip().split()) > max_tgt_len:
                continue
            # write to files
            src_fw.write(input_str)
            tgt_fw.write(output_str)
    # validate the generated files:
    with src_file.open("r", encoding="utf-8") as src_fr, \
        tgt_file.open("r", encoding="utf-8") as tgt_fr:
        src_lines = src_fr.readlines()
        tgt_lines = tgt_fr.readlines()
        assert len(src_lines) == len(tgt_lines)
        dialogue_count = len(src_lines)
        word_count = sum([len(line.strip().split()) for line in src_lines]) + \
                     sum([len(line.strip().split()) for line in tgt_lines])
        print(f"Finally generate data with {dialogue_count} lines; {word_count} words")

if __name__ == '__main__':
    data = load_dataset(dataset_file=f"{BASE_DIR}/resources/med-dialog/covid_english.txt")
    train_data, val_data, test_data = split_data(data['data_frames'])

    output_dir = Path(f"{BASE_DIR}/datasets/med-dialog/english-dialog")
    os.makedirs(output_dir, exist_ok=True)
    make_dataset(data_frames=train_data, out_dir=output_dir, mode="train")
    make_dataset(data_frames=val_data, out_dir=output_dir, mode="val")
    make_dataset(data_frames=test_data, out_dir=output_dir, mode="test")
