"""
@Desc:
@Reference:
https://github.com/UCSD-AI4H/COVID-Dialogue/blob/master/src/preprocess.py
@Notes:
"""
import torch
import os
import codecs
import sys
import json
from pathlib import Path
import json
from typing import List

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path


def return_if_is_id(line):
    kw_len = len("id=")
    if line[:kw_len] == "id=":
        line = line.strip()
        return int(line[kw_len:])
    return False

def return_if_is_kw(line):
    kw_list = ["id=", "Description", "Dialogue", "Patient", "Doctor"]
    for kw in kw_list:
        kw_len = len(kw)
        if line[:kw_len] == kw:
            return line[:kw_len]
    return False

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
            line = line.strip()
            if not line:
                continue
            if return_if_is_id(line):
                id = return_if_is_id(line)
                if id in id_content_records: # there are two "id=18"
                    id += 1
                assert id not in id_content_records
                id_content_records[id] = []
                # ---- validation -------
                if pre_id is not None:
                    assert pre_id == id - 1
                pre_id = id
            else:
                id_content_records[id].append(line)

        for id, id_content_lines in id_content_records.items():
            assert id not in data["ids"]
            data["ids"].append(id)
            data_frame = {"id": id}
            # ---- parse content ----
            kw = None
            for line in id_content_lines:
                if return_if_is_kw(line):
                    kw = return_if_is_kw(line)
                    if kw == "Description":
                        assert "Description" not in data_frame
                        data_frame[kw] = ""
                    elif kw == "Dialogue":
                        assert "Dialogue" not in data_frame
                        data_frame[kw] = []
                    else:
                        assert "Dialogue" in data_frame
                        data_frame["Dialogue"].append([kw, ""])
                else:
                    if kw is None: # id
                        continue
                    elif kw == "Description":
                        data_frame[kw] += line if not data_frame[kw] else f" {line}"
                    elif kw == "Dialogue":
                        raise ValueError(f"This is not expected, line: {line}.")
                    else:
                        data_frame["Dialogue"][-1][1] += line if not data_frame["Dialogue"][-1][1] else f" {line}"

            data["data_frames"].append(data_frame)
        return data

def split_data(data):
    data_frames = data["data_frames"]
    total_id_num = len(data_frames)
    validate_idx = int(float(total_id_num) * 8 / 10)
    test_idx = int(float(total_id_num) * 9 / 10)

    train_data = data_frames[:validate_idx]
    val_data = data_frames[validate_idx:test_idx]
    test_data = data_frames[test_idx:]
    return train_data, val_data, test_data

def make_dataset(data: List[dict], out_dir: Path, mode:str="train"):
    # data: [{id:xxx, Description:xxx, Dialogue:xxx}]
    src_file = out_dir / f"{mode}.source.txt"
    tgt_file = out_dir / f"{mode}.target.txt"
    with src_file.open("w", encoding="utf-8") as src_fw, \
        tgt_file.open("w", encoding="utf-8") as tgt_fw:
        for one in data:
            diag_list = one['Dialogue']
            if len(diag_list) % 2 != 0:
                # dirty data
                diag_list = diag_list[:-1]
            if len(diag_list) == 2:
                assert diag_list[0][0] == 'Patient'
                assert diag_list[1][0] == 'Doctor'
                src_fw.write(f"Patient: {diag_list[0][1]}\n")
                tgt_fw.write(f"{diag_list[1][1]}\n")
            elif len(diag_list) > 2:
                for doctor_idx in range(1, len(diag_list), 2):
                    diag_history = " ".join([f"{diag[0]}: {diag[1]}" for diag in diag_list[:doctor_idx]])
                    src_fw.write(f"{diag_history}\n")
                    tgt_fw.write(f"{diag_list[doctor_idx][1]}\n")
            else:
                raise ValueError("Unexpected.")

if __name__ == '__main__':
    data = load_dataset(dataset_file=f"{BASE_DIR}/resources/med-dialog/covid_english.txt")
    train_data, val_data, test_data = split_data(data)

    output_dir = Path(f"{BASE_DIR}/datasets/med-dialog/english-dialog")
    os.makedirs(output_dir, exist_ok=True)
    make_dataset(data=train_data, out_dir=output_dir, mode="train")
    make_dataset(data=val_data, out_dir=output_dir, mode="val")
    make_dataset(data=test_data, out_dir=output_dir, mode="test")
