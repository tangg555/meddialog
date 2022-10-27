"""
@Desc:
@Reference:
corpus is download from https://github.com/UCSD-AI4H/Medical-Dialogue-System

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



if __name__ == '__main__':
    corpus_dir = Path(f"{BASE_DIR}/resources/med-dialog/large-english-dialog-corpus")
    data_frames = []
    for file_name in os.listdir(corpus_dir):
        file_path = corpus_dir / file_name
        data = load_dataset(file_path)
        data_frames += data["data_frames"]
        print(f"load data from {file_name}, loaded data: {len(data['data_frames'])}, current data_frames: {len(data_frames)}")

    output_dir = Path(f"{BASE_DIR}/datasets/med-dialog/large-english-dialog-corpus")
    os.makedirs(output_dir, exist_ok=True)
    train_data, val_data, test_data = split_data(data_frames)
    make_dataset(data_frames=train_data, out_dir=output_dir, mode="train")
    make_dataset(data_frames=val_data, out_dir=output_dir, mode="val")
    make_dataset(data_frames=test_data, out_dir=output_dir, mode="test")


