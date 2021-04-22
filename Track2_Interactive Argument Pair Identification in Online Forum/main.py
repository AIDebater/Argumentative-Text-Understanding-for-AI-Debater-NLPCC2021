from typing import Dict
import argparse
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils import get_path
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data import CMVDataset
from utils import create_mini_batch, load_torch_model, get_logits, compute_acc, get_answers
from model import BaselineModel
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig
from torch.utils.data.distributed import DistributedSampler


def main(config_file='config.json'):
    """Main method for training.
    Args:
        config_file: in config dir
    """
    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    get_path(os.path.join(config.model_path, config.experiment_name))
    get_path(config.log_path)

    # 1. Load data
    tokenizer = BertTokenizer.from_pretrained(config.model_version)
    # train_set = CMVDataset(path="train.txt", mode='train', tokenizer=tokenizer)
    # train_loader = DataLoader(train_set, batch_size=10, shuffle=True, collate_fn=create_mini_batch)
    data_set = CMVDataset(path=config.test_file_path, mode='test', tokenizer=tokenizer)
    data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False, collate_fn=create_mini_batch)

    # 2. Load model
    model = BaselineModel(config)
    # model = load_torch_model(
    #     model, model_path=os.path.join(config.model_path, 'model.bin'))
    model.load_state_dict(torch.load(os.path.join(config.model_path, 'model.bin')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 3. Evaluate
    logits = get_logits(data_loader=data_loader, tokenizer=tokenizer, device=device, model=model)
    label = [1] * len(pd.read_csv(config.test_file_path, header=None, sep='#'))
    acc = compute_acc(label=label, logits=logits)
    print(f"Model's Acc:{acc}")

    # 4. Output submission file
    answers = get_answers(logits)
    with open(config.output_file, 'w') as fout:
        fout.write('id,answer\n')
        for i, j in enumerate(answers):
            fout.write(str(i) + ',' + str(j) + '\n')


if __name__ == '__main__':
    main()