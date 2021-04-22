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

from tqdm import tqdm
from data import CMVDataset
from utils import create_mini_batch, load_torch_model, get_logits, compute_acc, get_answers, divide, compute_mrr, EarlyStopping
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
    if not os.path.exists(config.train_file): divide(config)
    tokenizer = BertTokenizer.from_pretrained(config.model_version)
    train_set = CMVDataset(path=config.train_file, mode='train', tokenizer=tokenizer)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, collate_fn=create_mini_batch)
    valid_set = CMVDataset(path=config.valid_file, mode='test', tokenizer=tokenizer)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, collate_fn=create_mini_batch)

    # 2. Build model
    model = BaselineModel(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    early_stopping = EarlyStopping(config.patience, path=os.path.join(config.model_path, 'model.bin'))

    # 3. Train and Save model
    running_loss = []
    label = [1] * len(pd.read_csv(config.valid_file, header=None, sep='#'))
    label_pred = []
    for epoch in range(15):
        model.train()
        temp = 0.0
        for data in tqdm(train_loader):
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
            optimizer.zero_grad()
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors,
                            labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            temp += loss.item()
        running_loss.append(temp)
        print('[epoch %d] loss: %.3f' % (epoch + 1, running_loss[epoch]))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for data in tqdm(valid_loader):
                tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
                optimizer.zero_grad()
                outputs = model(input_ids=tokens_tensors,
                                token_type_ids=segments_tensors,
                                attention_mask=masks_tensors,
                                labels=labels)
                loss = outputs[0]
                val_loss += loss.item()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            # early stopping criterion satisfied
            break


if __name__ == '__main__':
    main()