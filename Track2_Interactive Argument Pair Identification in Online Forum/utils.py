import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data import CMVDataset
from collections import OrderedDict

import pandas as pd
import random
import numpy as np
import os
from model import BaselineModel
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig
from torch.utils.data.distributed import DistributedSampler

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = True
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def create_mini_batch(samples):
    # create a batch tensor
    tokens_tensors = sum([sample[0] for sample in samples], [])
    segments_tensors = sum([sample[1] for sample in samples], [])
    if samples[0][2] is not None:
        labels_tensors = torch.stack(sum([sample[2] for sample in samples], []))
    else:
        labels_tensors = None
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    # create mask tensors
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    return tokens_tensors, segments_tensors, masks_tensors, labels_tensors

def get_logits(data_loader: DataLoader, tokenizer: BertTokenizer, device: str, model: BaselineModel) -> list:
    # Get logits from the trained model
    # data_set = CMVDataset(path=path, mode='test', tokenizer=tokenizer)
    # data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False, collate_fn=create_mini_batch)
    logits = []
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data = [t.to(device) for t in data if t is not None]
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(tokens_tensors=tokens_tensors, segments_tensors=segments_tensors, masks_tensors=masks_tensors)
            logits.append(outputs[0][:, 1].tolist())
    return logits

def divide(config):
    f = open(config['train_file_path'], mode='r', encoding='utf-8')
    train = open(config['train_file'], mode='w', encoding='utf-8')
    valid = open(config['valid_file'], mode='w', encoding='utf-8')
    lines = list(f.readlines())
    train_num = int(0.8 * len(lines))
    train.writelines(lines[:train_num])
    valid.writelines(lines[train_num:])


def get_path(path):
    """Create the path if it does not exist.
    Args:
        path: path to be used
    Returns:
        Existed path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_torch_model(model, model_path):
    """Load state dict to model.
    Args:
        model: model to be loaded
        model_path: state dict file path
    Returns:
        loaded model
    """
    pretrained_model_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, value in pretrained_model_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict, strict=True)
    return model

def compute_acc(label: list, logits: list) -> float:
    # Calculate the Accuracy according to the given output logits.
    count = 0
    for i in range(len(label)):
        if label[i] == np.array(logits[i]).argmax() + 1:
            count += 1
    return count / len(label)

def get_answers(logits: list) -> list:
    answers = []
    for i in range(len(logits)):
        answers.append(np.array(logits[i]).argmax() + 1)
    return answers


def compute_mrr(label: list, logits: list) -> float:
    # Calculate the MRR according to the given output logits.
    count = 0
    for i in range(len(label)):
        order = np.array(logits[i]).argsort()[::-1]
        for j in range(len(order)):
            if order[j] == label[i] - 1:
                count += 1.0/(j+1)
    return count / len(label)