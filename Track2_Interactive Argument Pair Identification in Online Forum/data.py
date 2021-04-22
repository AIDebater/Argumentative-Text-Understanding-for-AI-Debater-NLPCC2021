import torch
from torch.utils.data import Dataset

import pandas as pd

class CMVDataset(Dataset):
    def __init__(self, path: str, mode: str, tokenizer):
        assert mode in ["train", "test"]
        self.df = pd.read_csv(path, sep='#', header=None)
        self.mode = mode
        self.size = len(self.df)
        self.tokenizer = tokenizer
        # self.tags = 'ABCDE'
        # self.ans2tag = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
        self.max_encode_len = 254

    def __getitem__(self, idx):
        record = self.df.iloc[idx, :]
        tokens_tensor = []
        segments_tensor = []
        if self.mode == 'train':
            labels_tensor = [torch.tensor(1)] + [torch.tensor(0)] * 4
        else:
            labels_tensor = None
        for i in range(5):
            text_a = record[1]
            text_b = record[2 + 2 * i]
            sentence_tokens_tensor, sentence_segments_tensor = self.convert(text_a, text_b)
            tokens_tensor.append(sentence_tokens_tensor)
            segments_tensor.append(sentence_segments_tensor)

        return tokens_tensor, segments_tensor, labels_tensor

    def convert(self, text_a, text_b):
        # convert text_a and text_b to bert input
        word_pieces = ["[CLS]"] + self.tokenizer.tokenize(text_a)[: self.max_encode_len] + ["[SEP]"]
        len_a = len(word_pieces)
        word_pieces += self.tokenizer.tokenize(text_b)[: self.max_encode_len] + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        tokens_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(word_pieces))
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)
        return tokens_tensor, segments_tensor

    def __len__(self):
        return self.size