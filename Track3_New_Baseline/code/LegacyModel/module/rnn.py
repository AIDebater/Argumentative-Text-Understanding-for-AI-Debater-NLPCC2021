import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.jit as jit
from typing import Optional

class BiLSTMEncoder(nn.Module):
    """
    BILSTM encoder
    """
    def __init__(self, input_dim, hidden_dim, drop_lstm=0.5, num_lstm_layers=1):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim//2, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(drop_lstm)

    def forward(self, sent_rep: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        param:
        sent_rep: (batch_size, num_sents, emb_size)
        seq_lens: (batch_size, )
        return: 
        feature_out: (batch_size, num_sents, hidden_dim)
        """
        sorted_seq_len, permIdx = seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = sent_rep[permIdx]

        packed_sents = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
        lstm_out, _ = self.lstm(packed_sents, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        feature_out = self.drop_lstm(lstm_out)

        return feature_out[recover_idx]
