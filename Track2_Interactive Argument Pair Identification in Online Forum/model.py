import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig
from torch.utils.data.distributed import DistributedSampler

class BaselineModel(torch.nn.Module):
    def __init__(self, config):
        """Initialize the model with config dict.
        Args:
            config: python dict must contains the attributes below:
                config.model_version: pretrained model path or model type
                    e.g. 'bert-base-uncased'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super(BaselineModel, self).__init__()
        self.model_version = config['model_version']
        self.num_labels = config['num_labels']
        self.output_hidden_states = config['output_hidden_states']
        self.dropout = config['dropout']
        self.hidden_size = config['hidden_size']
        self.bert = BertForSequenceClassification.from_pretrained(
            self.model_version, num_labels = self.num_labels, hidden_size = self.hidden_size,
            output_hidden_states = self.output_hidden_states)

    def forward(self, tokens_tensors, segments_tensors, masks_tensors, labels=None):
        """Forward inputs and get logits.
                Args:
                    input_ids: (batch_size, max_seq_len)
                    attention_mask: (batch_size, max_seq_len)
                    token_type_ids: (batch_size, max_seq_len)
                Returns:
                    logits: (batch_size, num_classes)
                """
        outputs = self.bert(input_ids=tokens_tensors,
                        token_type_ids=segments_tensors,
                        attention_mask=masks_tensors,
                        labels=labels)
        return outputs