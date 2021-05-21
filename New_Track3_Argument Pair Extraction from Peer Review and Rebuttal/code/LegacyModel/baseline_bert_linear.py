import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer
from termcolor import colored

from module.rnn import BiLSTMEncoder
from module.inferencer import LinearCRF
from module.embedder import Embedder, TokenEmbedder

from data import START_TAG, STOP_TAG, label2idx, idx2labels
from typing import Dict, List, Tuple, Any

class BaselineBertModel(nn.Module):
    
    def __init__(self, args, bertModel):
        super(BaselineBertModel, self).__init__()
        if args.token_embedding:
            self.embedder = TokenEmbedder(bertModel, args)
        else:
            self.embedder = Embedder(bertModel)
        self.type_embedder = nn.Embedding(2, args.embedding_dim)
        self.lstm_encoder = BiLSTMEncoder(args.bert_feature_dim + args.embedding_dim, args.hidden_dim)
        self.linear_crf = LinearCRF(label2idx, idx2labels, START_TAG=START_TAG, STOP_TAG=STOP_TAG)
        classifier = [nn.Linear(args.bert_feature_dim * 2 + args.embedding_dim * 2, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, args.class_num)]
        self.hidden2tag = nn.Sequential(*classifier)
        self.hidden2biotag = nn.Linear(args.hidden_dim, args.bio_class_num)
        info = f"""[model info] model: BaselineBertModel
                BERT model: {args.bert_model_path} 
                bert_feature_dim: {args.bert_feature_dim} 
                type_embedding_dim: {args.embedding_dim}
                hidden_dim: {args.hidden_dim}"""
        print(colored(info, 'yellow'))
    
    def forward(self, embedder_input, input_mask, seq_lens, bio_tags, type_idx):
        
        bert_embedding = self.embedder(*embedder_input)
        type_embedding = self.type_embedder(type_idx)
        bert_embedding = torch.cat((bert_embedding, type_embedding), dim=-1)
        lstm_embedding = self.lstm_encoder(bert_embedding, seq_lens)
        crf_input = self.hidden2biotag(lstm_embedding)
        crf_loss = self.linear_crf(crf_input, seq_lens, bio_tags, input_mask)
        _, num_sents, _ = lstm_embedding.size()
        expanded_bert_embedding = bert_embedding.unsqueeze(2).expand([-1, -1, num_sents, -1])
        expanded_bert_embedding_t = bert_embedding.unsqueeze(1).expand([-1, num_sents, -1, -1])
        cross_feature = torch.cat((expanded_bert_embedding, expanded_bert_embedding_t), dim=3)
        pair_output = self.hidden2tag(cross_feature)

        return pair_output, crf_loss
    
    def decode(self, embedder_input, input_mask, seq_lens, type_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        """
        bert_embedding = self.embedder(*embedder_input)
        type_embedding = self.type_embedder(type_idx)
        bert_embedding = torch.cat((bert_embedding, type_embedding), dim=-1)
        lstm_embedding = self.lstm_encoder(bert_embedding, seq_lens)
        crf_input = self.hidden2biotag(lstm_embedding)
        best_scores, decode_idx = self.linear_crf.decode(crf_input, seq_lens)
        _, num_sents, _ = lstm_embedding.size()
        expanded_bert_embedding = bert_embedding.unsqueeze(2).expand([-1, -1, num_sents, -1])
        expanded_bert_embedding_t = bert_embedding.unsqueeze(1).expand([-1, num_sents, -1, -1])
        cross_feature = torch.cat((expanded_bert_embedding, expanded_bert_embedding_t), dim=3)
        pair_output = self.hidden2tag(cross_feature)       
        
        return pair_output, best_scores, decode_idx
