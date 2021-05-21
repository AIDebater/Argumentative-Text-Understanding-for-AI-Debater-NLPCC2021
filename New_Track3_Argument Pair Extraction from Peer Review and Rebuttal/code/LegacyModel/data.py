import math
import torch
import numpy as np
from transformers import *
from utils import context_models

START_TAG = 'START'
STOP_TAG = 'STOP'
PAD_TAG = 'PAD'
sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}
biotags2id = {'O': 0, 'B': 1, 'I': 2}
label2idx = {'O': 0, 'B': 1, 'I': 2, START_TAG: 3, STOP_TAG: 4, PAD_TAG: 5}
idx2labels = ['O', 'B', 'I', START_TAG, STOP_TAG, PAD_TAG]

def get_spans(tags):
    """
    for spans
    """
    tags = tags.strip().split('<tag>')
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans

class Instance(object):
    def __init__(self, tokenizer, sentence_pack, args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.last_review = sentence_pack['split_idx']
        self.tokens = self.sentence.strip().split(' <sentsep> ')
        self.sen_length = len(self.tokens)
        self.bert_tokens = []
        self.num_tokens = []
        for i, sent in enumerate(self.tokens):
            word_tokens = tokenizer.tokenize(" " + sent)
            input_ids = tokenizer.convert_tokens_to_ids(
                [tokenizer.cls_token] + word_tokens + [tokenizer.sep_token])
            self.bert_tokens.append(input_ids)
            self.num_tokens.append(min(len(word_tokens), args.max_bert_token-1))
        # self.bert_tokens = tokenizer.encode(self.sentence)
        self.length = len(self.bert_tokens)
        self.token_range = [[i, i] for i in range(self.length)]
        self.tags = torch.zeros(self.length, self.length).long()
        self.bio = torch.zeros(self.length).long()
        self.type_idx = torch.zeros(self.length).long()
        self.type_idx[self.last_review+1:self.length] = 1

        self.bio[:] = label2idx['O']
        if args.cls_method == 'binary':
            self.tags[:self.last_review+1, :self.last_review+1] = -1
            self.tags[self.last_review+1:, self.last_review+1:] = -1
            self.tags[self.last_review+1:, :self.last_review+1] = -1

        for triple in sentence_pack['triples']:
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            for l, r in aspect_span:
                for i in range(l, r+1):
                    if i == l:
                        self.bio[i] = biotags2id['B']
                    else:
                        self.bio[i] = biotags2id['I']
                    if args.cls_method == 'multiclass':
                        for j in range(l, r+1):
                            self.tags[i][j] = 1

            for l, r in opinion_span:
                for i in range(l, r+1):
                    if i == l:
                        self.bio[i] = biotags2id['B']
                    else:
                        self.bio[i] = biotags2id['I']
                    if args.cls_method == 'multiclass':
                        for j in range(l, r+1):
                            self.tags[i][j] = 2

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            if args.task == 'pair':
                                if args.cls_method == 'binary':
                                    self.tags[i][j] = 1
                                    # self.tags[j][i] = 1
                                else:
                                    self.tags[i][j] = 3
                                    # self.tags[j][i] = 3
                            elif args.task == 'triplet':
                                    self.tags[i][j] = sentiment2id[triple['sentiment']]


def load_data_instances(sentence_packs, args):
    instances = list()
    tokenizer = context_models[args.bert_tokenizer_path]['tokenizer'].from_pretrained(args.bert_tokenizer_path)
    if args.num_instances != -1:
        for sentence_pack in sentence_packs[:args.num_instances]:
            instances.append(Instance(tokenizer, sentence_pack, args))
    else:
        for sentence_pack in sentence_packs:
            instances.append(Instance(tokenizer, sentence_pack, args))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)
        self.max_bert_token = args.max_bert_token

    def get_batch(self, index):
        sentence_ids = []
        sentences = []
        sens_lens = []
        lengths = []
        last_review_indice = []

        batch_size = min((index + 1) * self.args.batch_size, len(self.instances)) - index * self.args.batch_size
        max_num_sents = max([self.instances[i].length for i in range(index * self.args.batch_size,
                            min((index + 1) * self.args.batch_size, len(self.instances)))])
        max_sent_length = min(max([max(map(len, self.instances[i].bert_tokens)) for i in range(index * self.args.batch_size,
                            min((index + 1) * self.args.batch_size, len(self.instances)))]), self.max_bert_token)

        bert_tokens = torch.zeros(batch_size, max_num_sents, max_sent_length, dtype=torch.long)
        attn_masks = torch.zeros(batch_size, max_num_sents, max_sent_length, dtype=torch.long)
        masks = torch.zeros(batch_size, max_num_sents, dtype=torch.long)
        tags = -torch.ones(batch_size, max_num_sents, max_num_sents).long()
        biotags = torch.full((batch_size, max_num_sents), label2idx[PAD_TAG]).long()
        type_idx = torch.zeros(batch_size, max_num_sents).long()
        num_tokens = torch.ones(batch_size, max_num_sents).long()

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            lengths.append(self.instances[i].length)
            last_review_indice.append(self.instances[i].last_review)
            masks[i-index * self.args.batch_size, :self.instances[i].length] = 1
            tags[i-index * self.args.batch_size, :self.instances[i].length, :self.instances[i].length] = self.instances[i].tags
            biotags[i-index * self.args.batch_size, :self.instances[i].length] = self.instances[i].bio
            type_idx[i-index * self.args.batch_size, :self.instances[i].length] = self.instances[i].type_idx
            num_tokens[i-index * self.args.batch_size, :self.instances[i].length] = torch.LongTensor(self.instances[i].num_tokens)

            for j in range(self.instances[i].length):
                length_filled = min(self.max_bert_token, len(self.instances[i].bert_tokens[j]))
                bert_tokens[i-index * self.args.batch_size, j, :length_filled] = \
                    torch.LongTensor(self.instances[i].bert_tokens[j][:length_filled])
                attn_masks[i-index * self.args.batch_size, j, :length_filled] = 1

        bert_tokens = bert_tokens.to(self.args.device)
        attn_masks = attn_masks.to(self.args.device)
        tags = tags.to(self.args.device)
        biotags = biotags.to(self.args.device)
        lengths = torch.tensor(lengths).to(self.args.device)
        masks = masks.to(self.args.device)
        type_idx = type_idx.to(self.args.device)

        if self.args.token_embedding:
            return sentence_ids, (bert_tokens, attn_masks, num_tokens), lengths, sens_lens, tags, biotags, masks, last_review_indice, type_idx
        else:
            return sentence_ids, (bert_tokens, attn_masks), lengths, sens_lens, tags, biotags, masks, last_review_indice, type_idx
