import multiprocessing
import pickle
import numpy as np
import sklearn
from typing import List, Tuple, Any
from transformers import *
import torch.nn as nn
from termcolor import colored
import os
from collections import defaultdict
import itertools

context_models = {
    'bert-base-uncased' : {"model": BertModel, "tokenizer" : BertTokenizer },
    'bert-base-cased' : {"model": BertModel, "tokenizer" : BertTokenizer },
    'bert-large-cased' : {"model": BertModel, "tokenizer" : BertTokenizer },
    'bert-base-chinese' : {"model": BertModel, "tokenizer" : BertTokenizer },
    'openai-gpt': {"model": OpenAIGPTModel, "tokenizer": OpenAIGPTTokenizer},
    'gpt2': {"model": GPT2Model, "tokenizer": GPT2Tokenizer},
    'ctrl': {"model": CTRLModel, "tokenizer": CTRLTokenizer},
    'transfo-xl-wt103': {"model": TransfoXLModel, "tokenizer": TransfoXLTokenizer},
    'xlnet-base-cased': {"model": XLNetModel, "tokenizer": XLNetTokenizer},
    'xlm-mlm-enfr-1024': {"model": XLMModel, "tokenizer": XLMTokenizer},
    'distilbert-base-cased': {"model": DistilBertModel, "tokenizer": DistilBertTokenizer},
    'roberta-base': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'roberta-large': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'xlm-roberta-base': {"model": XLMRobertaModel, "tokenizer": XLMRobertaTokenizer},
}

class Metric():
    def __init__(self, args, predictions, goldens, bert_lengths, sen_lengths, last_review_indice, golden_bio, pred_bio):
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.data_num = len(self.predictions)
        self.last_review_indice = last_review_indice
        self.golden_bio = golden_bio
        self.pred_bio = [pred_seq[::-1] for pred_seq in pred_bio]

    def get_aspect_spans(self, biotags, last_review_idx): # review
        spans = []
        start = -1
        for i in range(last_review_idx+1):
            if biotags[i] == 1:
                start = i
                if i == last_review_idx:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, start])
            elif biotags[i] == 2:
                if i == last_review_idx:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, i])
        return spans

    def get_opinion_spans(self, biotags, length, last_review_idx): # rebuttal
        spans = []
        start = -1
        for i in range(last_review_idx+1, length):
            if biotags[i] == 1:
                start = i
                if i == length-1:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, start])
            elif biotags[i] == 2:
                if i == length-1:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, i])
        return spans

    def find_pair(self, tags, aspect_spans, opinion_spans):
        pairs = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * self.args.class_num
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        tag_num[int(tags[i][j])] += 1
                        # tag_num[int(tags[j][i])] += 1
                if tag_num[self.args.class_num-1] < 1*(ar-al+1)*(pr-pl+1)*self.args.pair_threshold: continue
                sentiment = -1
                pairs.append([al, ar, pl, pr, sentiment])
        return pairs

    def score_aspect(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = self.get_aspect_spans(self.golden_bio[i], self.last_review_indice[i])
            for spans in golden_aspect_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_aspect_spans = self.get_aspect_spans(self.pred_bio[i], self.last_review_indice[i])
            for spans in predicted_aspect_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        # precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        # recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # return precision, recall, f1
        return correct_num, len(predicted_set), len(golden_set)

    def score_opinion(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_opinion_spans = self.get_opinion_spans(self.golden_bio[i], self.sen_lengths[i], self.last_review_indice[i])
            for spans in golden_opinion_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_opinion_spans = self.get_opinion_spans(self.pred_bio[i], self.sen_lengths[i], self.last_review_indice[i])
            for spans in predicted_opinion_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        # precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        # recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # return precision, recall, f1
        return correct_num, len(predicted_set), len(golden_set)

    def score_bio(self, aspect, opinion):
        correct_num = aspect[0] + opinion[0]
        pred_num = aspect[1] + opinion[1]
        gold_num = aspect[2] + opinion[2]
        precision = correct_num / pred_num * 100 if pred_num > 0 else 0
        recall = correct_num / gold_num * 100 if gold_num > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_pair(self):
        # self.all_labels: (batch_size, num_sents, num_sents)
        all_labels = [k for i in range(self.data_num) for j in self.goldens[i] for k in j]
        all_preds = [k for i in range(self.data_num) for j in self.predictions[i] for k in j]
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        for i in range(len(all_labels)):
            if all_labels[i] != -1:
                if all_labels[i] == 1 and all_preds[i] == 1:
                    tp += 1
                elif all_labels[i] == 1 and all_preds[i] == 0:
                    fn += 1
                elif all_labels[i] == 0 and all_preds[i] == 1:
                    fp += 1
                elif all_labels[i] == 0 and all_preds[i] == 0:
                    tn += 1
        precision = 1.0 * tp / (tp + fp) * 100 if tp + fp != 0 else 0
        recall = 1.0 * tp / (tp + fn) * 100 if tp + fn != 0 else 0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return precision, recall, f1

    def score_uniontags(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = self.get_aspect_spans(self.golden_bio[i], self.last_review_indice[i])
            golden_opinion_spans = self.get_opinion_spans(self.golden_bio[i], self.sen_lengths[i], self.last_review_indice[i])
            # print(golden_aspect_spans)
            # print(golden_opinion_spans)
            golden_tuples = self.find_pair(self.goldens[i], golden_aspect_spans, golden_opinion_spans)
            # print(golden_tuples)
            for pair in golden_tuples:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))

            predicted_aspect_spans = self.get_aspect_spans(self.pred_bio[i], self.last_review_indice[i])
            predicted_opinion_spans = self.get_opinion_spans(self.pred_bio[i], self.sen_lengths[i], self.last_review_indice[i])
            predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans)
            for pair in predicted_tuples:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))

        # print('gold: ', golden_set)
        # print('pred: ', predicted_set)

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) * 100 if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) * 100 if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

def get_huggingface_optimizer_and_scheduler(args, model: nn.Module,
                                            num_training_steps: int,
                                            weight_decay: float = 0.0,
                                            eps: float = 1e-8,
                                            warmup_step: int = 0):
    """
    Copying the optimizer code from HuggingFace.
    """
    print(colored(f"Using AdamW optimizer by HuggingFace with {args.lr} learning rate, "
                  f"eps: {eps}, weight decay: {weight_decay}, warmup_step: {warmup_step}, ", 'yellow'))
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )
    return optimizer, scheduler

class DisjointSet:
    
    def __init__(self, size):
        self.size_ = size
        self.size = [1]*size
        self.connection = list(range(size))
    
    def root(self, a):
        if self.connection[a] == a:
            return a
        else:
            return self.root(self.connection[a])
    
    def find(self, a, b):
        return self.root(a) == self.root(b)
    
    def union(self, a, b):
        if self.size[a] > self.size[b]:
            self.size[self.root(a)] += self.size[self.root(b)]
            self.connection[self.root(b)] = self.root(a)
        else:
            self.size[self.root(b)] += self.size[self.root(a)]
            self.connection[self.root(a)] = self.root(b)
    
    def unify_(self):
        for i in range(self.size_):
            root_ = self.root(self.connection[i])
            self.connection[i] = root_
    
    def cluster(self, paired_spans):
        self.unify_()
        dic = defaultdict(list)
        for i in range(self.size_):
            # the key is just a dummy key since list is not hashable
            dic[self.connection[i]].append(paired_spans[i])
        return dic

class Writer():
    """
    output test dataset results to file
    """
    def __init__(self, args, predictions, goldens, bert_lengths, sen_lengths, last_review_indice, golden_bio, pred_bio):
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.data_num = len(self.predictions)
        self.last_review_indice = last_review_indice
        self.golden_bio = golden_bio
        self.pred_bio = [pred_seq[::-1] for pred_seq in pred_bio]
        self.output_dir = os.path.join(args.output_dir, args.model_dir.strip('/'), args.model_dir.strip('/') + '.txt')

    def get_review_spans(self, biotags, last_review_idx):
        spans = []
        start = -1
        for i in range(last_review_idx+1):
            if biotags[i] == 1:
                start = i
                if i == last_review_idx:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, start])
            elif biotags[i] == 2:
                if i == last_review_idx:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, i])
        return spans

    def get_rebuttal_spans(self, biotags, length, last_review_idx):
        spans = []
        start = -1
        for i in range(last_review_idx+1, length):
            if biotags[i] == 1:
                start = i
                if i == length-1:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, start])
            elif biotags[i] == 2:
                if i == length-1:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, i])
        return spans

    def find_pair(self, tags, review_spans, reply_spans):
        pairs = []
        for al, ar in review_spans:
            for pl, pr in reply_spans:
                tag_num = [0] * self.args.class_num
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        tag_num[int(tags[i][j])] += 1
                        # tag_num[int(tags[j][i])] += 1
                if tag_num[self.args.class_num-1] < 1*(ar-al+1)*(pr-pl+1)*self.args.pair_threshold: continue
                pairs.append([al, ar, pl, pr])
        return pairs
    
    def output_results(self):
        with open(self.output_dir, 'w') as f:
            # f.write('\t'.join(['golden', 'pred']) + '\n')
            for i in range(self.data_num):
                golden_review_spans = self.get_review_spans(self.golden_bio[i], self.last_review_indice[i])
                # review_golden = '|'.join(map(lambda span: '-'.join(map(str, span)), golden_review_spans))

                predicted_review_spans = self.get_review_spans(self.pred_bio[i], self.last_review_indice[i])
                # review_pred = '|'.join(map(lambda span: '-'.join(map(str, span)), predicted_review_spans))
                
                golden_reply_spans = self.get_rebuttal_spans(self.golden_bio[i], self.sen_lengths[i], self.last_review_indice[i])
                # reply_golden = '|'.join(map(lambda span: '-'.join(map(str, span)), golden_reply_spans))

                predicted_reply_spans = self.get_rebuttal_spans(self.pred_bio[i], self.sen_lengths[i], self.last_review_indice[i])
                # reply_pred = '|'.join(map(lambda span: '-'.join(map(str, span)), predicted_reply_spans))
                
                golden_pairs = self.find_pair(self.goldens[i], golden_review_spans, golden_reply_spans)
                # pair_golden = '|'.join(map(lambda pair: '-'.join(map(str, pair)), golden_pairs))

                predicted_pairs = self.find_pair(self.predictions[i], predicted_review_spans, predicted_reply_spans)
                # pair_pred = '|'.join(map(lambda pair: '-'.join(map(str, pair)), predicted_pairs))

                golden_labels, pred_labels = ['O'] * self.sen_lengths[i], ['O'] * self.sen_lengths[i]
                
                for start, end in golden_review_spans:
                    golden_labels[start] = 'B'
                    for idx in range(start+1, end + 1):
                        golden_labels[idx] = 'I'
                for start, end in golden_reply_spans:
                    golden_labels[start] = 'B'
                    for idx in range(start+1, end + 1):
                        golden_labels[idx] = 'I'
                for start, end in predicted_review_spans:
                    pred_labels[start] = 'B'
                    for idx in range(start+1, end + 1):
                        pred_labels[idx] = 'I'
                for start, end in predicted_reply_spans:
                    pred_labels[start] = 'B'
                    for idx in range(start+1, end + 1):
                        pred_labels[idx] = 'I'
                
                golden_disjoint_set = DisjointSet(len(golden_pairs))
                for m, n in itertools.combinations(range(len(golden_pairs)), 2):
                    review_start1, review_end1, reply_start1, reply_end1 = golden_pairs[m]
                    review_start2, review_end2, reply_start2, reply_end2 = golden_pairs[n]
                    if (review_start1 == review_start2 and review_end1 == review_end2) or (reply_start1 == reply_start2 and reply_end1 == reply_end2):
                        golden_disjoint_set.union(m, n)
                paired_golden_spans = golden_disjoint_set.cluster(golden_pairs)
                for pair_idx, paired_spans in enumerate(paired_golden_spans.values(), 1):
                    for review_start, review_end, reply_start, reply_end in paired_spans:
                        for idx in range(review_start, review_end + 1):
                            if not golden_labels[idx][-1].isdigit():
                                golden_labels[idx] += '-' + str(pair_idx) 
                        for idx in range(reply_start, reply_end + 1):
                            if not golden_labels[idx][-1].isdigit():
                                golden_labels[idx] += '-' + str(pair_idx)
                
                pred_disjoint_set = DisjointSet(len(predicted_pairs))
                for m, n in itertools.combinations(range(len(predicted_pairs)), 2):
                    review_start1, review_end1, reply_start1, reply_end1 = predicted_pairs[m]
                    review_start2, review_end2, reply_start2, reply_end2 = predicted_pairs[n]
                    if (review_start1 == review_start2 and review_end1 == review_end2) or (reply_start1 == reply_start2 and reply_end1 == reply_end2):
                        pred_disjoint_set.union(m, n)
                paired_pred_spans = pred_disjoint_set.cluster(predicted_pairs)
                for pair_idx, paired_spans in enumerate(paired_pred_spans.values(), 1):
                    for review_start, review_end, reply_start, reply_end in paired_spans:
                        for idx in range(review_start, review_end + 1):
                            if not pred_labels[idx][-1].isdigit():
                                pred_labels[idx] += '-' + str(pair_idx)
                        for idx in range(reply_start, reply_end + 1):
                            if not pred_labels[idx][-1].isdigit():
                                pred_labels[idx] += '-' + str(pair_idx)

                # for pair_idx, (review_start, review_end, reply_start, reply_end) in enumerate(golden_pairs, 1):
                #     for idx in range(review_start, review_end + 1):
                #         golden_labels[idx] += '-' + str(pair_idx)
                #     for idx in range(reply_start, reply_end + 1):
                #         golden_labels[idx] += '-' + str(pair_idx)
                # for pair_idx, (review_start, review_end, reply_start, reply_end) in enumerate(predicted_pairs, 1):
                #     for idx in range(review_start, review_end + 1):
                #         pred_labels[idx] += '-' + str(pair_idx)
                #     for idx in range(reply_start, reply_end + 1):
                #         pred_labels[idx] += '-' + str(pair_idx)
                
                # golden_labels_str, pred_labels_str = ' '.join(golden_labels), ' '.join(pred_labels)
                # f.write('\t'.join([golden_labels_str, pred_labels_str]) + '\n')
                
                for j in range(self.sen_lengths[i]):
                    f.write('\t'.join([golden_labels[j], pred_labels[j]]) + '\n')
                f.write('\n')

                    
                
