import json, os
import random
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import trange

from data import load_data_instances, DataIterator
from baseline_bert_linear import BaselineBertModel
from baseline_lstm_linear import BaselineLSTMModel
from utils import Metric, Writer, get_huggingface_optimizer_and_scheduler, context_models
import math
from termcolor import colored

def train(args):
    
    random.seed(args.random_seed)
    
    if not args.test_code:
        train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))
        random.shuffle(train_sentence_packs)
        dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))
        test_sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    else:
        train_sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
        dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
        test_sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))

    train_batch_count = math.ceil(len(train_sentence_packs)/args.batch_size)

    instances_train = load_data_instances(train_sentence_packs, args)
    instances_dev = load_data_instances(dev_sentence_packs, args)
    instances_test = load_data_instances(test_sentence_packs, args)

    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)
    testset = DataIterator(instances_test, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    bertModel = context_models[args.bert_model_path]['model'].from_pretrained(args.bert_model_path, return_dict=False)
    if args.model == 'BaselineLSTMModel':
        model = BaselineLSTMModel(args, bertModel).to(args.device)
    elif args.model == 'BaselineBertModel':
        model = BaselineBertModel(args, bertModel).to(args.device)
    
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(args, model, num_training_steps=train_batch_count * args.epochs,
                                                                    weight_decay=0.0, eps = 1e-8, warmup_step=0)

    best_joint_f1 = -1
    best_joint_epoch = 0
    best_joint_precision = 0
    joint_precision = 0
    best_joint_recall = 0
    
    for i in range(1, args.epochs+1):
        model.zero_grad()
        model.train()
        print('Epoch:{}'.format(i))
        losses = []
        pair_losses = []
        crf_losses = []
    
        for j in trange(trainset.batch_count):
            
            _, embedder_input, lengths, _, tags, biotags, masks, _, type_idx = trainset.get_batch(j)
            
            pair_logits, crf_loss = model(embedder_input, masks, lengths, biotags, type_idx)
            logits_flatten = pair_logits.reshape(-1, pair_logits.size()[-1])
            tags_flatten = tags.reshape([-1])
            pair_loss = F.cross_entropy(logits_flatten, tags_flatten, ignore_index=-1, reduction='sum')
            loss = args.pair_weight*pair_loss + crf_loss
            losses.append(loss.item())
            pair_losses.append(pair_loss.item()*args.pair_weight)
            crf_losses.append(crf_loss.item())
            loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if args.optimizer == 'adamw':
                scheduler.step()
            model.zero_grad()

        print('average loss {:.4f}'.format(np.average(losses)))
        print('average pairing loss {:.4f}'.format(np.average(pair_losses)))
        print('average crf loss {:.4f}'.format(np.average(crf_losses)))
        print(colored('Evaluating dev set: ', color='red'))
        joint_precision, joint_recall, joint_f1 = eval(model, devset, args)
        print(colored('Evaluating test set: ', color='red'))
        _, _, _ = eval(model, testset, args)

        if joint_f1 > best_joint_f1:
            model_path = args.model_dir + args.model + args.task + '.pt'
            torch.save(model, model_path)
            best_joint_precision = joint_precision
            best_joint_recall = joint_recall
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print(colored('Final evluation on dev set: ', color='red'))
    print('best epoch: {}\tbest dev precision: {:.5f}\tbest dev recall: {:.5f}\tbest dev f1: {:.5f}\n\n'.format(best_joint_epoch, best_joint_precision, best_joint_recall, best_joint_f1))


def eval(model, dataset, args, output_results=False):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_last_review_indice = []
        all_bio_preds = []
        all_bio_golds = []
        for i in range(dataset.batch_count):
            sentence_ids, embedder_input, lengths, sens_lens, tags, biotags, masks, last_review_indice, type_idx = dataset.get_batch(i)
            pair_logits, _, decode_idx = model.decode(embedder_input, masks, lengths, type_idx)
            pair_preds = torch.argmax(pair_logits, dim=3)
            all_preds.extend(pair_preds.cpu().tolist())
            all_labels.extend(tags.cpu().tolist())
            all_lengths.extend(lengths.cpu().tolist())
            all_sens_lengths.extend(sens_lens)
            all_ids.extend(sentence_ids)
            all_last_review_indice.extend(last_review_indice)
            all_bio_golds.extend(biotags.cpu().tolist())
            all_bio_preds.extend(decode_idx.cpu().tolist())

        metric = Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_last_review_indice, all_bio_golds, all_bio_preds)
        precision, recall, f1 = metric.score_uniontags()
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()
        bio_results = metric.score_bio(aspect_results, opinion_results)
        pair_results = metric.score_pair()
        # print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
        #                                                           aspect_results[2]))
        # print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
        #                                                            opinion_results[2]))
        print('Argument\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(bio_results[0], bio_results[1],
                                                                   bio_results[2]))
        print('Pairing\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(pair_results[0], pair_results[1],
                                                               pair_results[2]))
        print('Overall\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))
    
    if output_results:
            writer = Writer(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_last_review_indice, all_bio_golds, all_bio_preds)
            writer.output_results()

    model.train()
    return precision, recall, f1


def test(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, args.model_dir.strip('/'))):
        os.makedirs(os.path.join(args.output_dir, args.model_dir.strip('/')))

    print(colored('Final evluation on test set: ', color='red'))
    model_path = args.model_dir + args.model + args.task + '.pt'
    model = torch.load(model_path).to(args.device)
    model.eval()

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    instances = load_data_instances(sentence_packs, args)
    testset = DataIterator(instances, args)
    eval(model, testset, args, output_results=True)

    """
    count number of parameters
    """
    num_param = 0
    for layer, weights in model.state_dict().items():
        if layer.startswith('embedder.bert'):
            continue
        prod = 1
        for dim in weights.size():
            prod *= dim
        num_param += prod
    print(colored('There are in total {} parameters within the model'.format(num_param), color='yellow'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savedmodel/",
                        help='model path prefix')
    parser.add_argument('--output_dir', type=str, default="./outputs",
                        help='test dataset outputs directory')
    parser.add_argument('--task', type=str, default="pair", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--dataset', type=str, default="rr-submission-new",
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=210,
                        help='max length of a sentence')
    parser.add_argument('--max_token_len', type=int, default=10000,
                        help='max length of a paragraph')
    parser.add_argument('--max_bert_token', type=int, default=200,
                        help='max length of bert tokens for one sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')
    parser.add_argument('--split_size', type=int, default=10,
                        help='split for bert model')
    parser.add_argument('--num_instances', type=int, default=-1,
                        help='number of instances to read')
    parser.add_argument('--test_code', type=bool, default=False,
                        help='to read train/dev/test or test/test/test')
    parser.add_argument('--random_seed', type=int, default=1,
                        help='set random seed')

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-cased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_tokenizer_path', type=str,
                        default="bert-base-cased",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--token_embedding', type=bool, default=False,
                        help='additional lstm embedding over pre-trained bert token embeddings')
    parser.add_argument('--freeze_bert', type=bool, default=False,
                        help='whether to freeze parameters of pre-trained bert model')
    parser.add_argument('--num_embedding_layer', type=int, default=1,
                        help='number of layers for token LSTM')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--hidden_dim', type=int, default=200,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--embedding_dim', type=int, default=20,
                        help='dimension of type embedding')
    parser.add_argument('--sentiment_dim', type=int, default=10,
                        help='dimension of sentiment embedding')
    parser.add_argument('--function_dim', type=int, default=10,
                        help='dimension of function embedding')
    parser.add_argument('--layer_norm', type=bool, default=False,
                        help='whether apply layer normalization to RNN model')
    parser.add_argument('--bidirectional', type=bool, default=False,
                        help='whether use bidirectioanl 2D-GRU for RNN model')
    parser.add_argument('--attention', type=str, default='tanh', choices=['tanh', 'cosine_similarity'],
                        help='attention mechanism')
    parser.add_argument('--parallel', type=bool, default=False,
                        help='whether apply parallelism for bert embedder')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate') 
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='maximum gradient norm used during backprop') 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam'],
                        help='optimizer choice') 
    parser.add_argument('--pair_weight', type=float, default=0.5,
                        help='pair loss weight coefficient for loss computation')
    parser.add_argument('--attn_weight', type=float, default=10,
                        help='attention loss weight coefficient for loss computation')
    parser.add_argument('--ema', type=float, default=1.0,
                        help='EMA coefficient alpha')
    parser.add_argument('--cnn_classifier', type=bool, default=False,
                        help='whether to use cnn for pairing predictor') 
                        
    parser.add_argument('--pair_threshold', type=float, default=0.5,
                        help='pairing threshold during evaluation') 
    parser.add_argument('--iteration', type=int, default=2,
                        help='iteration for iterative model')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='kernel size for both CNN module')
    parser.add_argument('--cls_method', type=str, default='multiclass', choices=['binary', 'multiclass'],
                        help='relates to pairing classification')
    parser.add_argument('--model', type=str, default='JointModel', choices=['IterativeModel', 'IterativeAttnLossModel', 'ExternalRNNModel', 'JointModel', 'BaselineLSTMModel', 'BaselineBertModel', 'RNNModel', 'ExternalLSTMModel'],
                        help='model choice')
    parser.add_argument('--sentiment_model_path', type=str, default='../../external/results/sentiment_model',
                        help='pre-trained sentiment model path')
    parser.add_argument('--function_model_path', type=str, default='../../external/results/function_model',
                        help='pre-trained function model path')
                        

    parser.add_argument('--nhops', type=int, default=1,
                        help='inference times')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=4,
                        help='label number')
    parser.add_argument('--bio_class_num', type=int, default=6,
                        help='label number')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'

    if args.task == 'triplet':
        args.class_num = 6

    if args.mode == 'train':
        train(args)
        test(args)
    else:
        test(args)
