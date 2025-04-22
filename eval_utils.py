# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch.nn as nn
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup
import re
from data_utils import aspect_cate_list
import spacy 
import editdistance
from data_utils import read_line_examples_from_file


sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
sentiment_word_list = ['positive', 'negative', 'neutral']
sentiment_word_lis = ['great', 'bad', 'ok']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}

aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'food general',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']


def read_line(data_im_path, silence):

    with open(data_im_path, 'r', encoding='UTF-8') as fp:
        im_inf = []
        for line in fp:
            line = line.strip()
            if line != '':
                im_inf.append(line)
    if silence:
        print(f"Total examples")

    # print(im_inf)

    return im_inf


def extract_spans_para(task, seq, review, seq_type):
    review = review.lower()
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    if task == 'at':
        for s in sents:
            # It is bad because editing is problem.
            try:
                
                # ac, at = s.split(': ')
                at = s
                at = at.lower()
                idx = at.find("'")
                if idx != -1:
                    at = s[:idx].lower() + " '" + s[idx+1:].lower()

                if at != 'it' and at not in review:
                    print(at)
                    print(review)
                    break

                # if the aspect term is implicit
                if at == 'it':
                    at = 'NULL'
    
            except ValueError:
                at = ''
            quads.append((at))

    elif task == 'aesc':
        for s in sents:
            # It is bad because editing is problem.
            try:

                # at_, sp = s.split(', sentiment polarity: ')
                # a, at=at_.split('aspect term: ')

                at, sp = s.split(' is ')

                at = at.lower()
                idx = at.find("'")
                if idx != -1:
                    at = s[:idx].lower() + " '" + s[idx+1:].lower()
                sp = opinion2word.get(sp, 'nope')    # 'good' -> 'positive'

                if at != 'it' and at not in review:
                    break


                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
    
            except ValueError:
                at, sp = '', ''
            quads.append((at, sp))

    elif task == 'tasd':
        for s in sents:
            # food quality is bad because pizza is bad.
            try:
                
                # at_, ac_sp = s.split(', sentiment polarity: ')
                # a, at=at_.split('aspect term: ')
                # sp, ac = ac_sp.split(', aspect category: ')

                # at_sp, ac_sp = s.split(' means ')

                at_sp, ac_sp = s.split(' indicates ')
                ac, sp = ac_sp.split(' is ')
                at, sp2 = at_sp.split(' is ')
                #if ac not in aspect_cate_list:
                    # print(ac)
                    # print(review)
                    # break

                at = at.lower()
                idx = at.find("'")
                if idx != -1:
                    at = s[:idx].lower() + " '" + s[idx+1:].lower()
                sp = opinion2word.get(sp, 'nope')
                sp2 = opinion2word.get(sp2, 'nope')     

                if at != 'it' and at not in review:
                    print(at)
                    print(s)
                    print(review)                   
                    break

                # if sp != sp2:
                #     print(f'Sentiment polairty of AC({sp}) and AT({sp2}) is inconsistent!')
                
                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
       
            except ValueError:
                ac, at, sp = '', '', ''
            quads.append((at, ac, sp))

    elif task == 'aste':
        for s in sents:
            # It is bad because editing is problem.
            try:
                # at_ot, sp = s.split(', sentiment polarity: ')
                # at_, ot = at_ot.split(', opinion term: ')
                # a, at=at_.split('aspect term: ')

                # at_ot, ac_sp = s.split(' means ')

                at_ot, ac_sp = s.split(' indicates ')
                at, ot = at_ot.split(' is ')
                ac, sp = ac_sp.split(' is ')       
                # if 'indicates' in at:
                #     at_ot, ac_sp = at.split(' indicates ')
                #     at, ot = at_ot.split(' is ')
                #     ac, sp = ac_sp.split(' is ')                   

                at = at.lower()
                idx = at.find("'s")
                if idx != -1:
                    at = s[:idx].lower() + " '" + s[idx+1:].lower()

                ot = ot.lower()
                sp = opinion2word.get(sp, 'nope') # 'good' -> 'positive'

                if at != 'it' and at not in review:
                    print(at)
                    print(review)
                    break
                if ot not in review:
                    break

                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
    
            except ValueError:
                at, ot, sp = '', '', ''
            quads.append((at, sp, ot))

    elif task == 'asqp':
        for s in sents:
            # food quality is bad because pizza is over cooked.
            try:
                # at_ot, ac_sp = s.split(', sentiment polarity: ')
                # at_, ot = at_ot.split(', opinion term: ')
                # a, at=at_.split('aspect term: ')
                # sp, ac = ac_sp.split(', aspect category: ')

                # at_ot, ac_sp = s.split(' means ')
                
                at_ot, ac_sp = s.split(' indicates ')
                at, ot = at_ot.split(' is ')
                ac, sp = ac_sp.split(' is ')

                #if ac not in aspect_cate_list:
                #    print(ac)
                #    print(review) 
                #    break
                at = at.lower()
                idx = at.find("'")
                if idx != -1:
                    at = s[:idx].lower() + " '" + s[idx+1:].lower()
                ot = ot.lower()
                sp = opinion2word.get(sp, 'nope') # 'good' -> 'positive'

                if at != 'it' and at not in review:
                    print(at)
                    print(review) 
                    break
                if ot not in review:
                    break
                
                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'

            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                ac, at, sp, ot = '', '', '', ''
            quads.append((at, ac, sp, ot))
    else:
        raise NotImplementedError
    quads = list(set(quads))
    # print(quads)
    return quads

def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}
    return scores

def write_line_examples_to_file(reviews, labels, output_path):
    """
    Write data to file in the format: sent####labels
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='UTF-8') as fp:
        for i in range(len(reviews)):
            line = '####'.join([reviews[i], str(labels[i])])
            fp.write(line + '\n')


def compute_scores(dataset_type, task_t, pred_seqs, gold_seqs, reviews, sents):
    """
    Compute model performance
    """

    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    task = task_t
    for i in tqdm(range(num_samples)):
        # print('真实')
        gold_list = extract_spans_para(task, gold_seqs[i], reviews[i], 'gold')
        # print('预测')
        pred_list = extract_spans_para(task, pred_seqs[i], reviews[i], 'pred')
        all_labels.append(gold_list)
        all_preds.append(pred_list)

    # write_line_examples_to_file(reviews, all_preds, f'./fdata/rest16/5/{task}_preds.txt')
    
    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels)
    print(scores)

    return scores, all_labels, all_preds

