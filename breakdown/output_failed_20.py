# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import torch
import re

from tqdm import tqdm
import numpy as np
import spacy

from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr, spearmanr, truncnorm
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
)
import random
import nltk


logger = logging.getLogger(__name__)

def failed_out(data, mel_pred, om_pred, label):
    basic = target_extract(data['train'])

    succeed = []

    for i, sample in enumerate(data['test']):
        target = sample[0]
        sentence = sample[1]
        index = sample[2]

        if target not in basic.keys():
            if mel_pred[i] == label[i]:
                continue
            if om_pred[i] == label[i]:
                succeed.append([target, index, sentence, label[i]])

    anl_cases(succeed)
    path = 'nb_cases.csv'
    save_tsv(succeed, path, headline=['target','index','sentence','label'])

def anl_cases(cases):
    meta = 0
    liter = 0
    for sample in cases:
        label = sample[3]
        if str(label) == '0':
            liter+=1
        else:
            meta+=1
    print(f'meta: {meta}, literal: {liter}')

def find_unknown(data, pred, label):
    basic = target_extract(data['train'])
    meta_basic = meta_extract(data['train'])
    all_basic = target_extract(data['train'], False)
    basicer = DefaultBasic()

    lk_lo_pred = []
    lk_lo_label = []
    lk_m_pred = []
    lk_m_label = []
    om_pred = []
    om_label = []
    uk_pred = []
    uk_label = []
    wn_m_pred = []
    wn_m_label = []
    wn_uk_pred = []
    wn_uk_label = []
    uk_m_pred = []
    uk_m_label = []
    uk_uk_pred = []
    uk_uk_label = []
    for i, sample in enumerate(data['test']):
        target = sample[0]
        sentence = sample[1]
        index = sample[2]
        
        target = sentence.split()
        target = target[int(index)]
        if target in meta_basic.keys():
            if target in basic.keys():
                lk_m_pred = add_numpy(lk_m_pred, pred[i])
                lk_m_label = add_numpy(lk_m_label, label[i])
            else:
                om_pred = add_numpy(om_pred, pred[i])
                om_label = add_numpy(om_label, label[i])
                target = re.sub(r'[,.?();!:]','',target)
                basic_sen, basic_idx = basicer(target)
                if basic_sen == None:
                    uk_m_pred = add_numpy(uk_m_pred, pred[i])
                    uk_m_label = add_numpy(uk_m_label, label[i])
                else:    
                    wn_m_pred = add_numpy(wn_m_pred, pred[i])
                    wn_m_label = add_numpy(wn_m_label, label[i])
        else:
            if target in basic.keys():
                lk_lo_pred = add_numpy(lk_lo_pred, pred[i])
                lk_lo_label = add_numpy(lk_lo_label, label[i])
                outlog_basic = 0
            else:
                uk_pred = add_numpy(uk_pred, pred[i])
                uk_label = add_numpy(uk_label, label[i])
                target = re.sub(r'[,.?();!:]','',target)
                basic_sen, basic_idx = basicer(target)
                if basic_sen == None:
                    uk_uk_pred = add_numpy(uk_uk_pred, pred[i])
                    uk_uk_label = add_numpy(uk_uk_label, label[i])
                else:    
                    wn_uk_pred = add_numpy(wn_uk_pred, pred[i])
                    wn_uk_label = add_numpy(wn_uk_label, label[i])
    
    lk_m_pred, lk_m_label, lk_m_len = hatch_pairs(lk_m_pred, lk_m_label)
    lk_lo_pred, lk_lo_label, lk_lo_len = hatch_pairs(lk_lo_pred, lk_lo_label)
    wn_m_pred, wn_m_label, wn_m_len = hatch_pairs(wn_m_pred, wn_m_label)
    wn_uk_pred, wn_uk_label, wn_uk_len = hatch_pairs(wn_uk_pred, wn_uk_label)
    uk_m_pred, uk_m_label, uk_m_len = hatch_pairs(uk_m_pred, uk_m_label)
    uk_uk_pred, uk_uk_label, uk_uk_len = hatch_pairs(uk_uk_pred, uk_uk_label)
    om_pred, om_label, om_len = hatch_pairs(om_pred, om_label)
    uk_pred, uk_label, uk_len = hatch_pairs(uk_pred, uk_label)
    
    print(f'{lk_m_len},{lk_lo_len},{om_len},{uk_len}')
    #print(f'{lk_m_len},{lk_lo_len},{wn_m_len},{uk_m_len},{wn_uk_len},{uk_uk_len}')
    print('############## basic contain meta & literal ##############')
    print_metrics(lk_m_pred[0], lk_m_label[0])
    print('############## basic contain literal only ##############')
    print_metrics(lk_lo_pred[0], lk_lo_label[0])
    print('############## basic contain meta only ##############')
    print_metrics(om_pred[0], om_label[0])
    '''
    print('****************** from wordnet ***********')
    print_metrics(wn_m_pred[0], wn_m_label[0])
    print('****************** not in wordnet ***********')
    print_metrics(uk_m_pred[0], uk_m_label[0])
    '''
    print('############## not in basic ##############')
    print_metrics(uk_pred[0], uk_label[0])
    '''
    print('****************** from wordnet ***********')
    print_metrics(wn_uk_pred[0], wn_uk_label[0])
    print('****************** not in wordnet ***********')
    print_metrics(uk_uk_pred[0], uk_uk_label[0])
    '''
    return 0

def hatch_pairs(pred, label):
    if len(pred) == 0:
        pred.append(np.array([0]))
        label.append(np.array([1]))
        length = 0
    else:
        length = pred[0].shape[0]
    return pred, label, length

def add_numpy(nlist, elem):
    if len(nlist) == 0:
        nlist.append(elem)
    else:
        nlist[0] = np.append(nlist[0], elem)
    return nlist

def vua_reform(data, new=False):
    r_data=[]
    for s in tqdm(data):
        sentence = s[2].lower()
        if new:
            w_index = int(s[5])
            fgpos = s[4]
        else:
            w_index = int(s[4])
            fgpos = s[3]
        target = sentence.split()[w_index]
        label = int(s[1])
        index = s[0]
        pos = s[3]
        
        r_data.append([target, sentence, w_index, label, pos, fgpos, index])
    return r_data

def save_tsv(data, path, headline=None):
    print(f'{path} len: {len(data)}')
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        if headline:
            writer.writerow(headline)
        writer.writerows(data)
        
def load_tsv(path):
    data=[]
    with open(path) as f:
        lines = csv.reader(f, delimiter='\t')
        next(lines)
        for line in lines:
            data.append(list(line))
    return data

def target_extract(train_set, basic=True):
    basic_train = {}
    for sample in tqdm(train_set):
        target = sample[0]
        sentence = sample[1]
        index = sample[2]
        label = str(sample[3])
        if label == '1' and basic:
            continue
        if target in basic_train.keys():
            basic_train[target]['sam'].append([sentence, index])
        else:
            basic_train[target] = {'sam':[[sentence, index]]}
    print(f'length: {len(basic_train)}')
    return basic_train

def meta_extract(train_set, basic=True):
    basic_train = {}
    for sample in tqdm(train_set):
        target = sample[0]
        sentence = sample[1]
        index = sample[2]
        label = str(sample[3])
        if label == '0' and basic:
            continue
        if target in basic_train.keys():
            basic_train[target]['sam'].append([sentence, index])
        else:
            basic_train[target] = {'sam':[[sentence, index]]}
    print(f'length: {len(basic_train)}')
    return basic_train


      
class DefaultBasic:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])

    def __call__(self, word):
        sent = None
        index = None
        if len(wn.synsets(word))==0:
            return None, None
        syn = wn.synsets(word)[0]
        lemmas = syn.lemmas()
        examples = syn.examples()
        if not examples: # Means there is no example sentence in wordnet
            return None, None
        doc = self.nlp(examples[0])
        for lemma in lemmas:
            for idx, t in enumerate(doc):
                if lemma.name() == t.lemma_:
                    index = idx
                    sent = [token.text for token in doc]
                    break
            else:
                continue
            break
        return sent, index


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def seq_accuracy(preds, labels):
    acc = []
    for idx, pred in enumerate(preds):
        acc.append((pred == labels[idx]).mean())
    return acc.mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def all_metrics(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    pre = precision_score(y_true=labels, y_pred=preds)
    rec = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return all_metrics(preds, labels)

def print_metrics(pred, label):
    result = compute_metrics(pred, label)
    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    data = {}
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    path = '../data/VUA20'
    val_path = '../data/VUA18/val.tsv'
    test_path = '../data/VUA20/test.tsv'
    train_path = '../data/VUA20/train.tsv'
    data_emb_path = '../data/VUA20/basic_emb.json'

    logger.info('*****Load VUA Data*****')
    raw_train = load_tsv(train_path)
    raw_test = load_tsv(test_path)
    raw_val = load_tsv(val_path)
    data['train'] = vua_reform(raw_train, True)
    data['test'] = vua_reform(raw_test, True)
    data['val'] = vua_reform(raw_val)

    mel_pred=np.load("mel_20_preds.npy")
    basic_pred=np.load("20_preds.npy")
    #basic_pred=np.load("../preds.npy")
    mel_label=np.load("mel_20_labels.npy")
    basic_label=np.load("20_labels.npy")
    #nsistent')
    print_metrics(mel_pred, mel_label)
    print_metrics(basic_pred, basic_label)

    find_unknown(data, mel_pred, mel_label)
    find_unknown(data, basic_pred, basic_label)
    failed_out(data, mel_pred, basic_pred, mel_label)
    
if __name__ == '__main__':
     main()
