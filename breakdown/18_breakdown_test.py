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

def find_unknown(data, pred, label):
    basic = target_extract(data['train'])
    basicer = DefaultBasic()
    know_target = []
    uk_target = []

    lk_pred = []
    lk_label = []
    wn_pred = []
    wn_label = []
    uk_pred = []
    uk_label = []
    for i, sample in enumerate(data['test']):
        target = sample[0]
        sentence = sample[1]
        index = sample[2]
        
        outlog_basic=0
        target = sentence.split()
        target = target[int(index)]
        target = re.sub(r'[,.?();!:]','',target)
        if target in basic.keys():
            know_target.append(target)
            lk_pred = add_numpy(lk_pred, pred[i])
            lk_label = add_numpy(lk_label, label[i])
            outlog_basic = 0
        else:
            uk_target.append(target)
            basic_sen, basic_idx = basicer(target)
            if True:
            #if basic_sen == None:
                uk_pred = add_numpy(uk_pred, pred[i])
                uk_label = add_numpy(uk_label, label[i])
            else:    
                wn_pred = add_numpy(wn_pred, pred[i])
                wn_label = add_numpy(wn_label, label[i])
    #print(f'{lk_pred[0].shape},{wn_pred[0].shape},{uk_pred[0].shape}')
    print(f'{len(set(know_target))}, {len(set(uk_target))}')
    print(f'{lk_pred[0].shape},{uk_pred[0].shape}')
    print('********** all known ***********')
    print_metrics(lk_pred[0], lk_label[0])
    #print('********** word net ***********')
    #print_metrics(wn_pred[0], wn_label[0])
    print('********** unknow ***********')
    print_metrics(uk_pred[0], uk_label[0])
    
    return 0

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
        target = re.sub(r'[,.?();!:]','',target)
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
    path = '../data/VUA18'
    val_path = '../data/VUA18/val.tsv'
    test_path = '../data/VUA18/test.tsv'
    train_path = '../data/VUA18/train.tsv'
    data_emb_path = '../data/VUA18/basic_emb.json'

    logger.info('*****Load VUA Data*****')
    raw_train = load_tsv(train_path)
    raw_test = load_tsv(test_path)
    raw_val = load_tsv(val_path)
    data['train'] = vua_reform(raw_train)
    data['test'] = vua_reform(raw_test)
    data['val'] = vua_reform(raw_val)

    mel_pred=np.load("18_mel_preds.npy")
    #basic_pred=np.load("18_nwn_basic_preds.npy")
    basic_pred=np.load("18_wn_basic_preds.npy")
    mel_label=np.load("18_mel_labels.npy")
    #basic_label=np.load("18_nwn_basic_labels.npy")
    basic_label=np.load("18_wn_basic_labels.npy")
    print_metrics(mel_pred, mel_label)
    print_metrics(basic_pred, basic_label)

    find_unknown(data, mel_pred, mel_label)
    find_unknown(data, basic_pred, basic_label)
    
if __name__ == '__main__':
     main()
