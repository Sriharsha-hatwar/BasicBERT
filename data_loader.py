import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, IterableDataset, BatchSampler
from run_classifier_dataset_utils import (
    convert_examples_to_two_features,
    convert_examples_to_features,
    convert_two_examples_to_features,
)


def load_train_data_melbert(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    # Prepare data loader
    if task_name == "vua":
        train_examples = processor.get_train_examples(args.data_dir)
    elif task_name == "trofi":
        train_examples = processor.get_train_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    # make features file
    if args.model_type == "BERT_BASE":
        train_features = convert_two_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        train_features = convert_examples_to_two_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    # make features into tensor
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # add additional features for MELBERT_MIP and MELBERT
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
        all_input_mask_2 = torch.tensor([f.input_mask_2 for f in train_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in train_features], dtype=torch.long)
        all_basic_ids = torch.tensor([f._input_ids for f in train_features], dtype=torch.long)
        all_basic_mask = torch.tensor([f._input_mask for f in train_features], dtype=torch.long)
        all_basic_segment = torch.tensor([f._segment_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
            all_basic_ids,
            all_basic_mask,
            all_basic_segment,
        )
        '''
        print(f'input_ids : {all_input_ids}')
       
        print(f'input_mask : {all_input_mask}')
        print(f'segment_ids : {all_segment_ids}')
        
        print(f'_input_ids : {all_basic_ids}')
        
        print(f'_input_mask : {all_basic_mask}')
        print(f'_segment_ids : {segment_ids}')
        '''
    else:
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    return train_dataloader

class BasicLoader(IterableDataset):
    def __init__(self, batch_size, feature, val=False):
        self.feature = feature
        self.val = val
        self._sampler = SequentialSampler if self.val else RandomSampler
        self.batch_size = batch_size
        self._batch_sampler = BatchSampler(self._sampler(self.feature), self.batch_size, False)

        self.max_steps = len(self._batch_sampler)

    def _collate_fn(self, batch):
        if self.val:
            input_ids,input_mask,segment_ids,label_ids, all_idx,input_ids_2,input_mask_2,segment_ids_2,basic_ids,basic_mask,basic_segment = zip(*batch)
        else:
            input_ids,input_mask,segment_ids,label_ids,input_ids_2,input_mask_2,segment_ids_2,basic_ids,basic_mask,basic_segment = zip(*batch)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        #label_ids = pad_sequence(label_ids, batch_first=True, padding_value=0)
        input_ids_2 = pad_sequence(input_ids_2, batch_first=True, padding_value=0)
        input_mask_2 = pad_sequence(input_mask_2, batch_first=True, padding_value=0)
        segment_ids_2 = pad_sequence(segment_ids_2, batch_first=True, padding_value=0)
        basic_ids = pad_sequence(basic_ids, batch_first=True, padding_value=0)
        basic_mask = pad_sequence(basic_mask, batch_first=True, padding_value=0)
        basic_segment = pad_sequence(basic_segment, batch_first=True, padding_value=0) # [N,BN,L]

        
        
        labels = torch.tensor([t for t in label_ids])
        if self.val:
            all_idx = torch.tensor([t for t in all_idx])
            return input_ids,input_mask,segment_ids,labels, all_idx,input_ids_2,input_mask_2,segment_ids_2,basic_ids.view([int(4*input_ids.shape[0]), input_ids.shape[1]]),basic_mask.view([int(4*input_ids.shape[0]), input_ids.shape[1]]),basic_segment.view([int(4*input_ids.shape[0]), input_ids.shape[1]])
        return input_ids,input_mask,segment_ids,labels,input_ids_2,input_mask_2,segment_ids_2,basic_ids.view([int(4*input_ids.shape[0]), input_ids.shape[1]]),basic_mask.view([int(4*input_ids.shape[0]), input_ids.shape[1]]),basic_segment.view([int(4*input_ids.shape[0]), input_ids.shape[1]])

    def _sample_batch(self, idxs):
        batch=[]
        for n, i in enumerate(idxs):
            f = self.feature[i]
            all_idx = torch.tensor(i, dtype=torch.long)
            input_ids = torch.tensor(f.input_ids, dtype=torch.long)
            input_mask = torch.tensor(f.input_mask, dtype=torch.long)
            segment_ids = torch.tensor(f.segment_ids, dtype=torch.long)
            label_ids = torch.tensor(f.label_id, dtype=torch.long)
            input_ids_2 = torch.tensor(f.input_ids_2, dtype=torch.long)
            input_mask_2 = torch.tensor(f.input_mask_2, dtype=torch.long)
            segment_ids_2 = torch.tensor(f.segment_ids_2, dtype=torch.long)
            basic_ids = torch.tensor(f._input_ids, dtype=torch.long)
            basic_mask = torch.tensor(f._input_mask, dtype=torch.long)
            basic_segment = torch.tensor(f._segment_ids, dtype=torch.long)
            if self.val:
                batch.append([input_ids,input_mask,segment_ids,label_ids, all_idx,input_ids_2,input_mask_2,segment_ids_2,
                             basic_ids,basic_mask,basic_segment])
            else:   
                batch.append([input_ids,input_mask,segment_ids,label_ids,input_ids_2,input_mask_2,segment_ids_2,
                             basic_ids,basic_mask,basic_segment])
        return self._collate_fn(batch)
                

    def __iter__(self):
        for idxs in self._batch_sampler:
            batch = self._sample_batch(idxs)
            yield batch

    def __len__(self):
        return self.max_steps

def load_train_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    if task_name == "vua":
        train_examples = processor.get_train_examples(args.data_dir)
    elif task_name == "trofi":
        train_examples = processor.get_train_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    # make features file
    if args.model_type == "BERT_BASE":
        train_features = convert_two_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        train_features = convert_examples_to_two_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    dataloader = BasicLoader(args.train_batch_size, train_features)

    return dataloader

def load_test_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    if task_name == "vua":
        eval_examples = processor.get_test_examples(args.data_dir)
    elif task_name == "trofi":
        eval_examples = processor.get_test_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    if args.model_type == "BERT_BASE":
        eval_features = convert_two_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        eval_features = convert_examples_to_two_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    logger.info("***** Running evaluation *****")
    
    all_guids = [f.guid for f in eval_features]
    dataloader = BasicLoader(args.eval_batch_size, eval_features, True)

    return all_guids, dataloader
    

def load_train_data_kf(
    args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None
):
    # Prepare data loader
    if task_name == "vua":
        train_examples = processor.get_train_examples(args.data_dir)
    elif task_name == "trofi":
        train_examples = processor.get_train_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    # make features file
    if args.model_type == "BERT_BASE":
        train_features = convert_two_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        train_features = convert_examples_to_two_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    # make features into tensor
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # add additional features for MELBERT_MIP and MELBERT
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
        all_input_mask_2 = torch.tensor([f.input_mask_2 for f in train_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor(
            [f.segment_ids_2 for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    else:
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    gkf = StratifiedKFold(n_splits=args.num_bagging).split(X=all_input_ids, y=all_label_ids.numpy())
    return train_data, gkf


def load_test_data_origin(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    if task_name == "vua":
        eval_examples = processor.get_test_examples(args.data_dir)
    elif task_name == "trofi":
        eval_examples = processor.get_test_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    if args.model_type == "BERT_BASE":
        eval_features = convert_two_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        eval_features = convert_examples_to_two_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    logger.info("***** Running evaluation *****")
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_guids = [f.guid for f in eval_features]
        all_idx = torch.tensor([i for i in range(len(eval_features))], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in eval_features], dtype=torch.long)
        all_input_mask_2 = torch.tensor([f.input_mask_2 for f in eval_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in eval_features], dtype=torch.long)
        all_basic_ids = torch.tensor([f._input_ids for f in eval_features], dtype=torch.long)
        all_basic_mask = torch.tensor([f._input_mask for f in eval_features], dtype=torch.long)
        all_basic_segment = torch.tensor([f._segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_idx,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
            all_basic_ids,
            all_basic_mask,
            all_basic_segment,
        )
    else:
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_guids = [f.guid for f in eval_features]
        all_idx = torch.tensor([i for i in range(len(eval_features))], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_idx
        )

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return all_guids, eval_dataloader
