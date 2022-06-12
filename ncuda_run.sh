#!/bin/bash
python main.py --model_type MELBERT --bert_model roberta-base --learning_rate 5e-5
python main.py --model_type MELBERT --bert_model roberta-base --num_train_epoch 5
python main.py --model_type MELBERT --bert_model roberta-base --learning_rate 5e-5 --num_train_epoch 5
