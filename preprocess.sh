#!/bin/bash

train_test_data_dir='./corpus/small_dataset'
data_dir='./data'
if [ ! -d "$data_dir" ]; then mkdir -p "$data_dir"; fi
data_prefix="$data_dir/wmt14"

python3 ./preprocess.py -train_src $train_test_data_dir/train.de \
                       -train_tgt $train_test_data_dir/train.en \
                       -valid_src $train_test_data_dir/val.de  \
                       -valid_tgt $train_test_data_dir/val.en \
                       -save_data $data_prefix \
                       -src_vocab_size 5000  \
                       -tgt_vocab_size 5000 \
                       -src_seq_length 150 \
                       -tgt_seq_length 150 \
                       -share_vocab
