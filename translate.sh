#!/bin/bash

train_test_data_dir='/home/zaiheshi/algorithm-work/opennmt-base/data/test'
model_file='/home/zaiheshi/algorithm-work/opennmt-base/data/model_step_10.pt'
output_dir='/home/zaiheshi/algorithm-work/opennmt-base/data/test'
output_file=$output_dir/test.tran

CUDA_VISIBLE_DEVICES=2  python3 ./translate.py \
                        -src $train_test_data_dir/test.de \
                        -output $output_file \
                        -model $model_file \
                        -beam_size 5