#!/bin/bash

data_prefix='./data/wmt14'
model_dir='./data/model'
if [ ! -d "$model_dir" ]; then mkdir -p "$model_dir"; fi

CUDA_VISIBLE_DEVICES=2  python3 train.py \
                        -data $data_prefix \
                        -save_model $model_dir \
                        -save_checkpoint_steps 10 \
                        -valid_steps 5000 \
                        -report_every 20 \
                        -keep_checkpoint 20 \
                        -seed 3435 \
                        -train_steps 10 \
                        -warmup_steps 4000 \
                        --share_embeddings \
                        --share_decoder_embeddings \
                        --position_encoding \
                        --optim adam \
                        -adam_beta1 0.9 \
                        -adam_beta2 0.998 \
                        -decay_method noam \
                        -learning_rate 1.0 \
                        -max_grad_norm 0.0 \
                        -batch_size 4096 \
                        -batch_type tokens \
                        -normalization tokens \
                        -dropout 0.1 \
                        -label_smoothing 0.1 \
                        -max_generator_batches 100 \
                        -param_init 0.0 \
                        -param_init_glorot \
                        -valid_batch_size 16 \
                        -enc_layers 1 \
                        -dec_layers 1 \
                        -transformer_ff 512
