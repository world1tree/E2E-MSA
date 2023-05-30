"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.constants as Constants

from onmt.transformer_encoder import TransformerEncoder
from onmt.transformer_decoder import TransformerDecoder

from onmt.embeddings import Embeddings
from utils.logging import logger

from transformers import T5EncoderModel, T5Tokenizer

class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths):
        # (seq_len, batch_size), 有<bos>与<eos
        # 目的是排除<eos>? 但是有一部分数据仅仅排除了padding  TODO
        # tgt = tgt[:-1]  # exclude last target from inputs

        # emb: (seq_len, batch_size, dim), encoder_out: (seq_len, batch_size, dim), src_length: batch_size
        _, memory_bank, lengths = self.encoder(src, lengths)
        self.decoder.init_state(src, memory_bank)
        dec_out, attns = self.decoder(tgt)

        # decoder_output: (seq_len, batch_size, dim)
        # context-attn: (**seq_len_tgt**, batch_size, seq_len_src)
        return dec_out, attns

def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    return TransformerEncoder(opt.enc_layers, opt.enc_rnn_size,
                              opt.heads, opt.transformer_ff,
                              opt.dropout, embeddings)

def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    return TransformerDecoder(opt.dec_layers, opt.dec_rnn_size,
                              opt.heads, opt.transformer_ff,
                              opt.dropout, embeddings)

def build_embeddings():
    embedding = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    embedding.tokenizer =tokenizer
    return embedding

def build_base_model(model_opt, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """

    # for backward compatibility
    if model_opt.enc_rnn_size != model_opt.dec_rnn_size:
        raise AssertionError("""We do not support different encoder and
                         decoder rnn sizes for translation now.""")

    # Build encoder.
    src_embeddings = build_embeddings()
    encoder = build_encoder(model_opt, src_embeddings)

    # Build decoder.
    tgt_embeddings = src_embeddings

    if not model_opt.share_embeddings:
        raise RuntimeError("Should share embedding.")

    decoder = build_decoder(model_opt, tgt_embeddings)

    # Build NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)

    # Build Generator.
    # gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(model_opt.dec_rnn_size, 1), # 修改为回归模型
        # gen_func
    )
    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = \
            {fix_key(k): v for (k, v) in checkpoint['model'].items()}
        # end of patch for backward compatibility

        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
        # 使用xavier初始化
        for name, p in model.named_parameters():
            if "embeddings" in name:
                continue
            if p.dim() > 1:
                xavier_uniform_(p)
        for name, p in generator.named_parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    model.generator = generator
    return model


def build_model(model_opt, checkpoint=None):
    """ Build the Model """
    logger.info('Building model...')
    model = build_base_model(model_opt, checkpoint)
    logger.info(model)
    return model
