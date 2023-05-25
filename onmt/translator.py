#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import configargparse
import onmt.opts as opts
import torch
import onmt.transformer as nmt_model
from inputters.dataset import build_dataset, OrderedIterator, make_features
from onmt.beam import Beam
from utils.misc import tile
import onmt.constants as Constants 
import time

def build_translator(opt):
  dummy_parser = configargparse.ArgumentParser(description='translate.py')
  opts.model_opts(dummy_parser)
  dummy_opt = dummy_parser.parse_known_args([])[0]

  fields, model = nmt_model.load_test_model(opt, dummy_opt.__dict__)
  
  translator = Translator(model, fields, opt)

  return translator

class Translator(object):
  def __init__(self, model, fields, opt, out_file=None):
    self.model = model
    self.fields = fields
    self.gpu = opt.gpu
    self.cuda = opt.gpu > -1
    self.device = torch.device('cuda' if self.cuda else 'cpu')
    self.decode_extra_length = opt.decode_extra_length
    self.decode_min_length = opt.decode_min_length
    self.beam_size = opt.beam_size
    self.min_length = opt.min_length
    self.minimal_relative_prob = opt.minimal_relative_prob
    self.out_file = out_file
    self.tgt_eos_id = fields["tgt"].vocab.stoi[Constants.EOS_WORD]
    self.tgt_bos_id = fields["tgt"].vocab.stoi[Constants.BOS_WORD]
    self.src_eos_id = fields["src"].vocab.stoi[Constants.EOS_WORD]
  
  def build_tokens(self, idx, side="tgt"):
    assert side in ["src", "tgt"], "side should be either src or tgt"
    vocab = self.fields[side].vocab
    if side == "tgt":
      eos_id = self.tgt_eos_id
    else:
      eos_id = self.src_eos_id
    tokens = []
    for tok in idx:
      if tok == eos_id:
        break
      if tok < len(vocab):
        tokens.append(vocab.itos[tok])
    return tokens  
  
  def translate(self, src_data_iter, tgt_data_iter, batch_size, out_file=None):
    data = build_dataset(self.fields,
                         src_data_iter=src_data_iter,
                         tgt_data_iter=tgt_data_iter,
                         use_filter_pred=False)
    
    def sort_translation(indices, translation):
      ordered_transalation = [None] * len(translation)
      for i, index in enumerate(indices):
        ordered_transalation[index] = translation[i]
      return ordered_transalation
    
    if self.cuda:
        cur_device = "cuda"
    else:
        cur_device = "cpu"

    data_iter = OrderedIterator(
      dataset=data, device=cur_device,
      batch_size=batch_size, train=False, sort=True,
      sort_within_batch=True, shuffle=True)
    start_time = time.time()
    print("Begin decoding ...")
    batch_count = 0
    all_translation = []
    for batch in data_iter:
      hyps, scores = self.translate_batch(batch)
      assert len(batch) == len(hyps)
      batch_transtaltion = []
      for src_idx_seq, tran_idx_seq, score in zip(batch.src[0].transpose(0, 1), hyps, scores):
        src_words = self.build_tokens(src_idx_seq, side='src')
        src = ' '.join(src_words)
        tran_words = self.build_tokens(tran_idx_seq, side='tgt')
        tran = ' '.join(tran_words)
        batch_transtaltion.append(tran)
        print("SOURCE: " + src + "\nOUTPUT: " + tran + "\n")
      for index, tran in zip(batch.indices.data, batch_transtaltion):
        while (len(all_translation) <=  index):
          all_translation.append("")
        all_translation[index] = tran
      batch_count += 1
      print("batch: " + str(batch_count) + "...")
      
    if out_file is not None:
      for tran in all_translation:
        out_file.write(tran + '\n')
    print('Decoding took %.1f minutes ...'%(float(time.time() - start_time) / 60.))
  
  def translate_batch(self, batch):
    def get_inst_idx_to_tensor_position_map(inst_idx_list):
      ''' Indicate the position of an instance in a tensor. '''
      return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}
    
    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
      ''' Collect tensor parts associated to active instances. '''

      _, *d_hs = beamed_tensor.size()
      n_curr_active_inst = len(curr_active_inst_idx)
      new_shape = (n_curr_active_inst * n_bm, *d_hs)

      beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
      beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
      beamed_tensor = beamed_tensor.view(*new_shape)

      return beamed_tensor
    
    def beam_decode_step(
      inst_dec_beams, len_dec_seq, inst_idx_to_position_map, n_bm):
      ''' Decode and update beam status, and then return active beam idx '''
      # inst_dec_beams: 一个list, 元素是Beam的一个实例, 长度为batch_size
      # len_dec_seq: tgt step
      # inst_idx_to_position_map: {0: 0, 1: 1, ..., batch_size-1: batch_size-1}

      def prepare_beam_dec_seq(inst_dec_beams):
        # [(beam_size, 1), (beam_size, 1), ...]
        dec_seq = [b.get_last_target_word() for b in inst_dec_beams if not b.done]
        # dec_seq: (<=batch_size, beam_size), 表示每个sentence中解码出的概率最高的beam_size个单词
        # stack会创建新维度, 默认dim=0
        dec_seq = torch.stack(dec_seq).to(self.device)
        # dec_seq: (1, <=batch_size * beam_size)作为decoder端输入
        dec_seq = dec_seq.view(1, -1)
        return dec_seq

      def predict_word(dec_seq, n_active_inst, n_bm, len_dec_seq):
        # dec_seq: (1, <=batch_size * beam_size), 所有句子中前一个出现概率最高的beam_size, 平铺排开
        # n_active_inst: <=batch_size
        # n_bm: beam_size
        # len_dec_seq: tgt step

        # 如果是train过程中，那么输入是(seq_len, batch_size)
        # 现在是inference过程，输入是(1, <=batch_size * beam_size)
        # 此处的输入仅仅考虑已经预测出的tgt句子中的上一个单词(每个sentence都有beam_size个上一个单词)
        # encoder端的输出已经通过init_state保存到decoder.state中了

        # dec_output: (1, <=batch_size*beam_size, dim)
        # 1. 这里的mask是如何做的?
        # 2. 这里是如何与enc_output做context_attn的?

        '''
        train过程中的注意力过程
          1. encoder
            src_emb _> q, k, v -> self-attn + src_pad_mask(pad cols) -> ... -> enc_output
          2. decoder
            tgt_emb _> q, k, v -> self-attn + tgt_pad_mask(pad cols) + \
             tgt_feature_mask(mask对角线以上) -> ... -> k, v from (enc_output) +\
             q from tgt-self-attn -> context-attn + src_pad_mask(pad cols) -> ... -> dec_output
        '''

        dec_output, *_ = self.model.decoder(dec_seq, step=len_dec_seq)
        # word_prob: (<=batch_size * beam_size, vocab_size)
        word_prob = self.model.generator(dec_output.squeeze(0))
        # word_prob: (<=batch_size, beam_size, vocab_size)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)

        return word_prob

      def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
        # inst_beams: 一个list, 元素是Beam的一个实例, 长度为batch_size
        # word_prob: (<=batch_size, beam_size, vocab_size)
        # inst_idx_to_position_map: {0: 0, 1: 1, ..., batch_size-1: batch_size-1}
        active_inst_idx_list = []
        select_indices_array = []
        # 遍历每一个解码的句子
        for inst_idx, inst_position in inst_idx_to_position_map.items():
          # inst_beams[inst_idx]: sent, tgt_word: Beam instance
          # word_prob[inst_position]: sent, tgt_word: Beam预测结果
          is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
          # 如果inst结束，那么layer_cache中不会被选择到
          if not is_inst_complete:
            active_inst_idx_list += [inst_idx]
            # (beam_size, ) 需要加上偏移
            # 而layer_cache: (beam_size*batch_size, ... )
            select_indices_array.append(inst_beams[inst_idx].get_current_origin() + inst_position * n_bm)
        if len(select_indices_array) > 0:
          # 默认是dim=0 (<=batch_size*beam_size)
          select_indices = torch.cat(select_indices_array)
        else:
          select_indices = None
        return active_inst_idx_list, select_indices

      # n_active_inst <= batch_size, 当前正在解码的句子数量
      n_active_inst = len(inst_idx_to_position_map)

      # dec_seq: (1, <=batch_size * beam_size), 准备tgt端输入，每个句子有beam_size个同样含义的单词
      dec_seq = prepare_beam_dec_seq(inst_dec_beams)
      # (<=batch_size, beam_size, vocab_size)
      word_prob = predict_word(dec_seq, n_active_inst, n_bm, len_dec_seq)

      # Update the beam with predicted word prob information and collect incomplete instances
      active_inst_idx_list, select_indices = collect_active_inst_idx_list(
        inst_dec_beams, word_prob, inst_idx_to_position_map)

      # (<=batch_size*beam_size)
      if select_indices is not None:
        assert len(active_inst_idx_list) > 0
        # 获取下个step所需要的cache
        # src: (seq_len, <=batch_size*beam_size)
        # src_enc: (seq_len, <=batch_size*beam_size, dim)
        # layer_cache(k, v)
        # context-attn_k/v: (<=batch_size*beam_size, head_count, src_len, dim_per_head)
        # self_attn_k/v: (<=batch_size*beam_size, head_count, step?, dim_per_head)
        self.model.decoder.map_state(
            lambda state, dim: state.index_select(dim, select_indices))

      return active_inst_idx_list
    
    def collate_active_info(
        src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
      # Sentences which are still active are collected,
      # so the decoder will not run on completed sentences.
      n_prev_active_inst = len(inst_idx_to_position_map)
      active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
      active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

      active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
      active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
      active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

      return active_src_seq, active_src_enc, active_inst_idx_to_position_map

    def collect_best_hypothesis_and_score(inst_dec_beams):
      hyps, scores = [], []
      for inst_idx in range(len(inst_dec_beams)):
        hyp, score = inst_dec_beams[inst_idx].get_best_hypothesis()
        hyps.append(hyp)
        scores.append(score)
        
      return hyps, scores
    
    with torch.no_grad():
      #-- Encode
      # src_seq: (seq_len_src, batch_size)
      # 翻译的时候, 应该是排序了，长度越短的越先翻译 TODO: WHY?
      src_seq = make_features(batch, 'src')
      # src_emb: (seq_len_src, batch_size, emb_size)
      # src_enc: (seq_len_src, batch_size, emb_size)
      src_emb, src_enc, _ = self.model.encoder(src_seq)
      self.model.decoder.init_state(src_seq, src_enc)
      src_len = src_seq.size(0)
      
      #-- Repeat data for beam search
      n_bm = self.beam_size
      # TODO: 为什么需要batch_size个Beam的实例? => 每个实例对应单个sentence
      n_inst = src_seq.size(1)
      # 把同一个batch(句子)重复了beam_size次(放在相邻位置上), 确保后续decoder shape对的上, 只作用到src/src_output, 此时layer_cache还是None
      # src: (seq_len, batch_size) => (seq_len, batch_size*beam_size) -< context-attn, padding
      # src_enc_ouput: (seq_len, batch_size, dim) => (seq_len, batch_size*beam_size, dim)   <- context-attn, k与v
      self.model.decoder.map_state(lambda state, dim: tile(state, n_bm, dim=dim))

      #-- Prepare beams
      # 默认decoder_length 是 src_len + 50
      decode_length = src_len + self.decode_extra_length
      decode_min_length = 0
      if self.decode_min_length >= 0:
        decode_min_length = src_len - self.decode_min_length
      # 当前batch不会再创建Beam
      inst_dec_beams = [Beam(n_bm, decode_length=decode_length, minimal_length=decode_min_length, minimal_relative_prob=self.minimal_relative_prob, bos_id=self.tgt_bos_id, eos_id=self.tgt_eos_id, device=self.device) for _ in range(n_inst)]
      
      #-- Bookkeeping for active or not
      active_inst_idx_list = list(range(n_inst))
      # 这个map并不需要
      inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

      #-- Decode
      for len_dec_seq in range(0, decode_length):
        active_inst_idx_list = beam_decode_step(
          inst_dec_beams, len_dec_seq, inst_idx_to_position_map, n_bm)
        
        if not active_inst_idx_list:
          break  # all instances have finished their path to <EOS>

        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        
    batch_hyps, batch_scores = collect_best_hypothesis_and_score(inst_dec_beams)
    return batch_hyps, batch_scores
      
