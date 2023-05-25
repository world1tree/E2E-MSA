""" Manage beam search info structure.

    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch
import math
import onmt.constants as Constants

class Beam():
  ''' Beam Search
  Beam对应一句话，有beam_size个候选
  1. 如果没有early stop, 那么解码到decode_length会立即结束, 中间遇到eos结尾，会赋予一个很低的分数，确保解码过程中不会被选到
  2. 如果是early stop, 那么解码到minimal_length后，从2*beam_size中选出两类概率最高的前beam_size个句子，其中一类
  是eos结尾的(不能保证一定能选出beam_size个eos结尾的???)(finished_seq), 另一类是非eos结尾的(alive_seq),
  如果finished_seq中最小得分大于alive_seq最大得分结束
  '''

  def __init__(self, size, decode_length, bos_id, eos_id, minimal_length=0, alpha=0.6, minimal_relative_prob=0.0, stop_early=True, device=False):
    # Beam size
    self.size = size
    self.alpha = alpha
    self.stop_early = stop_early
    self.decode_length = decode_length
    self.minimal_length = minimal_length
    self._done = False
    self.device = device
    self.minimal_score = -1 * Constants.INF
    self.eos_id = eos_id
    self.minimal_relative_log_prob = math.log(minimal_relative_prob) if minimal_relative_prob > 0 else 0.0
    # alive_seq: (beam_size, 1)
    self.alive_seq = torch.zeros((size, 1), dtype=torch.long, device=device)
    self.alive_seq[0][0] = bos_id # 仅第一个值为<bos>, 其余全部为0
    # alive_log_prob: (beam_size, ), 从大到小排序的!!! TODO: 当前单词的log_prob作为整个的log_prob
    self.alive_log_prob = torch.zeros((size,), dtype=torch.float, device=device)

    # The score for each finished translation on the beam
    # finished_seq: (beam_size, 1), 全部初始化为eos_id
    self.finished_seq = torch.zeros(self.alive_seq.size(), dtype=torch.long, device=device) + self.eos_id
    # finished_scores: (beam_size, 1), 全部初始化为-1e7
    self.finished_scores = torch.ones((size,), dtype=torch.float, device=device) * self.minimal_score
    # finished_flags: (beam_size, 1), finished_score>0 表示结束
    self.finished_flags = self.finished_scores > 0

  def is_finished(self):
    # 1. stop_early为False, 没有到decode_length不会停(遇到<eos> score会被赋值为-inf, 不会被选到)
    # 并且在解码过程中batch_size不会变!!!!!!
    # 2. stop_early为True, 没有到minimal_length不会停(到了minimal_length后并且末尾是eos
    # 认为该路径结束, 此时计算已结束路径的最小得分如果比未结束路径的最大得分要高[这两个得分都是从
    # 2*beam_size中选择的，其中一个遇到eos是赋予-inf, 另一个非eos赋予-inf(互补)]，beam search结束)
    if not self.stop_early:    # 到达decode_length立马结束
      return self.alive_seq.size(1) < self.decode_length

    max_length_penalty = math.pow((5. + self.decode_length) / 6., self.alpha)
    #max_length_penalty = 1
    # lower_bound_alive_scores: scalar(当前beam_size个路径中最大得分)
    lower_bound_alive_scores = self.alive_log_prob[0] / max_length_penalty
    # self.finished_scores: (beam_size)
    # scalar also
    lowest_score_of_fininshed_in_finished = torch.min(self.finished_scores * self.finished_flags.type(torch.float))
    # non-zero value (must be less than 0) if at least one hypothesis is finished, 
    # 0 if all hypothesis are not finished
    at_least_one_finished = torch.sum(self.finished_flags) > 0
    # 如果没有一个结束，那么是-inf
    lowest_score_of_fininshed_in_finished += (
        (1. - at_least_one_finished.type(torch.float)) * self.minimal_score)
    
    # non-zero value (must be less than 0) if at least one hypothesis is finished,
    # -inf if all hypothesis are not finished
    # 尚未结束的seq中得分中最大的得分 < 已经结束的seq中的最小得分(都是从2*beam_size中选出的beam_size再计算得到)
    self._done = torch.lt(lower_bound_alive_scores, lowest_score_of_fininshed_in_finished)
    return self.done
  
  def _compute_topk_scores_and_seq(self, sequences, scores, scores_to_gather, flags):
    # sequences: (size * 2, ? + 1)
    # scores: (size * 2,)
    # scores_to_gather(size * 2,)
    # flags(size * 2,)
    _, topk_ids = scores.topk(self.size, 0, True, True)
    # topk_ids: (size,)
    topk_seq = torch.index_select(sequences, 0, topk_ids)
    top_log_prob = torch.index_select(scores_to_gather, 0, topk_ids)
    top_flags = torch.index_select(flags, 0, topk_ids)
    return topk_seq, top_log_prob, top_flags, topk_ids
  
  
  def grow_alive(self, curr_seq, curr_scores, curr_log_prob, curr_finished):
    # curr_seq: (size * 2, ? + 1)
    # curr_scores: (size * 2, 1), 原始log加了length_penalty后的得分
    # curr_log_prob: (size * 2, 1), 原始的log得分
    # finished_flag: (size * 2,) 1 for finished and 0 for not finished
    # 如果当前预测(2*beam_size, ?+1)已经结束了，那么masked_curr_soceres会非常小(-inf), 后续选beam_size不会选到
    # 如果当前预测(2*beam_size, ?+1)没有结束，那么masked_curr_soceres为curr_scores
    masked_curr_scores = curr_scores + curr_finished.type(torch.float) * self.minimal_score
    # curr_sores: (size * 2,) -inf for finished hypothesis, 0 for not finished
    
    return self._compute_topk_scores_and_seq(curr_seq, masked_curr_scores, curr_log_prob, curr_finished)
  
  def grow_finished(self, curr_seq, curr_scores, curr_finished):
    # curr_seq: (size * 2, ? + 1)
    # curr_sores: (size * 2, 1)  原始log加了length_penalty后的得分
    # curr_finished: (size * 2,) 1 for finished and 0 for not finished, top_ids.eq(eos_id)
    # 如果结束，那么score为curr_score, 否则为-inf
    masked_curr_scores = curr_scores + (1. - curr_finished.type(torch.float)) * self.minimal_score

    # 在之前self.finished_seq基础上加入eos, 全部都是<eos>????
    # (beam_size, ?+1)
    finished_seq = torch.cat((self.finished_seq, torch.zeros((self.size, 1), dtype=torch.long, device=self.device) + self.eos_id), dim=1)
    # curr_finished_seq: (beam_size * 3, ? + 1)
    curr_finished_seq = torch.cat((finished_seq, curr_seq), dim=0)
    # curr_finished_scores: (size * 3, 1)
    curr_finished_scores = torch.cat((self.finished_scores, masked_curr_scores), dim=0)
    # curr_finished_flags: (size * 3,  1)
    curr_finished_flags = torch.cat((self.finished_flags, curr_finished), dim=0)

    # 未达到minimal_length之前, 随着step增长, 只有finished_seq多了<eos>, 其余都无变化
    if (curr_finished_seq.size(1) < self.minimal_length):
      return finished_seq, self.finished_scores, self.finished_flags, None
    else:
      # 达到最小长度后，取curr_finished_scores中最大的beam_size个, 不能保证一定有beam_size个结束
      # curr_finished_scores有两部分，其中一部分是inf, 另一部分可能有eos结束的句子(score>0)
      return self._compute_topk_scores_and_seq(curr_finished_seq, curr_finished_scores, curr_finished_scores, curr_finished_flags)
  
  def advance(self, word_prob):
    "Update beam status and check if finished or not."
    # 从decoder预测(beam_size, vocab_size)中选出2*beam_size个
    # 再根据score从2*beam_size中选出beam_size
    # alive_seq: (beam_size, ?) -> (2*beam_size, ?+1) -> (beam_size, ?+1)

    if (self.minimal_relative_log_prob != 0.0):
      top_probs, _ = word_prob.topk(1, 1, True, True)
      # top_2probs, _ = word_prob.topk(2, 1, True, True)
      # probs = top_2probs.exp()
      word_prob = word_prob + torch.lt(word_prob, top_probs * self.minimal_relative_log_prob).type(torch.float) * self.minimal_score

    # word_prob: (beam_size, vocab_size)
    if self.alive_seq.size()[1] == 1:
      # predict the first word, (vocab_size), 直接从有<bos>开头的取2*beam_size个单词
      log_probs = word_prob[0]   # from log_softmax(<0)
    else:
      # log_probs: (beam_size, vocab_size)
      log_probs = word_prob + self.alive_log_prob.view(-1, 1)

    num_words = word_prob.size(1)

    # 随着长度增长，length_penalty会变大
    length_penalty = math.pow((5. + self.alive_seq.size(1) / 6.), self.alpha)
    #length_penalty = 1
    curr_scores = log_probs / length_penalty
    # curr_scores: (vocab_size)(step=0) or (beam_size, vocab_size)(step>0)
    flat_curr_scores = curr_scores.view(-1)

    # 选取前beam_size*2得分(从大到小排序), 以及对应的index
    # TODO: 为什么需要beam_size*2? (和eos有关?)
    topk_scores, topk_ids = flat_curr_scores.topk(self.size * 2, 0, True, True)
    # topk_scores: (size * 2,)
    # topk_ids: (size * 2, )

    # topk_log_probs: (size * 2,), TODO: length_penalty的作用?
    # TODO: 为什么使用log_softmax
    topk_log_probs = topk_scores * length_penalty

    # 由于进行了view操作，现在需要知道原始(beam_size, vocab_size)对应的索引
    topk_beam_index = topk_ids // num_words  # 对应beam_size选出的index
    topk_ids %= num_words                    # 对应vocab_size选出的index
    # topk_beam_index: (beam_size * 2,)
    # topk_ids: (beam_size * 2,)

    # alive_seq: (beam_size, step+1)
    # topk_seq: (beam_size * 2, step+1)

    # 根据prob(beam_size, vocab_size), 选出前2*beam_size个
    # 从(beam_size, step+1)中选出下一个step对应的(2*beam_size, step+1)
    topk_seq = torch.index_select(self.alive_seq, 0, topk_beam_index)
    # 把当前预测的单词放入(size * 2, step+2), 到此已经根据概率大小获取并保存了概率最高的2*beam_size个单词
    # 维度变化: (beam_size, ?) => (2*beam_size, ?+1)
    topk_seq = torch.cat((topk_seq, topk_ids.view(-1, 1)), dim=1)
    # topk_seq: (size * 2, ? + 1)

    # topk_finished: (beam_size*2,), 知道那些预测是结束了的
    topk_finished = topk_ids.eq(self.eos_id)

    # self.alive_seq: (beam_size, ?+1)
    # self.alive_log_prob: (beam_size)
    self.alive_seq, self.alive_log_prob, _, top_topk_beam_index = self.grow_alive(topk_seq, topk_scores, topk_log_probs, topk_finished)
    self.finished_seq, self.finished_scores, self.finished_flags, _ = self.grow_finished(topk_seq, topk_scores, topk_finished)

    # self.prev_ks: (beam_size,)
    self.prev_ks = torch.index_select(topk_beam_index, 0, top_topk_beam_index)

    return self.is_finished()
    
  def get_current_state(self):
    "Get the outputs for the current timestep."
    return self.alive_seq
  
  def get_current_origin(self):
    "Get the backpointers for the current timestep."
    return self.prev_ks
  
  def get_last_target_word(self):
    # self.alive_seq: (beam_size, current_decode_length)
    return self.alive_seq[:, -1] # beam_size

  @property
  def done(self):
    return self._done
  
  def get_best_hypothesis(self):
    if torch.sum(self.finished_flags) > 0:
      return self.finished_seq[0, 1:].data.cpu().numpy(), self.finished_scores[0].item()
    else:
      return self.alive_seq[0, 1:].data.cpu().numpy(), self.alive_log_prob[0].item()
