""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn

# from onmt.utils.misc import aeq


class MultiHeadedAttention(nn.Module):
  """
  Multi-Head Attention module from
  "Attention is All You Need"
  :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

  Similar to standard `dot` attention but uses
  multiple attention distributions simulataneously
  to select relevant items.

  Args:
     head_count (int): number of parallel heads
     model_dim (int): the dimension of keys/values/queries,
         must be divisible by head_count
     dropout (float): dropout parameter
  """

  def __init__(self, head_count, model_dim, dropout=0.1):
    assert model_dim % head_count == 0
    # 512 / 8 = 64
    self.dim_per_head = model_dim // head_count
    # 512
    self.model_dim = model_dim

    super(MultiHeadedAttention, self).__init__()
    # 8
    self.head_count = head_count

    # 512, 8 * 64
    self.linear_keys = nn.Linear(model_dim,
                                 head_count * self.dim_per_head)
    # 512, 8 * 64
    self.linear_values = nn.Linear(model_dim,
                                   head_count * self.dim_per_head)
    # 512, 8 * 64
    self.linear_query = nn.Linear(model_dim,
                                  head_count * self.dim_per_head)
    self.softmax = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(dropout)
    # 512, 512
    self.final_linear = nn.Linear(model_dim, model_dim)

  def forward(self, key, value, query, mask=None,
              layer_cache=None, type=None):
    # train: mask有值, layer_cache无值
    # inference: mask无值, layer_cache有值
    """
    Compute the context vector and the attention vectors.

    Args:
       key (`FloatTensor`): set of `key_len`
            key vectors `[batch, key_len, dim]`
       value (`FloatTensor`): set of `key_len`
            value vectors `[batch, key_len, dim]`
       query (`FloatTensor`): set of `query_len`
             query vectors  `[batch, query_len, dim]`
       mask: binary mask indicating which keys have
             non-zero attention `[batch, query_len, key_len]`
    Returns:
       (`FloatTensor`, `FloatTensor`) :

       * output context vectors `[batch, query_len, dim]`
       * one of the attention vectors `[batch, query_len, key_len]`
    """
    # key: (batch, key_len, dim)
    batch_size = key.size(0)
    # 64
    dim_per_head = self.dim_per_head
    # 8
    head_count = self.head_count
    key_len = key.size(1)
    query_len = query.size(1)

    def shape(x):
      """  projection """
      # q0, q1, q2, ....
      # (batch_size, seq_len, head_count * dim_per_head)
      #     ||
      #    \||/
      # (batch_size, seq_len, head_count, dim_per_head)
      #     ||
      #    \||/
      # (batch_size, head_count, seq_len, dim_per_head)
      return x.view(batch_size, -1, head_count, dim_per_head) \
          .transpose(1, 2)

    def unshape(x):
      """  compute context """
      # (batch_size, head_count, seq_len, dim_per_head
      #     ||
      #    \||/
      # (batch_size, seq_len, head_count, dim_per_head)
      #     ||
      #    \||/
      # (batch_size, seq_len, head_count * dim_per_head)
      return x.transpose(1, 2).contiguous() \
              .view(batch_size, -1, head_count * dim_per_head)


    # encoder->self_attn: forward输入相同, 无layer_cache

    # 1) Project key, value, and query.
    # translate.py用到
    if layer_cache is not None:
      # 自注意力
      if type == "self":
          # 输入的query, key, value其实是相同的
          # (<=batch_size*beam, 1, dim)
          query, key, value = self.linear_query(query),\
                              self.linear_keys(query),\
                              self.linear_values(query)
          # (<=batch_size*beam, head_count, 1, dim_per_head)
          key = shape(key)
          value = shape(value)

          device = key.device
          if layer_cache["self_keys"] is not None:
              # 获取当前的key数组
              key = torch.cat(
                  (layer_cache["self_keys"].to(device), key),
                  dim=2)
          if layer_cache["self_values"] is not None:
              # 获取当前的value数组
              value = torch.cat(
                  (layer_cache["self_values"].to(device), value),
                  dim=2)
          # cache, 用于下个单词继续cat
          '''
          layer_cache["self_keys/values"] dim随step变化,
          query: (<=batch_size*beam, head_count, 1, dim_per_head)
          1. step = 0
          (<=batch_size*beam, head_count, 1, dim_per_head)
          2. step = 1
          (<=batch_size*beam, head_count, 2, dim_per_head)
          3. step = 2
          (<=batch_size*beam, head_count, 3, dim_per_head)
          .
          .
          .
          通过当前单词的query, 与之前所有单词的key与value计算自注意力
          (<=batch_size*beam, head_count, 1, word_cnt)
          '''
          layer_cache["self_keys"] = key
          layer_cache["self_values"] = value
      # 上下文注意力
      elif type == "context":
        # query是tgt, self-attn中的输出 (<=batch_size*beam_size, 1, dim)
        query = self.linear_query(query)
        if layer_cache["memory_keys"] is None:
          # step = 0, key与value都是encoder_output
          # (<=batch_size*beam_size, src_len, dim)
          key, value = self.linear_keys(key),\
                       self.linear_values(value)
          key = shape(key)
          value = shape(value)
          # 除了step=0时是None, 其余每次都是相同的
          layer_cache["memory_keys"] = key
          layer_cache["memory_values"] = value
        else:
          key, value = layer_cache["memory_keys"],\
                     layer_cache["memory_values"]
    else:
      key = self.linear_keys(key)
      value = self.linear_values(value)
      query = self.linear_query(query)
      # (batch_size, head_count, seq_len, dim_per_head)
      key = shape(key)
      # (batch_size, head_count, seq_len, dim_per_head)
      value = shape(value)

  # (batch_size, head_count, seq_len, dim_per_head)
    query = shape(query)

    key_len = key.size(2)
    query_len = query.size(2)

    # 2) Calculate and scale scores.
    query = query / math.sqrt(dim_per_head)
    # (batch_size, head_count, seq_len_query, seq_len_key)
    scores = torch.matmul(query, key.transpose(2, 3))
    # 如果是inference, scores的维度是(<=batch_size*beam_size, head_count, 1, 1)

    '''
    # 测试inference阶段self-attn shape变化
    # (<=batch_size*beam_size, head_count, 1, 1->2->3->...)
    if layer_cache is not None and type != "context":
        print("scores_shape: ", scores.shape)
    '''

    # 广播机制: 当输入数组的某个维度的长度为1时，沿着此维度运算时都用此维度上的第一组值
    # 此处的mask仅仅是mask掉列的维度，行的维度没有处理, 这样的话阻止了一句话中某个单词对
    # padding单词的attention计算，但是没有阻止padding单词对其余非padding单词的attention计算
    if mask is not None:
        mask = mask.unsqueeze(1)  # [B, 1, 1, T_values] (T_values会重复seq_len次)
        # mask=True的位置的值被填充-1e18, 也就是说padding必须为1
        scores = scores.masked_fill(mask, -1e18)

    # 3) Apply attention dropout and compute context vectors.
    # (batch_size, head_count, seq_len, seq_len)
    # 这样非padding单词的softmax不会落到padding单词上
    # 但是padding单词的softmax计算也不会落到padding单词上(计算似乎无意义)
    attn = self.softmax(scores)
    # 需要dropout吗? dropout是在train过程中以特定概率p对attn中的元素置0, 并且以(1/1-p)对输出进行缩放
    # inference不需要
    drop_attn = self.dropout(attn)
    # (batch_size, seq_len, head_count * dim_per_head)
    # inference阶段是: (<=batch_size*beam_size, 1, 512)
    context = unshape(torch.matmul(drop_attn, value))

    # multi-head attention 最后需要linear
    output = self.final_linear(context)

    # Return one attn
    top_attn = attn \
        .view(batch_size, head_count,
              query_len, key_len)[:, 0, :, :] \
        .contiguous()

    return output, top_attn

class PositionwiseFeedForward(nn.Module):
  """ A two-layer Feed-Forward-Network.

      Args:
          d_model (int): the size of input for the first-layer of the FFN.
          d_ff (int): the hidden layer size of the second-layer
                            of the FNN.
          dropout (float): dropout probability(0-1.0).
  """

  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    # 512, 2048
    self.w_1 = nn.Linear(d_model, d_ff)
    # 2048, 512
    self.w_2 = nn.Linear(d_ff, d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.relu = nn.ReLU()

  def forward(self, x):
    """
    Layer definition.

    Args:
        input: [ batch_size, input_len, model_dim ]


    Returns:
        output: [ batch_size, input_len, model_dim ]
    """
    inter = self.dropout_1(self.relu(self.w_1(x)))
    output = self.w_2(inter)
    return output
