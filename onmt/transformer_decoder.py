"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

import onmt
from onmt.sublayer import PositionwiseFeedForward

MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout):
    super(TransformerDecoderLayer, self).__init__()

    # 自注意力
    self.self_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
    # 上下文注意力
    self.context_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
    # feed-forward
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 3个layr_norm
    self.self_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.enc_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    self.dropout = dropout
    self.drop = nn.Dropout(dropout)
    mask = self._get_attn_subsequent_mask(MAX_SIZE)
    # Register self.mask as a buffer in TransformerDecoderLayer, so
    # it gets TransformerDecoderLayer's cuda behavior automatically.
    # 也就是如果不用register_buffer, 可能会报设备不一致的错
    self.register_buffer('mask', mask)

  def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
              layer_cache=None, step=None):
    # inputs(tgt_emb(包含位置编码)): (batch_size, seq_len, dim)
    # memory_bank(encoder_output): (batch_size, seq_len, dim)

    # mask的意义是确保padding位置不会有attention(某个单词softmax后padding位置不可能有值)
    dec_mask = None
    if step is None:
      # 除了padding之外，还需要把上三角抹掉，因为tgt注意力不可见
      # tgt_pad_mask: [B, 1, T_tgt], 1扩展会重复后面维度的内容
      # self.mask: [1, T_tgt, T_tgt]
      dec_mask = torch.gt(tgt_pad_mask, 0)
      # dec_mask = torch.gt(tgt_pad_mask +
      #                     self.mask[:, :tgt_pad_mask.size(-1),
      #                               :tgt_pad_mask.size(-1)], 0)

    # 在inference阶段tgt只有一个单词，为什么self-attn不需要tgt_pad_mask和feature_mask
    # 因为可以确保当前没有padding, 不需要feature_mask是因为根本没有feature的注意力

    # norm1->attn->dropout->add->norm->context-attn->dropout->add->norm->feed_forward->add
    # do self attention
    input_norm = self.self_att_layer_norm(inputs)
    query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=dec_mask,
                                 layer_cache=layer_cache,
                                 type="self")
    # (batch_size, seq_len, dim)
    query = self.drop(query) + inputs

    # do encoding output attention
    query_norm = self.enc_att_layer_norm(query)
    # decoder端 tgt norm1->attn->dropout->add->norm后是作为query参与
    # 来自encoder端的key与value之间的attention计算
    # 所以这里的mask是针对src
    # self-attn是在做src与src(完全可见)之间的注意力(src_mask(padding))、
    # tgt与tgt(只能见到该单词之前的单词)之间的注意力(tgt_mask(padding)+feature_mask(下三角可见))
    # context_attn相当于是在做tgt与src之间的注意力(src_mask(padding), 确保tgt端的单词不会关注到padding单词)
    # 其中query来自decoder端self-attn的输出, 维度是(batch_size, seq_len, dim)
    #    key与value相同，均是encoder端的输出
    mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                  mask=src_pad_mask,
                                  layer_cache=layer_cache,
                                  type="context")
    mid = self.drop(mid) + query
    
    # do ffn
    mid_norm = self.ffn_layer_norm(mid)
    output = self.feed_forward(mid_norm)
    output = self.drop(output) + mid

    # decoder_output: (batch_size, seq_len, dim)
    # 只返回某个head的注意力
    # context-attn: (batch_size, **seq_len_tgt**, seq_len_src)
    return output, attn

  def _get_attn_subsequent_mask(self, size):
    # 获取上三角矩阵(主对角线也为0), 值为1的attention需要被mask掉
    attn_shape = (1, size, size)
    # np.triu(k=1) 矩阵下三角包括对角线置0,
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


class TransformerDecoder(nn.Module):
  def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
    super(TransformerDecoder, self).__init__()

    # Basic attributes.
    self.decoder_type = 'transformer'
    self.num_layers = num_layers
    self.embeddings = embeddings
    pad_token = self.embeddings.tokenizer.pad_token
    self.padding_idx = self.embeddings.tokenizer.convert_tokens_to_ids(pad_token)

    # Decoder State
    self.state = {}

    # Build TransformerDecoder.
    self.transformer_layers = nn.ModuleList(
      [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])

    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    self.linear_layer = nn.Linear(1024, 512)
    self.act = nn.Tanh()

  def init_state(self, src, src_enc):
    """ Init decoder state """
    # encoder端的输入和输出
    self.state["src"] = src        # (seq_len, batch_size)
    self.state["src_enc"] = src_enc # encoder_output: (seq_len, batch_size, dim)
    self.state["cache"] = None

  def map_state(self, fn):
    def _recursive_map(struct, batch_dim=0):
      for k, v in struct.items():
        if v is not None:
          if isinstance(v, dict):
            _recursive_map(v)
          else:
            struct[k] = fn(v, batch_dim)

    # src与src_enc保存的是encoder端的输出，不会变化
    self.state["src"] = fn(self.state["src"], 1)             # (seq_len, batch_size)
    self.state["src_enc"] = fn(self.state["src_enc"], 1)     # (seq_len, batch_size, dim)
    # cache内容随着step往后，shape不断增长
    if self.state["cache"] is not None:                      # context-attn: (<=batch_size*beam_size, head_count, src_len, dim_per_head)
      _recursive_map(self.state["cache"])                    # self-attn: (<=batch_size*beam_size, head_count, step+1, dim_per_head)

  def detach_state(self):
    # TODO 在哪调用的?
    self.state["src"] = self.state["src"].detach()

  def forward(self, tgt, step=None):
    """
    See :obj:`onmt.modules.RNNDecoderBase.forward()`
    """
    # step只要存在，那么seq_len即tgt.shape[0]一定是1,每次只有上一个单词参与decoder过程
    if step == 0:
      # inference阶段 step=0
      # tgt: (1, <=batch_size*beam_size), 如果某些句子提前done, 那么batch_size会减少
      # [[<bos_id>, 0, 0, ...,  # beam_size
      #   <bos_id>, 0, 0, ...
      #   <bos_id>, 0, 0, ...]]
      # TODO: 为什么不全部设置成<bos_id>?
      self._init_cache(self.num_layers)

    src = self.state["src"]
    src_memory_bank = self.state["src_enc"]

    # Initialize return variables.
    attns = {"std": []}

    attn_mask = (~tgt.data.eq(self.padding_idx)).to(torch.long)
    emb = self.embeddings(input_ids=tgt,attention_mask=attn_mask)
    # (B, T, 1024), 已经有位置信息
    emb = emb.last_hidden_state
    emb = self.act(self.linear_layer(emb))
    output = emb

    src_pad_mask = src.data.eq(self.padding_idx).unsqueeze(1)  # [B, 1, T_src]
    # inference: 如果当前单词是padding, 被mask
    tgt_pad_mask = tgt.data.eq(self.padding_idx).unsqueeze(1)  # [B, 1, T_tgt]

    for i in range(self.num_layers):
      output, attn = self.transformer_layers[i](
        output,  # tgt_emb
        src_memory_bank, # encoder_output
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=(
          self.state["cache"]["layer_{}".format(i)]
          if step is not None else None),
        step=step)

    output = self.layer_norm(output)

    # Process the result and update the attentions.
    # dec_outs = output.transpose(0, 1).contiguous()
    # attn = attn.transpose(0, 1).contiguous()

    attns["std"] = attn

    # TODO change the way attns is returned dict => list or tuple (onnx)
    return output, attns

  def _init_cache(self, num_layers):
    self.state["cache"] = {}

    for l in range(num_layers):
      # (<=batch_size*beam_size, head_count, src_len, dim_per_head)
      layer_cache = {
        "memory_keys": None, # inference阶段，已经解码出的单词对应的context-attn, keys
        "memory_values": None # inference阶段，已经解码出的单词对应的context-attn, values
      }
      # (<=batch_size*beam_size, head_count, step+1, dim_per_head)
      layer_cache["self_keys"] = None # inference阶段，已经解码出的单词对应的self-attn, keys
      layer_cache["self_values"] = None # inference阶段，已经解码出的单词对应的self-attn, values
      self.state["cache"]["layer_{}".format(l)] = layer_cache
