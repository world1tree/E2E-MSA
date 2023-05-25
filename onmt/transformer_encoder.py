"""Base class for encoders and generic multi encoders."""

import torch.nn as nn
import onmt
from utils.misc import aeq
from onmt.sublayer import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout):
    super(TransformerEncoderLayer, self).__init__()
    # self attention with dropout,  linear(dropout(softmax(mask(q*k^T)))*V)
    self.self_attn = onmt.sublayer.MultiHeadedAttention(
        heads, d_model, dropout=dropout)
    self.att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    # feed_forward: linear2(dropout(relu(linear1)))
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs, mask):
    # norm1->attn->dropout->add->norm->feed_forward->add
    # (batch_size, seq_len, dim)
    input_norm = self.att_layer_norm(inputs)
    # encoder->self_attn: forward输入相同, 无layer_cache
    outputs, _ = self.self_attn(input_norm, input_norm, input_norm,
                                mask=mask)  # mask中padding必须填1 (batch_size, 1, seq_len)
    inputs = self.dropout(outputs) + inputs
    
    input_norm = self.ffn_layer_norm(inputs)
    outputs = self.feed_forward(input_norm)
    inputs = outputs + inputs
    return inputs


class TransformerEncoder(nn.Module):

  def __init__(self, num_layers, d_model, heads, d_ff,
               dropout, embeddings):
    super(TransformerEncoder, self).__init__()
    # 6
    self.num_layers = num_layers
    # 512
    self.embeddings = embeddings
    self.transformer = nn.ModuleList(
      [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])
    # 6层结束有layer_norm
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def _check_args(self, src, lengths=None):
    _, n_batch = src.size()
    if lengths is not None:
      n_batch_, = lengths.size()
      aeq(n_batch, n_batch_)

  def forward(self, src, lengths=None):
    """ See :obj:`EncoderBase.forward()`"""
    self._check_args(src, lengths)

    # (seq_len, batch_size, dim)
    emb = self.embeddings(src)

    # (batch_size, seq_len, dim)
    out = emb.transpose(0, 1).contiguous()
    # (batch_size, seq_len)
    words = src.transpose(0, 1)
    padding_idx = self.embeddings.word_padding_idx
    mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
    # Run the forward pass of every layer of the tranformer.
    for i in range(self.num_layers):
      out = self.transformer[i](out, mask)
    out = self.layer_norm(out)

    # (seq_len, batch_size, dim), (seq_len, batch_size, dim), batch_size
    return emb, out.transpose(0, 1).contiguous(), lengths

