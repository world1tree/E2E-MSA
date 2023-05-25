# -*- coding: utf-8 -*-

import torch


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    x.shape: 2(seq_len, batch_size) or 3(seq_len, batch_size, dim)
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)

if __name__ == '__main__':
    # 把同一个batch重复了count次
    x_dim_2 = torch.arange(12).reshape(3, 4) # (seq_len, batch_size) => (seq_len, batch_size*count)
    output = tile(x_dim_2, 3, 1)
    print(output)
    #
    x_dim_3 = torch.arange(24).reshape(3, 4, 2) # (seq_len, batch_size, dim) => (seq_len, batch_size*count, dim)
    output = tile(x_dim_3, 3, 1)
    print(output)
