# -*- coding: UTF-8 -*- 

# Copyright 2018, Natural Language Processing Group, Nanjing University, 
#
#       Author: Zheng Zaixiang
#       Contact: zhengzx@nlp.nju.edu.cn 
#           or zhengzx.142857@gmail.com
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

from torch.nn import functional as F


def get_relative_position_matrix(length, max_relative_position, direction, offset=True):
    """ Generate matrix of relative positions between inputs ([..., length])."""
    range_vec = torch.arange(length).long()
    # if torch.cuda.is_available():
    #     range_vec = range_vec.cuda()
    range_mat = range_vec[:, None].expand(length, length)
    distance_mat = range_mat - range_mat.transpose(0, 1)
    if max_relative_position is None:
        distance_mat_clipped = distance_mat
    else:
        distance_mat_clipped = torch.clamp(distance_mat,
                                           -max_relative_position, max_relative_position)
    if direction:
        # Shift values to be >= 0. Each integer still uniquely identifies a relative
        # position difference.
        if offset and max_relative_position is not None:
            final_mat = distance_mat_clipped + max_relative_position
        else:
            final_mat = distance_mat_clipped
    else:
        # Do not distinguish the forward and backward positions.
        # Just leave the absolute relative position representation.
        final_mat = distance_mat_clipped.abs()
    return final_mat


class RelativePositionEmbeddings(nn.Module):
    """ Relative Position Representation in "Self-Attention with Relative Position Representations"
        (https://arxiv.org/pdf/1803.02155.pdf)
    Implementation inspired by
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    def __init__(self,
                 max_relative_position,
                 embedding_dim,
                 dropout=0.0,
                 direction=True, **params):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.embedding_dim = embedding_dim
        self.direction = direction
        if self.direction:
            vocab_size = max_relative_position*2 + 1
            self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=embedding_dim)
        else:
            vocab_size = max_relative_position + 1
            self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # self.reset_parameters()

    def forward(self, length):
        """Generate tensor of size [length, length, depth]"""
        relative_position_matrix = get_relative_position_matrix(
            length, self.max_relative_position, self.direction
        )
        embeddings = self.embeddings(relative_position_matrix)
        embeddings = self.dropout(embeddings)
        return embeddings
