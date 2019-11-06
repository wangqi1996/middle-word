import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.positional_embeddings import RelativePositionEmbeddings

class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dim, head_count, dim_per_head=None, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        if dim_per_head is None:
            assert model_dim % head_count == 0
            dim_per_head = model_dim // head_count

        self.head_count = head_count
        self.dim_per_head = dim_per_head
        self.model_dim = model_dim

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.dim_per_head * head_count, model_dim)

    def _split_heads(self, x):
        batch_size = x.size(0)

        return x.view(batch_size, -1, self.head_count, self.dim_per_head) \
            .transpose(1, 2).contiguous()

    def _combine_heads(self, x):
        """:param x: [batch_size * head_count, seq_len, dim_per_head]"""
        seq_len = x.size(2)

        return x.transpose(1, 2).contiguous() \
            .view(-1, seq_len, self.head_count * self.dim_per_head)

    def forward(self, key, value, query, mask=None, enc_attn_cache=None, self_attn_cache=None):
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

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # 1) Project key, value, and query.
        if enc_attn_cache is not None:
            key_up, value_up = enc_attn_cache
        else:
            key_up = self._split_heads(self.linear_keys(key))  # [batch_size, num_head, seq_len, dim_head]
            value_up = self._split_heads(self.linear_values(value))

        if self_attn_cache is not None:
            key_up_prev, value_up_prev = self_attn_cache
            # Append current key and value to the cache
            key_up = torch.cat([key_up_prev, key_up], dim=2)
            value_up = torch.cat([value_up_prev, value_up], dim=2)

        query_up = self._split_heads(self.linear_query(query))

        key_len = key_up.size(2)
        query_len = query_up.size(2)

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores)
        drop_attn = self.dropout(attn)
        context = self._combine_heads(torch.matmul(drop_attn, value_up))

        output = self.final_linear(context)

        # Return one attn
        top_attn = attn \
                       .view(batch_size, head_count,
                             query_len, key_len)[:, 0, :, :] \
            .contiguous()
        # END CHECK
        return output, top_attn, [key_up, value_up]


def _dot_product_attention_inner_relative(x, y, z, transpose):
    """Relative position-aware dot-product attention inner calculation.

    This batches matrix multiply calculations to avoid unnecessary broadcasting.

    Args:
      x: Tensor with shape [batch_size, heads, length, length or depth].
      y: Tensor with shape [batch_size, heads, length, depth].
      z: Tensor with shape [length, length, depth], the Relative Position Representation.
      transpose: Whether to transpose inner matrices of y and z. Should be true if
          last dimension of x is depth, not length.

    Returns:
      A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size, heads, length, _ = x.size()
    print(batch_size)
    # xy_matmul is [batch_size, heads, length, length or depth]
    xy_matmul = torch.matmul(x, y if not transpose else y.transpose(-2, -1))
    # x_t is [length, batch_size, heads, length or depth]
    x_t = x.permute(2, 0, 1, 3)
    # x_t_r is [length, batch_size * heads, length or depth]
    x_t_r = x_t.view(length, batch_size*heads, -1)
    # x_tz_matmul is [length, batch_size * heads, length or depth]
    x_tz_matmul = torch.matmul(x_t_r, z if not transpose else z.transpose(-2, -1))
    # x_tz_matmul_r is [length, batch_size, heads, length or depth]
    x_tz_matmul_r = x_tz_matmul.view(length, batch_size, heads, -1)
    # x_tz_matmul_r_t is [batch_size, heads, length, length or depth]
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)

    return xy_matmul + x_tz_matmul_r_t


def dot_product_attention_relative(q,
                                   k,
                                   v,
                                   bias,
                                   relative_embedding_keys,
                                   relative_embedding_values,
                                   dropout=None):
    """Calculate relative position-aware dot-product self-attention.
      The attention calculation is augmented with learned representations for the
      relative position between each element in q and each element in k and v.
      Args:
        q: a Tensor with shape [batch, heads, length, depth].
        k: a Tensor with shape [batch, heads, length, depth].
        v: a Tensor with shape [batch, heads, length, depth].
        bias: bias Tensor.
        relative_embedding_keys: a Tensor with shape [length, length, depth].
        relative_embedding_values: a Tensor with shape [length, length, depth].
        dropout (optional): nn.Dropout.

      Returns:
        A Tensor.
    """
    # Compute self attention considering the relative position embeddings.
    logits = _dot_product_attention_inner_relative(q, k, relative_embedding_keys, True)
    if bias is not None:
        logits += bias
    # [batch, heads, length, length]
    weights = F.softmax(logits, -1)

    if dropout is not None:
        weights = dropout(weights)
    return weights, _dot_product_attention_inner_relative(weights, v, relative_embedding_values, False)


class MultiHeadedAttentionRelative(MultiHeadedAttention):
    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_position=16, relative_direction=True,
                 relative_embedding_keys=None, relative_embedding_values=None):
        super().__init__(model_dim, head_count, dropout=dropout)

        if relative_embedding_keys is not None:
            self.relative_embedding_keys = relative_embedding_keys
        else:
            self.relative_embedding_keys = RelativePositionEmbeddings(
                max_relative_position,
                self.dim_per_head,
                dropout=dropout,
                direction=relative_direction) 
        if relative_embedding_values is not None:
            self.relative_embedding_values = relative_embedding_values
        else:
            self.relative_embedding_values = RelativePositionEmbeddings(
                max_relative_position,
                self.dim_per_head,
                dropout=dropout,
                direction=relative_direction) 

    def forward(self, key, value, query, mask=None, enc_attn_cache=None, self_attn_cache=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # 1) Project key, value, and query.
        if enc_attn_cache is not None:
            key_up, value_up = enc_attn_cache
        else:
            key_up = self._split_heads(self.linear_keys(key)) # [batch_size, num_head, seq_len, dim_head]
            value_up = self._split_heads(self.linear_values(value))

        if self_attn_cache is not None:
            key_up_prev, value_up_prev = self_attn_cache
            # Append current key and value to the cache
            key_up = torch.cat([key_up_prev, key_up], dim=2)
            value_up = torch.cat([value_up_prev, value_up], dim=2)

        query_up = self._split_heads(self.linear_query(query))
        query_up = query_up / math.sqrt(dim_per_head)

        key_len = key_up.size(2)
        query_len = query_up.size(2)

        # 2) Calculate and scale scores.
        if mask is not None:
            mask = mask.unsqueeze(1).expand(
                batch_size, head_count, query_len, key_len
            )
            bias = mask.masked_fill(mask, -1e18)
        else:
            bias = None

        # do attention
        attn, context = dot_product_attention_relative(
            query_up,
            key_up,
            value_up,
            bias,
            self.relative_embedding_keys,
            self.relative_embedding_values
        )

        # 3) Apply attention dropout and compute context vectors.
        # context ([batch, length, d_model])
        context = self._combine_heads(context)

        output = self.final_linear(context)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len).mean(1) \
            .contiguous()
        # END CHECK
        return output, top_attn, [key_up, value_up]


if __name__ == "__main__":
    rel_attn = MultiHeadedAttentionRelative(head_count=4,
                                            model_dim=64,
                                            dropout=0.1,
                                            max_relative_position=16,
                                            relative_direction=True)
    batch_size, seq_len, d_model = 5, 7, 64
    keys = torch.randn(batch_size, seq_len+1, d_model)
    values = torch.randn(batch_size, seq_len+1, d_model)
    query = torch.randn(batch_size, seq_len, d_model)

    mask = torch.ones(batch_size, seq_len)
    print(mask)
    context, attn_scores, cache_keys, cache_values = rel_attn(keys, values, query)

    # max_relative_position=16
    # # or you can pass shared relative position embeddings into multiple MultiHeadedAttentionRelative instances
    # relative_embedding_keys = RelativePositionEmbeddings(
    #     max_relative_position,
    #     d_model // 4,
    #     dropout=0.1,
    #     direction=True
    # )
    # relative_embedding_values = RelativePositionEmbeddings(
    #     max_relative_position,
    #     d_model // 4,
    #     dropout=0.1,
    #     direction=True
    # )
    #
    # rel_attn1 = MultiHeadedAttentionRelative(head_count=4,
    #                                          model_dim=64,
    #                                          dropout=0.1,
    #                                          relative_embedding_keys=relative_embedding_keys,
    #                                          relative_embedding_values=relative_embedding_values)
    #
    # rel_attn2 = MultiHeadedAttentionRelative(head_count=4,
    #                                          model_dim=64,
    #                                          dropout=0.1,
    #                                          relative_embedding_keys=relative_embedding_keys,
    #                                          relative_embedding_values=relative_embedding_values)
