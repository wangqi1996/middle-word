import math
import torch
import torch.nn as nn

from src.modules.sublayers import MultiHeadedAttention


def get_relative_position_matrix(length, max_relative_position, direction, offset=True):
    """
    Generate matrix of relative positions between inputs ([..., length]).
    :param length:
    :param max_relative_position:
    :param direction:
    :return:
    """
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
    return final_mat.cuda()


class RelativePositionEmbeddings(nn.Module):
    """
    绝对位置编码
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
            vocab_size = max_relative_position * 2 + 1
            self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=embedding_dim)
        else:
            vocab_size = max_relative_position + 1
            self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, length):
        """Generate tensor of size [length, length, depth]"""
        relative_position_matrix = get_relative_position_matrix(
            length, self.max_relative_position, self.direction
        )
        embeddings = self.embeddings(relative_position_matrix)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadedAttentionRelative(MultiHeadedAttention):
    def __init__(self, model_dim, head_count, dim_per_head=None, dropout=0.1,
                 max_relative_position=16, relative_direction=True, relative_embedding_keys=None,
                 relative_embedding_values=None):
        """
        只有decoder的自注意力用
        :param model_dim:  hidden_size
        :param head_count:
        :param dim_per_head: query_size = key_size = value_size
        :param dropout:
        """
        super().__init__(model_dim, head_count, dim_per_head, dropout)

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
        # 对于encoder： 输入三个encoder hidden
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # 1) Project key, value, and query.
        if enc_attn_cache is not None:
            key_up, value_up = enc_attn_cache
        else:  # 在inference时，有只取最后一个词的操作
            key_up = self._split_heads(self.linear_keys(key))  # [batch_size, num_head, key_len, dim_head]
            value_up = self._split_heads(self.linear_values(value))  # [batch_size, num_head, value_len, dim_head]

        # 在推理的时候会用到，一步一步的
        if self_attn_cache is not None:
            key_up_prev, value_up_prev = self_attn_cache
            # Append current key and value to the cache
            key_up = torch.cat([key_up_prev, key_up], dim=2)  # 相当于按照时间拼接吧
            value_up = torch.cat([value_up_prev, value_up], dim=2)  # [batch_size, num_head, value_len, dim]
        # inference时，query_up != key_up
        query_up = self._split_heads(self.linear_query(query))  # [batch_size, num_head, query_len, dim_head]

        key_len = key_up.size(2)
        query_len = query_up.size(2)
        value_len = value_up.size(2)

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))  # [batch_size, head_count, query_len, key_len]

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 计算value_rp view是按照行优先的
        # [query_len, batch_size*head_count, dim]
        temp_query_up = query_up.view(batch_size * head_count, query_len, dim_per_head).transpose(0, 1)
        # [query_len, key_len, dim]
        relative_key = self.relative_embedding_keys(key_len)[:query_len, :]
        # rp_scores: [query_len, batch_size*head_count, key_len]
        rp_scores = torch.matmul(temp_query_up, relative_key.transpose(-1, -2))
        rp_scores = rp_scores.transpose(0, 1).view(batch_size, head_count, query_len, key_len)
        # scores: [batch_size, head_count, query_len, key_len]
        scores = scores + rp_scores

        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores)
        drop_attn = self.dropout(attn)
        # [batch_size, head_count, query_len, dim]
        context = torch.matmul(drop_attn, value_up)  # 推出: query_len = value_len = key_len
        # [query_len, batch_size*head_count, key_len]
        temp_drop_attn = drop_attn.view(batch_size * head_count, query_len, key_len).transpose(0, 1)
        # [query_len, value_len, dim]
        relative_value = self.relative_embedding_values(value_len)[:query_len, :]
        # [query_len, batch_size*head_count, dim]
        rp_context = torch.matmul(temp_drop_attn, relative_value)
        rp_context = rp_context.transpose(0, 1).view(batch_size, head_count, query_len, self.dim_per_head)
        context = context + rp_context

        context = self._combine_heads(context)
        # [batch_size, head_count, key_len, pre_dim_head] -> [batch_size, key_len, head_count*pre_dim_head]
        # 拼接在一起，经过一个线性层
        output = self.final_linear(context)  # [batch_size, key_len, model_dim]

        # Return one attn
        top_attn = attn.view(batch_size, head_count, query_len, key_len)[:, 0, :, :].contiguous()
        # END CHECK
        return output, top_attn, [key_up, value_up]


if __name__ == "__main__":
    rel_attn = MultiHeadedAttentionRelative(head_count=4,
                                            model_dim=64,
                                            dropout=0.1,
                                            max_relative_position=16,
                                            relative_direction=True)
    batch_size, seq_len, d_model = 5, 7, 64
    max_relative_position = 16
    keys = torch.randn(batch_size, seq_len, d_model)
    values = torch.randn(batch_size, seq_len, d_model)
    query = torch.randn(batch_size, seq_len, d_model)

    mask = torch.ones(batch_size, seq_len)

    context, attn_scores, _ = rel_attn(keys, values, query)

    # or you can pass shared relative position embeddings into multiple MultiHeadedAttentionRelative instances
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
