import torch.nn as nn

from src.utils import init as my_init
from .attention import BahdanauAttention


class CGRUCell(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 context_size):
        super(CGRUCell, self).__init__()

        self.hidden_size = hidden_size
        self.context_size = context_size
        self.attn = BahdanauAttention(query_size=hidden_size, key_size=self.context_size)
        self.linear_emb = nn.Linear(in_features=input_size, out_features=input_size)
        self.linear_attn = nn.Linear(in_features=context_size, out_features=input_size)
        self.gru1 = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for weight in self.gru1.parameters():
            my_init.rnn_init(weight)

    def forward(self,
                input,
                hidden,
                context,
                context_mask=None,
                cache=None):
        """
        input: [batch_size, input_size] 一个时间步的嵌入输入emb_t
        context: [batch_size, src_seq_len, context_size]
        context_mask: [batch_size, src_seq_len]
        hidden: [batch_size, hid_dim]
        cache [batch_size, src_seq_len, context_size]
        :return:
        """
        # hidden1：[batch_size, hidden_size]
        attn_values, _ = self.attn(query=hidden, memory=context, cache=cache, mask=context_mask)

        input = self.linear_emb(input) + self.linear_attn(attn_values)

        hidden1 = self.gru1(input, hidden)

        return (hidden1, attn_values), hidden1

    def compute_cache(self, memory):
        # memory: [ *, context_size]
        # return [ *, context_size]
        # 经历了一个线性层
        return self.attn.compute_cache(memory)
