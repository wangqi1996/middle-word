import torch
import torch.nn as nn
import src.utils.init as my_init
from .basic import BottleSoftmax


class GeneralAttention(nn.Module):
    """
    generate attention:
    score(a,b) = a^T*W*b
    a: decoder
    b: encoder
    """

    def __init__(self, query_size, value_size):
        super().__init__()

        self.query_size = query_size
        self.value_size = value_size

        self.W = nn.Linear(value_size, query_size, bias=False)
        self.sm = nn.Softmax(dim=-1)
        # 将value_size大小转变成query_size大小，这样做可以保证每个注意力作用相同嘛？
        # 不因为大小而改变
        self.out = nn.Linear(value_size, query_size)

        self._reset_parameters()

    def _reset_parameters(self):

        my_init.default_init(self.W.weight)
        my_init.default_init(self.out.weight)

    def forward(self, query, value, enc_cache=None, emb_cache=None, hid_cache=None, value_mask=None, inference=False):
        """
        :param query: [batch_size, query_size]
        :param value: [batch_size, value_len, value_size]
        value_mask: [batch_size, value_size]
        cache: 矩阵乘法结合律： ABC = A(BC) [batch_size, ]
        :return:
        """
        """
        当enc_cache不为None时，计算的是 encoder和hidden的attention; 必须提前计算
        当emb_cache不为None时，计算的是embedding和hidden的attention, 不需要拼接，提前计算出所有的来了, 切割好传入
        当hid_cache不为None时，计算的是hiddens和hidden的attenton，需要拼接，value只传入最近生成的即可
        """
        if not inference:
            if enc_cache is not None:
                # [batch_size, value_len, query_size]
                cache = enc_cache
            elif emb_cache is not None:
                cache = emb_cache
            else:
                cache = self.W(value[:, -1].unsqueeze(1))  # 只有在time=1时，hid_cache才会走到这个分支，很鸡肋

            if hid_cache is not None:
                cache = torch.cat((hid_cache, cache), dim=1)
        else:
            if enc_cache is not None:
                # [batch_size, value_len, query_size]
                cache = enc_cache
            else:
                cache = self.W(value[:, -1].unsqueeze(1))  # 只有在time=1时，hid_cache才会走到这个分支，很鸡肋

            if hid_cache is not None:
                cache = torch.cat((hid_cache, cache), dim=1)
            if emb_cache is not None:
                cache = torch.cat((emb_cache, cache), dim=1)

        # [batch_size, 1, query_size]
        query = query.unsqueeze(1)
        # [batch_size, 1, value_len]
        weight = torch.matmul(query, cache.transpose(1, 2))
        if value_mask is not None:
            value_mask = value_mask.unsqueeze(1).expand_as(weight)
            weight = weight.masked_fill(value_mask, -1e18)

        # [batch_size, 1, value_len]
        attn = self.sm(weight)

        # 计算上下文注意力 [batch_size, 1, value_size]
        attn_context = torch.matmul(attn, value)

        # [batch_size, 1, query_size]
        attn_out = self.out(attn_context)

        # [batch_size, query_size]
        return attn_out.squeeze(1), cache

    def compute_cache(self, value):
        # [batch_size, query_size, src_seq_len]
        return self.W(value)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention
     score(a,b) = ab
     '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=1)

    def forward(self, q, k, v, attn_mask=None):
        """
        :type attn_mask: torch.FloatTensor
        :param attn_mask: Mask of the attention.
            3D tensor with shape [batch_size, time_step_key, time_step_value]
        """
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())
            attn = attn.masked_fill(attn_mask, -1e18)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class BahdanauAttention(nn.Module):

    def __init__(self, query_size, key_size, hidden_size=None):
        super().__init__()
        # 在decoder中使用的时候，query_size=1024(hidden_size), key_size=2048(context_size=2*hid_dim)
        self.query_size = query_size  # hidden_size
        self.key_size = key_size  # context_size

        if hidden_size is None:
            hidden_size = key_size

        self.hidden_size = hidden_size

        self.linear_key = nn.Linear(in_features=self.key_size, out_features=self.hidden_size)
        # query_size -> hidden_size
        self.linear_query = nn.Linear(in_features=self.query_size, out_features=self.hidden_size)
        self.linear_logit = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.softmax = BottleSoftmax(dim=1)
        self.tanh = nn.Tanh()

        self._reset_parameters()

    def _reset_parameters(self):
        for weight in self.parameters():
            my_init.default_init(weight)

    def compute_cache(self, memory):
        # memory: [*, key_size]
        # return: [*, key_size]
        # 经历了一个线性层？
        return self.linear_key(memory)

    def forward(self, query, memory, cache=None, mask=None):
        """
        self.attn(query=hidden1, memory=context, cache=cache, mask=context_mask)

        :param query: Key tensor. 传入的是decoder的hidden, 所以query_size=hidden_size
            with shape [batch_size, query_size]

        :param memory: Memory tensor. encoder的context
            with shape [batch_size, mem_len, hidden_size]

        :param mask: Memory mask which the PAD position is marked with true.
            with shape [batch_size, mem_len]

            cache: 提前计算出来的Wh, [batch_size, men_len, hidden_size]
        """
        # one_step = True
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch, 1, query_size]
            one_step = True
        else:
            one_step = False

        batch_size, q_len, q_size = query.size()
        _, m_len, m_size = memory.size()
        # q: [batch_size, q_len=1, q_size=query_size] q=[batch_size*q_len, q_size]
        q = self.linear_query(query.view(-1, q_size))  # [batch_size, q_len, hidden_size]

        if cache is not None:
            k = cache  # Wh
        else:
            # memory: [batch_size*mem_len, m_size]
            # k: [batch_size*men_len, m_size]
            k = self.linear_key(memory.view(-1, m_size))  # [batch_size, m_len, hidden_size]

        # logit = q.unsqueeze(0) + k # [mem_len, batch_size, dim]
        # logits: [batch_size, q_len, mem_len, hidden_size] : w*h+u*s_t
        logits = q.view(batch_size, q_len, 1, -1) + k.view(batch_size, 1, m_len, -1)
        logits = self.tanh(logits)
        # logits: [batch_size, q_len, m_len]: V^T * tanh(W*h + U*s_t)
        logits = self.linear_logit(logits.view(-1, self.hidden_size)).view(batch_size, q_len, m_len)

        if mask is not None:
            mask_ = mask.unsqueeze(1)  # [batch_size, 1, m_len]
            logits = logits.masked_fill(mask_, -1e18)
        # weights: 表示当前翻译的词对encoder的关注度的大小
        weights = self.softmax(logits)  # [batch_size, q_len, m_len]

        # [batch_size, q_len, m_len] @ [batch_size, m_len, m_size]
        # ==> [batch_size, q_len, m_size]  代表encoder的上下文向量
        attns = torch.bmm(weights, memory)

        if one_step:
            attns = attns.squeeze(1)  # ==> [batch_size, m_size]

        return attns, weights
