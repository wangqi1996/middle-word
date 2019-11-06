# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.init as my_init
from src.data.vocabulary import PAD
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.modules.cgru import CGRUCell
from src.modules.embeddings import Embeddings
from src.modules.rnn import RNN
from .base import NMTModel


class Encoder(nn.Module):
    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size
                 ):
        """

        :param n_words:  词典的大小
        :param input_size:  嵌入词向量的大小
        :param hidden_size:  隐层的大小
        """
        super(Encoder, self).__init__()

        # Use PAD
        self.embeddings = Embeddings(num_embeddings=n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     add_position_embedding=False)

        self.gru = RNN(type="gru", batch_first=True, input_size=input_size, hidden_size=hidden_size,
                       bidirectional=True)

    def forward(self, x):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len]
        """

        # detach() x和x.detach()公用同一块地址，但是x.detach()不需要梯度
        # x_mask: [batch_size, seq_len] 由0和1组成，表示对应位置是否为PAD
        x_mask = x.detach().eq(PAD)

        # emb: [batch_size, seq_len, input_size]
        emb = self.embeddings(x)

        ctx, _ = self.gru(emb, x_mask)

        # ctx: [batch_size, src_seq_len, hid_dim*num_directions]
        # x_mask: [batch_size, seq_len]
        return ctx, x_mask


class Decoder(nn.Module):

    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size,
                 context_size,
                 bridge_type="mlp",
                 dropout_rate=0.0):
        """

        :param n_words:       输入词典的大小
        :param input_size:    target词典大小
        :param hidden_size:   隐层的大小
        :param context_size:  d_model*2
        :param bridge_type:
        :param dropout_rate:
        """
        super(Decoder, self).__init__()
        self.bridge_type = bridge_type
        self.hidden_size = hidden_size
        self.context_size = context_size
        # self.context_size = hidden_size * 2

        self.embeddings = Embeddings(num_embeddings=n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     add_position_embedding=False)

        self.cgru_cell = CGRUCell(input_size=input_size, hidden_size=hidden_size, context_size=context_size)
        # 将 嵌入层的输出过一个线性层
        self.linear_input = nn.Linear(in_features=input_size, out_features=input_size)
        # 将隐层输出过一个线性层
        self.linear_hidden = nn.Linear(in_features=hidden_size, out_features=input_size)
        # 将上下文向量过一个线性层
        self.linear_ctx = nn.Linear(in_features=context_size, out_features=input_size)

        self.dropout = nn.Dropout(dropout_rate)

        self._reset_parameters()

        self._build_bridge()

    def _reset_parameters(self):

        my_init.default_init(self.linear_input.weight)
        my_init.default_init(self.linear_hidden.weight)
        my_init.default_init(self.linear_ctx.weight)

    # encoder到decoder的上下文
    def _build_bridge(self):

        if self.bridge_type == "mlp":
            # 显然是将context_size的大小转变成hidden_size
            # 使用一个FFN将encoder的输出转变成decoder的隐层大小
            self.linear_bridge = nn.Linear(in_features=self.context_size, out_features=self.hidden_size)
            my_init.default_init(self.linear_bridge.weight)
        elif self.bridge_type == "zero":
            pass
        else:
            raise ValueError("Unknown bridge type {0}".format(self.bridge_type))

    def init_decoder(self, context, mask):
        """
        这里传入的时encoder的context和mask
        :param context: [batch_size, seq_len, context_size]
        :param mask: [batch_size, seq_len]
        :return:
        """
        # Generate init hidden
        if self.bridge_type == "mlp":

            no_pad_mask = 1.0 - mask.float()
            # 这个是按照seq_len维度，求平均值
            # no_pad_mask.unsqueeze(2): [batch_size, seq_len, 1]  这一步是不是python的广播呀？
            # ctx_mean: [batch_size, context_size] 按照时间维度求平均值
            ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)
            # dec_init: [batch_size, hid_dim]
            dec_init = F.tanh(self.linear_bridge(ctx_mean))

        elif self.bridge_type == "zero":
            batch_size = context.size(0)
            dec_init = context.new(batch_size, self.hidden_size).zero_()
        else:
            raise ValueError("Unknown bridge type {0}".format(self.bridge_type))

        # context: [batch_size, seq_len, context_size]
        # dec_cache: [batch_size, seq_len, context_size]
        # 经历了一个线性层,其实是计算 attention层需要的Wh
        dec_cache = self.cgru_cell.compute_cache(context)

        # dec_init: [batch_size, hid_dim]
        # dec_cache [batch_size, seq_len, context_size]
        return dec_init, dec_cache

    def forward(self, y, context, context_mask, hidden, one_step=False, cache=None):
        """
        one_step： 在推理阶段，one_step为true，一次执行一步
        :param y:
        :param one_step: True or False
        context: [batch_size, src_seq_len, context_size]
        context_mask: [batch_size, src_seq_len]
        hidden: [batch_size, hid_dim]
        cache [batch_size, src_seq_len, context_size]
        :return:
        """
        emb = self.embeddings(y)  # [batch_size, trg_seq_len, dim]
        if one_step:
            (out, attn), hidden = self.cgru_cell(emb, hidden, context, context_mask, cache)
        else:
            # emb: [batch_size, seq_len, dim]
            out = []  # [trg_seq_len, batch_size, hidden_size ]
            attn = []  # [trg_seq_len, batch_size, context_size]
            # torch.split： 不等切分，传入列表表示按照列表切分
            # 把emb切分，split_size_or_sections传入整数表示等分
            # 所以这里 按照时间 切分成一个词一个词的

            for emb_t in torch.split(emb, split_size_or_sections=1, dim=1):
                (out_t, attn_t), hidden = self.cgru_cell(emb_t.squeeze(1), hidden, context, context_mask, cache)
                # out: [batch_size, hidden_size]
                # attn_t: [batch_size, context_size])  上下文向量
                # hidden2: [batch_size, hidden_size] hidden2其实就是out_t向量
                out += [out_t]
                attn += [attn_t]
            # out: [batch_size, trg_seq_len, hidden_size]
            out = torch.stack(out).transpose(1, 0).contiguous()
            # attn: [batch_size, trg_seq_len, context_size]
            attn = torch.stack(attn).transpose(1, 0).contiguous()
        # [batch_size, seq_len, emb_dim]
        # emb: 嵌入层
        # out: 第二个decoder输出的隐层，本来是根据这个计算softmax的
        # attn: 注意力权重
        logits = self.linear_input(emb) + self.linear_hidden(out) + self.linear_ctx(attn) # 这里其实是teacher forcing

        logits = F.tanh(logits)

        logits = self.dropout(logits)  # [batch_size, seq_len, emb_dim]

        # logits: [batch_size, seq_len, emb_dim]
        # hidden: [batch_size, hidden_size]
        return logits, hidden


class Generator(nn.Module):

    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):

        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        # 投影层，将hidden_size 投影到 n_words
        self.proj = nn.Linear(self.hidden_size, self.n_words, bias=False)

        if shared_weight is not None:
            self.proj.weight = shared_weight
        else:
            self._reset_parameters()

    def _reset_parameters(self):

        my_init.embedding_init(self.proj.weight)

    def _pad_2d(self, x):
        """
        不能预测 填充字符PAD
        :param x:
        :return:
        """
        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.padding_idx] = float('-inf')
            x_2d = x_2d + mask

            return x_2d.view(x_size)

    def forward(self, input, log_probs=True):
        """
        input: [batch_size, tgt_seq_len, hidden_size]
        input == > Linear == > LogSoftmax
        """
        # 其实这里传进来的时emb_dim, 将emb_dim 转变成 d_word_vec
        # [batch_size, tgr_seq_len, vocab_size]
        logits = self.proj(input)

        #
        logits = self._pad_2d(logits)

        # return: [batch_size, tgt_seq_len,  vocab_size]
        if log_probs:
            return torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            return torch.nn.functional.softmax(logits, dim=-1)


class DL4MT(NMTModel):

    def __init__(self, n_src_vocab, n_tgt_vocab, d_word_vec=512, d_model=512, dropout=0.5,
                 proj_share_weight=False, bridge_type="mlp", **kwargs):
        """

        :param n_src_vocab:  输入的词典大小
        :param n_tgt_vocab:  输出的词典大小
        :param d_word_vec:   词向量的大小，输入词向量的大小
        :param d_model:      隐层的大小, encoder和decoder的隐层大小相同
        :param dropout:      每一层的生效神经元数目
        :param proj_share_weight:  # 共享从embedding到hidden_dim映射的权重
        :param bridge_type:
        :param kwargs:
        """
        super().__init__()

        self.encoder = Encoder(n_words=n_src_vocab, input_size=d_word_vec, hidden_size=d_model)

        # 这里的context_size是encoder的hidden_size*2
        self.decoder = Decoder(n_words=n_tgt_vocab, input_size=d_word_vec,
                               hidden_size=d_model, context_size=d_model * 2,
                               dropout_rate=dropout, bridge_type=bridge_type)

        if proj_share_weight is False:
            generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=PAD)
        else:
            generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=PAD,
                                  shared_weight=self.decoder.embeddings.embeddings.weight)
        self.generator = generator

    def forward(self, src_seq, tgt_seq, log_probs=True):
        """

        :param src_seq:   [batch_size, src_seq_len]
        :param tgt_seq:
        :param log_probs:
        :return:
        """
        # encoder的GRU是双向的，而且context_size = hid_dim*2
        # ctx: [batch_size, seq_len, hid_dim*2]
        # ctx_mask: [batch_size, seq_len]
        ctx, ctx_mask = self.encoder(src_seq)
        # dec_init: [batch_size, hid_dim]
        # dec_cache [batch_size, seq_len, context_size]
        # 将encoder的隐层变成了decoder的隐层
        dec_init, dec_cache = self.decoder.init_decoder(ctx, ctx_mask)
        # logits: [batch_size, tgt_seq_len, emb_dim]
        logits, _ = self.decoder(tgt_seq,
                                 context=ctx,
                                 context_mask=ctx_mask,
                                 one_step=False,
                                 hidden=dec_init,
                                 cache=dec_cache)  # [batch_size, tgt_len, dim]

        return self.generator(logits, log_probs)

    def encode(self, src_seq):
        # ctx: [batch_size, seq_len, hid_dim*2=context_dim]
        # ctx_mask: [batch_size, seq_len]
        ctx, ctx_mask = self.encoder(src_seq)

        return {"ctx": ctx, "ctx_mask": ctx_mask}

    def init_decoder(self, enc_outputs, expand_size=1):
        # ctx: [batch_size, src_seq_len, context_size=hid_dim*2]
        ctx = enc_outputs['ctx']

        ctx_mask = enc_outputs['ctx_mask']

        dec_init, dec_caches = self.decoder.init_decoder(context=ctx, mask=ctx_mask)

        # expand_size = beam_size
        if expand_size > 1:
            ctx = tile_batch(ctx, expand_size)
            ctx_mask = tile_batch(ctx_mask, expand_size)
            dec_init = tile_batch(dec_init, expand_size)
            dec_caches = tile_batch(dec_caches, expand_size)

        # ctx: [batch_size * beam_size, src_seq_len, context_size]
        # ctx_mask: [batch_size * beam_size, src_seq_len]
        # dec_init: [batch_size * beam_size, hid_dim]
        # dec_cache [batch_size * beam_size, seq_len, context_size]
        return {"dec_hiddens": dec_init, "dec_caches": dec_caches, "ctx": ctx, "ctx_mask": ctx_mask}

    def decode(self, tgt_seq, dec_states, log_probs=True):
        # trg_seq: 已经生成的？
        ctx = dec_states['ctx']
        ctx_mask = dec_states['ctx_mask']

        dec_hiddens = dec_states['dec_hiddens']
        dec_caches = dec_states['dec_caches']

        final_word_indices = tgt_seq[:, -1].contiguous()

        logits, next_hiddens = self.decoder(final_word_indices, hidden=dec_hiddens, context=ctx, context_mask=ctx_mask,
                                            one_step=True, cache=dec_caches)

        scores = self.generator(logits, log_probs=log_probs)

        dec_states = {"ctx": ctx, "ctx_mask": ctx_mask, "dec_hiddens": next_hiddens, "dec_caches": dec_caches}

        return scores, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):

        dec_hiddens = dec_states["dec_hiddens"]  # [batch * beam, hidden_size]

        batch_size = dec_hiddens.size(0) // beam_size

        dec_hiddens = tensor_gather_helper(gather_indices=new_beam_indices,
                                           gather_from=dec_hiddens,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, -1])

        dec_states['dec_hiddens'] = dec_hiddens

        return dec_states
