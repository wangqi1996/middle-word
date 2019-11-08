# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch import nn

import src.utils.init as my_init
from src.data.vocabulary import PAD, BOS, EOS
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.models.dl4mt import Encoder, Generator
from src.modules.attention import BahdanauAttention
from src.modules.embeddings import Embeddings


class Decoder(nn.Module):

    def __init__(self, hidden_size, context_size, n_words, input_size, dropout=0.0, proj_share_weight=False,
                 bridge_type='mlp'):

        super().__init__()

        self.vocab_size = n_words
        self.bridge_type = bridge_type
        self.context_size = context_size
        self.hidden_size = hidden_size

        self.middle_classifier = nn.Linear(context_size, input_size)

        self.embeddings = Embeddings(num_embeddings=n_words, embedding_dim=input_size, dropout=dropout,
                                     add_position_embedding=False)

        self.enc_attn = BahdanauAttention(query_size=hidden_size, key_size=context_size)

        self.rnn_right = nn.GRU(batch_first=True, input_size=context_size,
                                hidden_size=hidden_size, dropout=dropout, bidirectional=False)
        self.rnn_left = nn.GRU(batch_first=True, input_size=context_size,
                               hidden_size=hidden_size, dropout=dropout, bidirectional=False)
        # 权重共享

        self.linear_out = nn.Linear(in_features=hidden_size, out_features=input_size)

        self.linear_emb = nn.Linear(in_features=input_size, out_features=input_size)

        if proj_share_weight is False:
            generator = Generator(n_words=n_words, hidden_size=input_size, padding_idx=PAD)
        else:
            generator = Generator(n_words=n_words, hidden_size=input_size, padding_idx=PAD,
                                  shared_weight=self.embeddings.embeddings.weight)
        self.generator = generator

        self._build_bridge()
        self._reset_parameters()

    def _reset_parameters(self):
        my_init.default_init(self.linear_out.weight)
        my_init.default_init(self.linear_emb.weight)
        my_init.default_init(self.middle_classifier.weight)
        for weight in self.rnn_right.parameters():
            my_init.rnn_init(weight)

        for weight in self.rnn_left.parameters():
            my_init.rnn_init(weight)

    def _build_bridge(self):

        self.linear_bridge = nn.Linear(in_features=self.context_size, out_features=self.hidden_size)
        my_init.default_init(self.linear_bridge.weight)

    def init_decoder(self, context, mask):

        no_pad_mask = 1.0 - mask.float()

        ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)

        dec_init = F.tanh(self.linear_bridge(ctx_mean))

        enc_cache = self.enc_attn.compute_cache(context)

        return dec_init, enc_cache

    def get_middle_word(self, trg, middle_ctx, inference=False):

        batch_size = middle_ctx.shape[0]

        middle_output = F.tanh(self.middle_classifier(middle_ctx))

        if not inference:
            # 处理成left_sequences 和 right_sequence
            middle_index = [1] * batch_size
            real_middle_words = trg.new(batch_size).cuda()
            for i in range(batch_size):
                real_middle_words[i] = trg[i][1]

        else:
            middle_prob = self.generator(middle_output)
            real_middle_words = torch.argmax(middle_prob, dim=-1)
            middle_index = None

        return middle_output, real_middle_words, middle_index

    def inference(self, right_next, hiddens, embeddings, ctx, ctx_mask, enc_cache):
        """
        推理阶段使用
        :param right_next:
        :param hiddens:
        :param embeddings:
        :param ctx:
        :param ctx_mask:
        :param enc_cache:
        :param trg_mask:
        :param emb_cache:
        :param hid_cache:
        :return:
        """
        if hiddens.shape[1] == 1:
            pre_hidden = hiddens[:, 0]
        else:
            pre_hidden = hiddens[:, -2].contiguous()

        if embeddings.shape[1] == 1:
            pre_emb = embeddings[:, 0]
        else:
            pre_emb = embeddings[:, -2].contiguous()

        hidden, output = self.one_step(pre_hidden, ctx, ctx_mask, pre_emb, enc_cache, inference=True,
                                       right_next=right_next)

        hiddens = torch.cat((hiddens, hidden.unsqueeze(1)), dim=1)
        logits = self.generator(output, log_probs=True)
        return hiddens, logits

    # 单步执行这个函数
    def one_step(self, hidden, ctx, ctx_mask, pre_emb, enc_cache, inference=False, right_next=True):

        enc_context, _ = self.enc_attn(query=hidden, memory=ctx, cache=enc_cache, mask=ctx_mask)

        if right_next:
            rnn_output, hidden = self.rnn_right(enc_context.unsqueeze(1), hidden.unsqueeze(0))
        else:
            rnn_output, hidden = self.rnn_left(enc_context.unsqueeze(1), hidden.unsqueeze(0))

        hidden = hidden.squeeze(0)

        # 再次使用teacher_forcing
        if inference:
            output = self.linear_out(rnn_output.squeeze(1)) + self.linear_emb(pre_emb)
        else:
            # 统一加
            output = self.linear_out(rnn_output.squeeze(1))

        return hidden, output

    def forward(self, ctx, ctx_mask, embeddings, hiddens, enc_cache, outputs):
        """

        """
        forward_hidden = hiddens[:, 0, :]
        backward_hidden = hiddens[:, 0, :]
        forward_emb = embeddings[:, 0, :]
        backward_emb = embeddings[:, 0, :]

        seq_len = embeddings.shape[1]
        right_next = True
        for t in range(1, seq_len):

            if right_next:
                forward_hidden, output = self.one_step(forward_hidden, ctx, ctx_mask, forward_emb, enc_cache,
                                                       inference=False, right_next=right_next)
                forward_emb = embeddings[:, t]
                outputs.append(output)
            else:
                backward_hidden, output = self.one_step(backward_hidden, ctx, ctx_mask, backward_emb, enc_cache,
                                                        inference=False, right_next=right_next)
                backward_emb = embeddings[:, t]
                outputs.append(output)
            right_next = not right_next

        outputs = torch.stack(outputs, dim=1)
        outputs[:, 1:] = outputs[:, 1:] + self.linear_emb(embeddings[:, :-1])
        outputs = self.generator(outputs)
        return outputs


class Middle(nn.Module):
    """
    模型整体架构
    """

    def __init__(self, n_src_vocab, d_word_vec, d_model, n_tgt_vocab, proj_share_weight=False,
                 dropout=0.1, bridge_type='mlp', n_layers=1, **kwargs):
        super().__init__()

        self.input_size = d_word_vec
        self.hidden_size = d_model

        self.encoder = Encoder(n_src_vocab, d_word_vec, d_model)

        self.decoder = Decoder(d_model, d_model * 2, n_tgt_vocab, d_word_vec, dropout=dropout,
                               proj_share_weight=proj_share_weight, bridge_type=bridge_type)

        # self.classifier = Classifier(enc_hid_dim, decoder_vocab_size)

    def encode(self, src_seq):
        # ctx: [batch_size, seq_len, hid_dim*2=context_dim]
        # ctx_mask: [batch_size, seq_len]
        ctx, ctx_mask = self.encoder(src_seq)

        return {"ctx": ctx, "ctx_mask": ctx_mask}

    def init_decoder(self, enc_outputs, expand_size=1):
        """
        beam_search时调用, 多beam相当于增大了batch_size
        :param enc_outputs:
        :param expands_size:
        :return:
        """
        ctx = enc_outputs['ctx']
        ctx_mask = enc_outputs['ctx_mask']

        dec_init, enc_cache = self.decoder.init_decoder(ctx, ctx_mask)

        middle_ctx, _ = self.decoder.enc_attn(query=dec_init, memory=ctx, cache=enc_cache, mask=ctx_mask)
        middle_output, middle_words, _ = self.decoder.get_middle_word(None, middle_ctx, inference=True)

        embed = self.decoder.embeddings(middle_words)

        trg_mask = middle_words.detach().eq(0)

        if expand_size > 1:
            # 同一个batch的几个beam_size会靠着
            ctx = tile_batch(ctx, expand_size)
            ctx_mask = tile_batch(ctx_mask, expand_size)
            dec_init = tile_batch(dec_init, expand_size)
            trg_mask = tile_batch(trg_mask, expand_size)
            embed = tile_batch(embed, expand_size)
            enc_cache = tile_batch(enc_cache, expand_size)

        # [batch_size, trg_seq_len, dec_hid_dim]
        hiddens = dec_init.unsqueeze(1)

        # [batch_size, trg_seq_len, vocab_size]
        embeddings = embed.unsqueeze(1)

        trg_mask = trg_mask.unsqueeze(1)

        result = {
            "ctx": ctx,
            "ctx_mask": ctx_mask,
            "hiddens": hiddens,
            "embeddings": embeddings,
            "trg_mask": trg_mask,
            "right_next": True,
            "enc_cache": enc_cache,
            "middle_words": middle_words,
        }

        return result

    def init_decoder_when_train(self, ctx, ctx_mask, trg, y_len):

        batch_size = ctx.shape[0]

        dec_init, enc_cache = self.decoder.init_decoder(ctx, ctx_mask)

        middle_ctx, _ = self.decoder.enc_attn(query=dec_init, memory=ctx, cache=enc_cache, mask=ctx_mask)

        middle_output, middle_words, middle_index = self.decoder.get_middle_word(trg, middle_ctx, inference=False)

        hiddens = dec_init.unsqueeze(1)

        outputs = [middle_output, ]

        """
        获取全部sorted_trg和embeddings以及trg_mask
        """
        sorted_trg_list = [middle_words, ]

        left_sequences = []
        right_sequences = []
        for i in range(batch_size):
            # 需要把左边的反转
            left_sequences.append(trg[i][:middle_index[i]].__reversed__())
            right_sequences.append(trg[i][middle_index[i] + 1:y_len[i] + 2])

        # pad 填充
        left_trg = rnn_utils.pad_sequence(left_sequences, batch_first=True)
        right_trg = rnn_utils.pad_sequence(right_sequences, batch_first=True)

        # 构造sorted_trg
        left_trg_len = left_trg.shape[-1]
        right_trg_len = right_trg.shape[-1]
        max_index = max(left_trg_len, right_trg_len)
        # [batch_size]
        PAD_tensor = left_trg.new(batch_size).fill_(PAD)  # 看一下数据类型，需要使用float()嘛？
        for i in range(max_index):
            if i < right_trg_len:
                sorted_trg_list.append(right_trg[:, i])
            else:
                sorted_trg_list.append(PAD_tensor)
            if i < left_trg_len:
                sorted_trg_list.append(left_trg[:, i])
            else:
                sorted_trg_list.append(PAD_tensor)

        # [batch_size, trg_len]
        sorted_trg = torch.stack(sorted_trg_list, dim=-1)
        # [batch_size, trg_len, emb_dim]
        embeddings = self.decoder.embeddings(sorted_trg)

        return embeddings, hiddens, outputs, sorted_trg, enc_cache

    def decode(self, dec_states):
        """
        trg_seq: 目前已经生成的decoder
        目前先只能单个操作把， 如果单个停止了，预测结果手动改成PAD？
        :param tgt_seq:
        :param dec_states:
        :return:
        """
        ctx = dec_states['ctx']
        ctx_mask = dec_states['ctx_mask']
        hiddens = dec_states['hiddens']
        embeddings = dec_states['embeddings']
        trg_mask = dec_states['trg_mask']
        right_next = dec_states['right_next']
        enc_cache = dec_states['enc_cache']

        hiddens, logits = self.decoder.inference(right_next, hiddens, embeddings, ctx, ctx_mask, enc_cache)

        dec_states = {
            "ctx": ctx,  # 这个不用变
            "ctx_mask": ctx_mask,  # 这个不用变
            "hiddens": hiddens,  # 这个变了
            "embeddings": embeddings,  # 这个还没有改变
            "trg_mask": trg_mask,  # 这个还没有改变
            "right_next": right_next,  # 这个是没有变化之前的right_next
            "enc_cache": enc_cache,  # 这个不变
        }

        return logits, dec_states

    def forward(self, src, trg, y_len=None):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        # ctx: [batch_size, src_
        # seq_len, context_size=enc_hid_dim*2]
        # ctx_mask: [batch_size, src_seq_len]

        ctx, ctx_mask = self.encoder(src)
        embeddings, hiddens, outputs, sorted_trg, enc_cache = self.init_decoder_when_train(ctx, ctx_mask, trg, y_len)

        # [batch_size, trg_seq_len, vocab_size]
        outputs = self.decoder(ctx, ctx_mask, embeddings, hiddens, enc_cache, outputs)

        return outputs, sorted_trg

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):
        # 相当于按照beam_size 重新排序, 现在的batch_size 其实是batch_size * beam_size

        hiddens = dec_states['hiddens']
        embeddings = dec_states['embeddings']
        trg_mask = dec_states['trg_mask']
        right_next = dec_states['right_next']

        # 根据beam的结果重新处理

        batch_size = hiddens.size(0) // beam_size

        # hiddens: 先拼接上当前beam的hidden，然后排序
        hiddens = tensor_gather_helper(gather_indices=new_beam_indices,
                                       gather_from=hiddens,
                                       batch_size=batch_size,
                                       beam_size=beam_size,
                                       gather_shape=hiddens.size())
        dec_states['hiddens'] = hiddens

        # 首先按照beam排序，后拼接上当前beam所采用的word
        embeddings = tensor_gather_helper(gather_indices=new_beam_indices,
                                          gather_from=embeddings,
                                          batch_size=batch_size,
                                          beam_size=beam_size,
                                          gather_shape=embeddings.size())

        # 首先按照beam排序，后拼接上当前beam所采用的word
        trg_mask = tensor_gather_helper(gather_indices=new_beam_indices,
                                        gather_from=trg_mask,
                                        batch_size=batch_size,
                                        beam_size=beam_size,
                                        gather_shape=trg_mask.size())

        final_word_indices = dec_states['final_word_indices']
        words = final_word_indices[:, :, -1].view(batch_size * beam_size)
        embed = self.decoder.embeddings(words).unsqueeze(1)

        words = words.detach()
        _trg_mask = (words.eq(PAD) + words.eq(EOS) + words.eq(BOS)).unsqueeze(1)

        # 直接拼接在最后
        embeddings = torch.cat((embeddings, embed), dim=1)
        trg_mask = torch.cat((trg_mask, _trg_mask), dim=1)

        dec_states['embeddings'] = embeddings
        dec_states['trg_mask'] = trg_mask

        dec_states['right_next'] = not right_next

        return dec_states
