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

from src.data.vocabulary import BOS, EOS, PAD
from src.models.base import NMTModel
from .utils import mask_scores, tensor_gather_helper


def beam_search(nmt_model, beam_size, max_steps, src_seqs, alpha=-1.0):
    """

    Args:
        nmt_model (NMTModel):
        beam_size (int):
        max_steps (int):
        src_seqs (torch.Tensor): [batch_size, seq_len]

    Returns:
    """
    batch_size = src_seqs.size(0)
    # enc_outputs:  ['ctx']: [batch_size, seq_len, hid_dim]
    enc_outputs = nmt_model.encode(src_seqs)
    # 假设还是从trg中取出middle_word
    init_dec_states = nmt_model.init_decoder(enc_outputs, expand_size=beam_size)

    # Prepare for beam searching
    beam_mask = src_seqs.new(batch_size, beam_size).fill_(1).float()
    _beam_mask = src_seqs.new(batch_size, beam_size).fill_(1).float()
    final_lengths = src_seqs.new(batch_size, beam_size).zero_().float()
    beam_scores = src_seqs.new(batch_size, beam_size).zero_().float()
    left_stop = src_seqs.new(batch_size, beam_size).fill_(0).byte()
    right_stop = src_seqs.new(batch_size, beam_size).fill_(0).byte()

    middle_words = init_dec_states['middle_words']
    final_word_indices = middle_words.unsqueeze(1).repeat(1, beam_size).unsqueeze(-1)  # [batch_size, beam_size, 1]

    dec_states = init_dec_states

    for t in range(max_steps - 1):
        # next_score: [batch*beam, vocab_size]
        next_scores, dec_states = nmt_model.decode(dec_states)
        next_scores = - next_scores  # convert to negative log_probs
        # next_scores: [batch, beam, vocab_size]
        next_scores = next_scores.view(batch_size, beam_size, -1)
        next_scores = mask_scores(scores=next_scores, beam_mask=beam_mask)

        beam_scores = next_scores + beam_scores.unsqueeze(2)  # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

        vocab_size = beam_scores.size(-1)

        if t == 0 and beam_size > 1:
            # Force tmo select first beam at step 0, 因为beam_size内的结果都是一样的
            beam_scores[:, 1:, :] = float('inf')

        # Length penalty
        if alpha > 0.0:
            normed_scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + beam_mask + final_lengths).unsqueeze(2) ** alpha
        else:
            normed_scores = beam_scores.detach().clone()

        normed_scores = normed_scores.view(batch_size, -1)

        # Get topK with beams
        # indices: [batch_size, ], 表示下标
        _, indices = torch.topk(normed_scores, k=beam_size, dim=-1, largest=False, sorted=False)
        next_beam_ids = torch.div(indices, vocab_size)  # [batch_size, ]
        next_word_ids = indices % vocab_size  # [batch_size, ]

        # Re-arrange by new beam indices
        beam_scores = beam_scores.view(batch_size, -1)
        # indices和beam_scores的大小相同，
        # beam_scores[i][j] = beam_scores[i][indices[i][j]] dim=1
        # 相当于给topk排序
        # [batch_size, beam_size]
        beam_scores = torch.gather(beam_scores, 1, indices)

        # 给beam_mask按照beam_id重新排序
        beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                         gather_from=beam_mask,
                                         batch_size=batch_size,
                                         beam_size=beam_size,
                                         gather_shape=[-1])

        final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                  gather_from=final_word_indices,
                                                  batch_size=batch_size,
                                                  beam_size=beam_size,
                                                  gather_shape=[batch_size * beam_size, -1])

        final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                             gather_from=final_lengths,
                                             batch_size=batch_size,
                                             beam_size=beam_size,
                                             gather_shape=[-1])

        left_stop = tensor_gather_helper(gather_indices=next_beam_ids,
                                         gather_from=left_stop,
                                         batch_size=batch_size,
                                         beam_size=beam_size,
                                         gather_shape=[-1])

        right_stop = tensor_gather_helper(gather_indices=next_beam_ids,
                                          gather_from=right_stop,
                                          batch_size=batch_size,
                                          beam_size=beam_size,
                                          gather_shape=[-1])

        # If next_word_ids is EOS, beam_mask_ should be 0.0
        # 改掉： 如果连这两个是EOS，则表示左右两个decoder全部结束，此时返回 两个EOS可能是表示同一边停止
        # beam_mask_ = 1.0 - next_word_ids.eq(EOS).float()
        # beam_mask_: 1表示未结束，0表示结束。
        right_next = dec_states['right_next']
        if right_next:
            next_word_ids.masked_fill_((right_stop).eq(1), PAD)
            right_stop = next_word_ids.detach().eq(EOS) + right_stop

            # If an EOS or PAD is encountered, set the beam mask to 0.0
            final_lengths += right_stop.float()
        else:
            next_word_ids.masked_fill_((left_stop).eq(1), PAD)
            left_stop = next_word_ids.detach().eq(EOS) + left_stop

            final_lengths += left_stop.float()

        # final_word_indices 先按照beam排序，后按照word排序
        final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

        dec_states['final_word_indices'] = final_word_indices
        dec_states = nmt_model.reorder_dec_states(dec_states, new_beam_indices=next_beam_ids, beam_size=beam_size)

        if left_stop.eq(1).all() and right_stop.eq(1).all():
            break

    # Length penalty
    if alpha > 0.0:
        scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + final_lengths) ** alpha
    else:
        scores = beam_scores / final_lengths

    _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

    # result: [batch_size, beam_size, seq_len]  每一个代表一个word
    result = tensor_gather_helper(gather_indices=reranked_ids,
                                  gather_from=final_word_indices,
                                  batch_size=batch_size,
                                  beam_size=beam_size,
                                  gather_shape=[batch_size * beam_size, -1])
    # 排序成正确的顺序
    sorted_result = None
    for i in range(batch_size):
        batch_result = None
        for j in range(beam_size):
            temp_sorted = result[i][j][0].unsqueeze(0)  # [seq_len]
            # 先向左翻译的
            right_next = True
            left_stop = False
            right_stop = False
            for k in range(1, len(result[i][j])):
                if left_stop and right_stop:
                    break
                if right_next and not right_stop:
                    # 左边终止了，可能也有凑字数拼凑的PAD
                    temp_sorted = torch.cat((temp_sorted, result[i][j][k].unsqueeze(0)), dim=-1)
                    if result[i][j][k].item() in (EOS, PAD):
                        right_stop = True
                elif not left_stop and not right_next:
                    if result[i][j][k] == EOS:
                        left_stop = True
                        right_next = not right_next
                        continue
                    temp_sorted = torch.cat((result[i][j][k].unsqueeze(0), temp_sorted), dim=-1)
                right_next = not right_next

            # 填充成最大的长度
            if temp_sorted.shape[-1] < result.shape[-1]:
                temp_sorted = torch.cat(
                    (temp_sorted, temp_sorted.new(result.shape[-1] - temp_sorted.shape[-1]).zero_().cuda()), dim=-1)
            if batch_result is None:
                batch_result = temp_sorted.unsqueeze(0)
            else:
                batch_result = torch.cat((batch_result, temp_sorted.unsqueeze(0)), dim=0)
        if sorted_result is None:
            sorted_result = batch_result.unsqueeze(0)
        else:
            sorted_result = torch.cat((sorted_result, batch_result.unsqueeze(0)), dim=0)

    return sorted_result, result[:, 0, 0]
