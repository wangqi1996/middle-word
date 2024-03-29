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

from src.data.vocabulary import BOS, EOS, PAD, TAG
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
    init_dec_states = nmt_model.init_decoder(enc_outputs, expand_size=beam_size)

    # Prepare for beam searching
    # beam_mask=1代表没有预测到EOS, =0代表预测到EOS
    beam_mask = src_seqs.new(batch_size, beam_size).fill_(1).float()
    final_lengths = src_seqs.new(batch_size, beam_size).zero_().float()
    beam_scores = src_seqs.new(batch_size, beam_size).zero_().float()
    final_word_indices = src_seqs.new(batch_size, beam_size, 1).fill_(BOS)

    dec_states = init_dec_states

    for t in range(max_steps):

        # next_score: [batch*beam, vocab_size]
        next_scores, dec_states = nmt_model.decode(final_word_indices.view(batch_size * beam_size, -1), dec_states)
        next_scores = - next_scores  # convert to negative log_probs
        # next_scores: [batch, beam, vocab_size]
        next_scores = next_scores.view(batch_size, beam_size, -1)
        next_scores = mask_scores(scores=next_scores, beam_mask=beam_mask)

        beam_scores = next_scores + beam_scores.unsqueeze(2)  # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

        vocab_size = beam_scores.size(-1)

        if t == 0 and beam_size > 1:
            # Force to select first beam at step 0, 因为beam_size内的结果都是一样的
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

        dec_states = nmt_model.reorder_dec_states(dec_states, new_beam_indices=next_beam_ids, beam_size=beam_size)

        # If next_word_ids is EOS, beam_mask_ should be 0.0
        beam_mask_ = 1.0 - next_word_ids.eq(EOS).float()

        # beam_mask=1代表没有预测到EOS, =0代表预测到EOS
        # beam_mask_=1代表没有预测到EOS， =0代表预测到EOS
        next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0),
                                   PAD)  # If last step a EOS is already generated, we replace the last token as PAD
        beam_mask = beam_mask * beam_mask_

        # # If an EOS or PAD is encountered, set the beam mask to 0.0
        final_lengths += beam_mask

        final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

        if beam_mask.eq(0.0).all():
            break

    # Length penalty
    if alpha > 0.0:
        scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + final_lengths) ** alpha
    else:
        scores = beam_scores / final_lengths

    _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

    result = tensor_gather_helper(gather_indices=reranked_ids,
                                  gather_from=final_word_indices[:, :, 1:].contiguous(),
                                  batch_size=batch_size,
                                  beam_size=beam_size,
                                  gather_shape=[batch_size * beam_size, -1])

    sorted_result = None
    EOS_tensor = torch.tensor([EOS, ]).cuda()
    for i in range(batch_size):
        batch_result = None
        for j in range(beam_size):
            k = 0
            for k in range(len(result[i][j])):
                if result[i][j][k] == TAG:
                    break
            _right = result[i][j][:k]
            kk = -1
            for kk in range(k + 1, len(result[i][j])):
                if result[i][j][kk] == EOS:
                    break
            _left = result[i][j][k + 1:kk]
            temp_sorted = torch.cat((_left, _right), dim=-1)
            temp_sorted = torch.cat((temp_sorted, EOS_tensor), dim=-1)

            # 填充成最大的长度
            if temp_sorted.shape[-1] < result.shape[-1]:
                # PAD = 0
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

    return sorted_result
