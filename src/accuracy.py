import os
import random

import math
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.metric.bleu_scorer import SacreBLEUScorer
from src.decoding.beam_search import beam_search
from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary, BOS, EOS, PAD
from src.models import build_model
from src.utils.bleu_util import count_gram
from src.utils.common_utils import *
from src.utils.logging import *


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def load_model_parameters(path, map_location):
    state_dict = torch.load(path, map_location=map_location)

    if "model" in state_dict:
        return state_dict["model"]
    return state_dict


def split_shard(*inputs, split_size=1):
    if split_size <= 1:
        yield inputs
    else:

        lengths = [len(s) for s in inputs[-1]]  #
        sorted_indices = np.argsort(lengths)

        # sorting inputs

        inputs = [
            [inp[ii] for ii in sorted_indices]
            for inp in inputs
        ]

        # split shards
        total_batch = sorted_indices.shape[0]  # total number of batches

        if split_size >= total_batch:
            yield inputs
        else:
            shard_size = total_batch // split_size

            _indices = list(range(total_batch))[::shard_size] + [total_batch]

            for beg, end in zip(_indices[:-1], _indices[1:]):
                yield (inp[beg:end] for inp in inputs)


def prepare_data(seqs_x, seqs_y=None, cuda=False, batch_first=True):
    """
    Args:
        eval ('bool'): indicator for eval/infer.

    Returns:

    """

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [BOS] + s + [EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y = list(map(lambda s: [BOS] + s + [EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=PAD,
                         cuda=cuda, batch_first=batch_first)

    return x, y


def accuracy_translate(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    training_configs = configs['training_configs']
    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    if FLAGS.data_name == 'sample_data':
        valid_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=data_configs['sample_data'][0],
                            vocabulary=vocab_src,
                            ),
            TextLineDataset(data_path=data_configs['sample_data'][1],
                            vocabulary=vocab_tgt,
                            )
        )
        reference_path = data_configs['sample_data'][1]
        num_refs = 1
    elif FLAGS.data_name == 'valid_data':
        valid_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=data_configs['valid_data'][0],
                            vocabulary=vocab_src,
                            ),
            TextLineDataset(data_path=data_configs['valid_data'][1],
                            vocabulary=vocab_tgt,
                            ),
            TextLineDataset(data_path=data_configs['valid_data'][2],
                            vocabulary=vocab_tgt,
                            ),
            TextLineDataset(data_path=data_configs['valid_data'][3],
                            vocabulary=vocab_tgt,
                            ),
            TextLineDataset(data_path=data_configs['valid_data'][4],
                            vocabulary=vocab_tgt,
                            )
        )
        reference_path = data_configs["bleu_valid_reference"]
        num_refs = data_configs['num_refs']
    else:
        raise Exception(u'只能为sample_data或者valid_data')

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=FLAGS.batch_size,
                                  use_bucket=True, mbuffer_size=100000, numbering=True)

    bleu_scorer = SacreBLEUScorer(reference_path=reference_path,
                                  num_refs=num_refs,
                                  lang_pair=data_configs["lang_pair"],
                                  sacrebleu_args=training_configs["bleu_valid_configs"]['sacrebleu_args'],
                                  postprocess=training_configs["bleu_valid_configs"]['postprocess']
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()

    if GlobalNames.USE_GPU:
        nmt_model = nmt_model.cuda()

    params = load_model_parameters(FLAGS.model_path, map_location='cpu')

    nmt_model.load_state_dict(params)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')

    timer.tic()

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator()

    index_list = [0, 1, 2, -3, -2, -1]

    #                  1  2  3 -3 -2 -1
    all_correct_num = [0, 0, 0, 0, 0, 0]  # 长度问题
    seqs_correct_nums = np.zeros((4, 6))
    correct_num = [0, 0, 0, 0, 0, 0]

    trans = []
    numbers = []

    max_n = 4  # 最大为4-gram

    clipped_count = np.zeros((len(index_list), max_n))
    count = np.zeros((len(index_list), max_n))

    for i, batch in enumerate(valid_iter):
        if FLAGS.data_name == 'sample_data':
            seq_nums, seqs_x, seqs_y1 = batch
            seqs_y2, seqs_y3, seqs_y4 = seqs_y1, seqs_y1, seqs_y1
        else:
            seq_nums, seqs_x, seqs_y1, seqs_y2, seqs_y3, seqs_y4 = batch
        numbers += seq_nums

        batch_size_t = len(seqs_x)

        x = prepare_data(seqs_x=seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = beam_search(nmt_model=nmt_model, beam_size=FLAGS.beam_size,
                                   max_steps=training_configs["bleu_valid_configs"]["max_steps"],
                                   src_seqs=x, alpha=training_configs["bleu_valid_configs"]["alpha"])

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            x_tokens = []

            for wid in sent_t[0]:
                if wid == EOS:
                    break
                x_tokens.append(vocab_tgt.id2token(wid))

            if len(x_tokens) > 0:
                trans.append(vocab_tgt.tokenizer.detokenize(x_tokens))
            else:
                trans.append('%s' % vocab_tgt.id2token(EOS))

        # if FLAGS.is_n_gram:
        #     count_gram(word_ids, [seqs_y1, seqs_y2, seqs_y3, seqs_y4], clipped_count, count, index_list, max_n)
        # else:
        #     batch_id = -1
        #     for sent_t, trg_seq1, trg_seq2, trg_seq3, trg_seq4 in zip(word_ids, seqs_y1, seqs_y2, seqs_y3,
        #                                                               seqs_y4):  # 循环batch
        #         # 计算准确率
        #         batch_id += 1
        #         line = sent_t[0]  # score得分最大的
        #         line = [wid for wid in line if wid != PAD and wid != EOS]
        #
        #         # for index in [0, 1, -4, -3, -2]:
        #         for index in index_list:
        #             flag = False
        #
        #             if (index >= 0 and len(line) > index) or (index < 0 and len(line) >= abs(index)):
        #                 pre_word = line[index]
        #                 all_correct_num[index] += 1
        #             else:
        #                 continue
        #
        #             for seq_index, trg_seq in enumerate([trg_seq1, trg_seq2, trg_seq3, trg_seq4]):
        #                 if (index >= 0 and len(trg_seq) > index) or (index < 0 and len(trg_seq) >= abs(index)):
        #                     trg_word = trg_seq[index]
        #                     if pre_word == trg_word:
        #                         flag = True
        #                         seqs_correct_nums[seq_index][index] += 1
        #             if flag:
        #                 correct_num[index] += 1

        infer_progress_bar.update(batch_size_t)

    infer_progress_bar.close()

    INFO('Done. Speed: {0:.2f} words/sec'.format(0 / (timer.toc(return_seconds=True))))
    #
    # # 计算准确率
    # print(all_correct_num)
    # print(seqs_correct_nums)
    # print(correct_num)

    # 计算 n-gram log概率
    # result = [0] * len(index_list)
    # for index in index_list:
    #     p = 0
    #     for n in range(max_n):
    #         p += math.log(clipped_count[index][n] / count[index][n]) * (1 / max_n)
    #     result[index] = math.exp(p)
    # for i in result:
    #     print(' %.4f |' % i)
    # print(clipped_count)
    # print(count)

    # 计算bleu得分
    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]

    infer_progress_bar.close()

    if not os.path.exists(FLAGS.valid_path):
        os.mkdir(FLAGS.valid_path)

    uidx = '_test_bleu_score'
    hyp_path = os.path.join(FLAGS.valid_path, 'trans.iter{0}.txt'.format(uidx))

    with open(hyp_path, 'w') as f:
        for line in trans:
            f.write('%s\n' % line)

    with open(hyp_path) as f:
        bleu_v = bleu_scorer.corpus_bleu(f)

    print(bleu_v)
