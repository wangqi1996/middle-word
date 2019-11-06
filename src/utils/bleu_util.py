# coding=utf-8

"""
手动计算BLEU得分
"""
from src.data.vocabulary import PAD, EOS


def count_gram(candidate_list, reference_list, clipped_count, count, index_list=None, max_n=4):
    """
    传统的bleu得分是按照string计算的，这里直接按照id计算
    :param candidate: list[list]
    :param references: list[list]
    计算到4-gram
    :return:
    """

    cand_len = len(candidate_list)
    # 遍历每一个句子
    for i in range(0, cand_len):
        # 构造reference的n-gram的
        ref_gram = []
        for ref in reference_list:
            ref_sentences = ref[i]  # ref对应的句子
            for n in range(1, max_n + 1):
                ref_gram.extend(generate_gram(ref_sentences, n))
        # 计算候选项的得分
        candidate = candidate_list[i][0]
        str_candidate = [str(i) for i in candidate if i != PAD and i != EOS]
        candidate_len = len(str_candidate)
        for index in index_list:
            if index < 0:
                real_index = candidate_len + index
            else:
                real_index = index
            for n in range(1, max_n + 1):
                # for gram_index in range(real_index - n + 1, real_index + 1):
                # 如果index为正数，只包含以它开头的; 如果index为负数，则包含以他结尾的
                if index >= 0:
                    for gram_index in range(real_index, real_index + 1):
                        # 不存在该gram
                        if gram_index < 0 or (gram_index + n) > candidate_len:
                            continue
                        cand_gram = ' '.join(str_candidate[gram_index:gram_index + n])
                        count[index][n - 1] += 1
                        if cand_gram in ref_gram:
                            clipped_count[index][n - 1] += 1
                else:  # index < 0
                    for gram_index in range(real_index - n + 1, real_index - n + 2):
                        # 不存在该gram
                        if gram_index < 0 or (gram_index + n) > candidate_len:
                            continue
                        cand_gram = ' '.join(str_candidate[gram_index:gram_index + n])
                        count[index][n - 1] += 1
                        if cand_gram in ref_gram:
                            clipped_count[index][n - 1] += 1


def generate_gram(sentences, n=4):
    """
    构造gram, id id
    """
    result = []
    str_sentences = [str(i) for i in sentences]
    for index in range(len(sentences) - n + 1):
        gram = ' '.join(str_sentences[index:index + n])
        result.append(gram)
    return result

# if __name__ == '__main__':
#     can = [[1, 2, 3, 4, 6, 0, 2], [1, 2, 3, 57, 8, 9]]
#     ref = [[
#         [1, 4, 5, 3, 2, 7, 8, 22],
#         [2, 3, 3, 57, 8, 9]
#     ], [
#         [1, 4, 5, 3, 7, 8, 3],
#         [2, 3, 4, 2, 44, 2, 5]
#     ],
#         [
#             [1, 2, 3, 4, 7, 8, 44],
#             [2, 3, 4, 7, 22, 4, 2, 5]
#         ]
#     ]
#     count_gram(can, ref, index_list=[0, 1, -2, -1], max_n=3)
