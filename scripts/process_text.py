# coding=utf-8
from shutil import copyfile

import random


def get_middle(file_names):
    """
    获取中间词
    :param file_names:
    :return:
    """
    for file_name in file_names:
        with open(file_name, 'r') as f_read:
            with open(file_name + '.middle', 'w') as f_write:
                contents = f_read.readlines()
                for line in contents:
                    content_list = line.strip().split()
                    content = content_list[len(content_list) // 2]
                    f_write.write(content)
                    f_write.write('\n')


def sample(file_names, nums):
    """
    随机采样
    :param file_names:
    :param times:
    :return:
    """
    samples = []
    for file_name in file_names:
        with open(file_name, 'r') as f:
            content = f.readlines()
            if len(samples) <= 0:
                all_num = len(content)
                samples = random.sample(range(0, all_num), nums)
            with open(file_name + '.sample6', 'w') as f_new:
                for index in samples:
                    f_new.write(content[index])


def dump(file_names, times):
    """
    复制多份
    :param file_names:
    :param times:
    :return:
    """
    for file in file_names:
        new_file_name = file + '.temp'
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context.endswith('\n'):
                        context = context[:-1]
                    if context.__contains__('\n'):
                        print(context)
                    if context:
                        context = context.strip()
                        for _ in range(times):
                            f_target.write(context)
                            f_target.write('\n')


def middle(file_names, times):
    """
    将文章处理成middle的格式
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '.temp'
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context.endswith('\n'):
                        context = context[:-1]
                    if context.__contains__('\n'):
                        print(context)
                    if context:
                        context = context.strip()
                        context = context.split(' ')
                        for _ in range(times):
                            if len(context) - 2 > 0:
                                index = random.randint(1, len(context) - 2)
                                left = context[0:index]
                                left.reverse()
                                right = context[index + 1:]
                                middle = context[index]
                                left_len = len(left)
                                right_len = len(right)
                                new_context = [middle]
                                for i in range(max(left_len, right_len)):
                                    if len(left) > i:
                                        new_context.append(left[i])
                                    else:
                                        new_context.append('<EOS>')

                                    if len(right) > i:
                                        new_context.append(right[i])
                                    else:
                                        new_context.append('<EOS>')
                                new_context = ' '.join(new_context)
                            else:
                                new_context = ' '.join(context)
                            f_target.write(new_context)
                            f_target.write('\n')


def shuffle(file_names):
    """
    将文章处理成r2l
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '_temp'
        copyfile(file, new_file_name)
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context.endswith('\n'):
                        context = context[:-1]
                    if context.__contains__('\n'):
                        print(context)
                    if context:
                        context_list = context.split(' ')
                        random.shuffle(context_list)
                        context = ' '.join(context_list)
                    f_target.write(context)
                    f_target.write('\n')


def r2l(file_names):
    """
    将文章处理成r2l
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '_temp'
        copyfile(file, new_file_name)
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context.endswith('\n'):
                        context = context[:-1]
                    if context.__contains__('\n'):
                        print(context)
                    if context:
                        context_list = context.split(' ')
                        context_list.reverse()
                        context = ' '.join(context_list)
                    f_target.write(context)
                    f_target.write('\n')


def first_2(file_names):
    """
    将文章处理成r2l
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '_temp'
        copyfile(file, new_file_name)
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context:
                        if context.endswith('\n'):
                            context = context[:-1]
                        if context.__contains__('\n'):
                            print(context)
                        context_list = context.split(' ')
                        if len(context_list) > 1:
                            context_list.append(context_list[0])
                            context_list.pop(0)
                            context = ' '.join(context_list)
                    f_target.write(context)
                    f_target.write('\n')


def first_2_r2l(file_names):
    """
    将文章处理成r2l
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '_temp'
        copyfile(file, new_file_name)
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context:
                        if context.endswith('\n'):
                            context = context[:-1]
                        if context.__contains__('\n'):
                            print(context)
                        context_list = context.split(' ')
                        if len(context_list) > 1:
                            # 将第一个单词拼接在最后 并使用r2l
                            context_list.append(context_list[0])
                            context_list.pop(0)
                            context_list.reverse()
                            context = ' '.join(context_list)
                    f_target.write(context)
                    f_target.write('\n')


def remove_1(file_names):
    """
    删除第一个词语
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '_temp'
        copyfile(file, new_file_name)
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context:
                        if context.endswith('\n'):
                            context = context[:-1]
                        if context.__contains__('\n'):
                            print(context)
                        context_list = context.split(' ')
                        if len(context_list) > 1:
                            context_list.pop(0)
                            context = ' '.join(context_list)
                    f_target.write(context)
                    f_target.write('\n')


if __name__ == '__main__':
    # file_names = [
    #     "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/r2l/mt03.ref0",
    #     "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/r2l/mt03.ref1",
    #     "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/r2l/mt03.ref2",
    #     "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/r2l/mt03.ref3",
    #     "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/train/r2l/train.en"
    # ]
    # r2l(file_names)
    # file_names = ["/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2/mt03.ref0",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2/mt03.ref1",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2/mt03.ref2",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2/mt03.ref3",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/train/first_2/train.en"]
    # first_2(file_names)
    # file_names = ["/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2_r2l/mt03.ref0",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2_r2l/mt03.ref1",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2_r2l/mt03.ref2",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2_r2l/mt03.ref3",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/train/first_2_r2l/train.en"]
    # first_2_r2l(file_names)

    # file_names = ["/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/remove_1/mt03.ref0",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/remove_1/mt03.ref1",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/remove_1/mt03.ref2",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/remove_1/mt03.ref3",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/train/remove_1/train.en"]
    # remove_1(file_names)

    # file_name = ["/home/user_data55/wangdq/data/nist_zh-en_1.34m/test/"]
    # dump(file_name, 3)

    # file_names = ['/home/user_data55/wangdq/data/nist_zh-en_1.34m/train/train.zh',
    #               '/home/user_data55/wangdq/data/nist_zh-en_1.34m/train/train.en.l2r']
    #
    # sample(file_names, 1000)

    # file_name = ['/home/user_data55/wangdq/data/nist_zh-en_1.34m/test/mt03.ref0']
    # get_middle(file_name)

    file_name = ['/home/user_data55/wangdq/data/nist_zh-en_1.34m/train/middle/train.en']
    middle(file_name, 1)
