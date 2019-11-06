import argparse

from src.main import train
from src.accuracy import accuracy_translate
from src.bin import auto_mkdir

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,
                    help="The name of the model. Will alse be the prefix of saving archives.")

parser.add_argument('--reload', action="store_true",
                    help="Whether to restore from the latest archives.")

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--log_path', type=str, default="/home/wangdq/save/log/test/",
                    help="The path for saving tensorboard logs. Default is ./log")

parser.add_argument('--saveto', type=str, default="/home/wangdq/save/model/test/",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--debug', action="store_true",
                    help="Use debug mode.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument('--pretrain_path', type=str, default="", help="The path for pretrained model.")

parser.add_argument("--valid_path", type=str, default="/home/wangdq/save/valid/test",
                    help="""Path to save translation for bleu evaulation. Default is ./valid.""")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir(args.log_path)
    auto_mkdir(args.saveto)
    auto_mkdir(args.valid_path)

    args.config_path = r'/home/wangdq/code/python/nju_nlp/configs/configs.yaml'
    args.use_gpu = True
    args.model_name = 'Transformer'
    train(args)

    # args.batch_size = 512 / 4
    # args.beam_size = 4
    # args.model_path = r'/home/wangdq/njunlp_code/save/model/l2r/Transform.best.final'
    # # args.data_name = 'valid_data'
    # args.data_name = 'sample_data'
    # args.is_n_gram = True
    # accuracy_translate(args)


if __name__ == '__main__':
    run()



# CUDA_VISIBLE_DEVICES=2 nohup python -u /home/wangdq/njunlp_code/test/src/bin/train.py  >> /home/wangdq/njunlp_code/save/l2r.txt  2>&1 &CUDA_VISIBLE_DEVICES=2 nohup python -u /home/wangdq/njunlp_code/test/src/bin/train.py

# CUDA_VISIBLE_DEVICES=2  python -u /home/wangdq/njunlp_code/test/src/bin/train.py