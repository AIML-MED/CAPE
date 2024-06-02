import argparse
import helpers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cub', help='dataset name i.e. cub')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--config_file', type=str, default='', help='file name of configuration in configs folder \
                        i.e. configs/cub/resnet50_TS.py')
    parser.add_argument('--eval', type=helpers.str2bool, default='False',
                        help="evaluate the model, no training")
    
    args = parser.parse_args()
    
    return args