import argparse
import random
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loading import *

import debugpy

def get_training_args():
    """
    解析参数
    """
    parser = argparse.ArgumentParser()
    # Dataset related
    parser.add_argument("--file_dir", type=str, default="data/DBP15K/zh_en",  \
                        required=False, help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--rate", type=float, default=0.2, help="training set rate")


    # Traing related
    parser.add_argument("--save", default="", help="the output dictionary of the model and embedding. (should be created manually)")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--check_point", type=int, default=100, help="check point")
    parser.add_argument("--bsize", type=int, default=7500, help="batch size")

    # Model related 
    parser.add_argument("--hidden_units", type=str, default="128,128,128", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
    parser.add_argument("--instance_normalization", action="store_true", default=False, help="enable instance normalization")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")

    # optim
    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dist", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')")

    args = parser.parse_args()

    model_config = {
        "num_heads_per_layer": [2, 2],
        "dropout": 0.0
    }

    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    
    training_config.update(model_config)

    return training_config


def set_seed(init_seed, use_cuda=True):
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(init_seed)



def train_model(config):
    device = torch.device("cuda" if config["cuda"] and torch.cuda.is_available() else "cpu")

    # Step1: load the graph data
    load_graph(config['file_dir'], config['rate'], device)

    # Step2: prepare the model


    # Step3: prepare loss & optimizer


    # Step4: start the training procedure


def main():
    config = get_training_args()
    set_seed(config['seed'], config['cuda'])


    train_model(config)
    


if __name__ == "__main__":
    # debugpy.listen(5678)
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()
    main()