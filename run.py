import sys
import os
import argparse
import json
import torch
import random
import numpy as np
from model import DeepFM
from exp import Trainer
from utils import load_config
from criteo import Tokenizer
from dataloader import get_dataloader
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='criteo')
    parser.add_argument('--dataset_config_path', type=str, default='dataset/config.yaml')
    parser.add_argument('--model', type=str, default='DeepFM')
    parser.add_argument('--model_config_path', type=str, default='model/DeepFM/config.yaml')
    args = parser.parse_args()
    random.seed(1337)
    torch.manual_seed(1337)
    np.random.seed(1337)
    # 1. load config
    config = load_config(args)

    # 2. bucket & tokenizer
    tokenizer = Tokenizer()
    with open('dataset/criteo/tokenizer.json', 'r') as f:
        tokenizer.tokenizer = json.load(f)

    # 3. dataloader
    train_loader, valid_loader, test_loader = get_dataloader(config.data_root,
                                                             config.batch_size)
    
    # 4. load or initialize model
    device = 'cuda:0'
    model_id = eval(config.model_id)
    model = model_id(config, tokenizer.tokenizer).to(device)
    #model = torch.compile(model)

    # 5. train
    tqdm.write('***** start training *****')
    trainer = Trainer(model=model,
                      device=device,
                      tokenizer=tokenizer,
                      config=config,
                      train_loader=train_loader,
                      valid_loader=valid_loader,
                      test_loader=test_loader)
    trainer.train()
    tqdm.write('***** finish *****')
