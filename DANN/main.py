#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:00:28 2018

@author: gaoyi
"""

import params
from utils import get_data_loader, init_model, init_random_seed
from models.model import CNNModel
from core import train
from core.train import eval_
if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)
    
    #set loader
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset,train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    #set models
    cnn_model = init_model(net=CNNModel(),restore=params.model_restore)
    
    if not (cnn_model.restored and params.model_trained):
        train(cnn_model,src_data_loader, tgt_data_loader,src_data_loader_eval,tgt_data_loader_eval)
    
    else:
        eval_(cnn_model,src_data_loader_eval,'src')
        eval_(cnn_model,tgt_data_loader_eval,'tgt')
        
    

