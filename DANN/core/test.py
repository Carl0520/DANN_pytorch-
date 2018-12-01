#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:58:47 2018

@author: gaoyi
"""
import torch
import torch.nn as nn
import torch.optim as optim
import params
from utils import make_cuda, save_model
import numpy as np

def eval_(model, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    model.eval()
    alpha = 0
    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.NLLLoss()
    len_dataloader = len(dataloader)
    data_source_iter = iter(dataloader)

    i = 0
    # evaluate network
    while i < len_dataloader:
        
        data_source = data_source_iter.next()
        s_img, s_label = data_source
            
        s_image = make_cuda(s_img)
        s_label = make_cuda(s_label)
        preds, _ = model(s_image, alpha)
        
        loss += criterion(preds, s_label).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(s_label).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
