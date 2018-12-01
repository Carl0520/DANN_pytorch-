#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:44:17 2018

@author: gaoyi
"""
import torch
import torch.nn as nn
import torch.optim as optim
import params
from utils import make_cuda, save_model
import numpy as np

def train(model,dataloader_source,dataloader_target,source_data_loader_eval,target_data_loader_eval):
    
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate,)
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    
    model = model.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()
    
    for epoch in range(params.num_epochs):
        model.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)
    
        i = 0
        while i < len_dataloader:
            
            p = float(i + epoch * len_dataloader) / params.num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
    
            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source
            
            s_image = make_cuda(s_img)
            s_label = make_cuda(s_label)
    
            model.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long().cuda()
    
    
            class_output, domain_output = model(s_image, alpha=alpha)
            
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)
    
            # training model using target data
            data_target = data_target_iter.next()
            t_img, _ = data_target
            t_img = make_cuda(t_img)
  
            batch_size = len(t_img)
    
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long().cuda()
            
            _, domain_output = model(t_img, alpha=alpha)
             
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()
            
            if ((i + 1) % params.log_step== 0):
                print("Epoch [{}/{}] Step [{}/{}]: d_loss_t={} / d_loss_s={} / c_loss_s={}"
                      .format(epoch + 1,
                              params.num_epochs,
                              i + 1,
                              len_dataloader,
                              err_t_domain.item()
                              ,err_s_domain.item()
                              ,err_s_label.item()))
            i+=1
        eval_(model,source_data_loader_eval,'src')
        eval_(model,target_data_loader_eval,'tgt')
        
        if ((epoch + 1) % params.save_step == 0):
            save_model(model, "DANN-{}.pt".format(epoch + 1))
                
                
    return model

def eval_(model, data_loader,mode):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    model.eval()
    alpha = 0
    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    len_dataloader = len(data_loader)
    data_iter = iter(data_loader)

    i = 0
    # evaluate network
    while i < len_dataloader:
        
        data_source = data_iter.next()
        s_img, s_label = data_source
            
        s_image = make_cuda(s_img)
        s_label = make_cuda(s_label)
        preds, _ = model(s_image, alpha)

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(s_label).cpu().sum().item()
        
        i+=1

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("{}, Avg Accuracy = {:2%}".format(mode, acc))
            
    
    
