#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:29:49 2018

@author: gaoyi
"""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 28

# params for source dataset
src_dataset = "MNIST"

# params for target dataset
tgt_dataset = 'mnist_m'
tgt_dataset_root = '/home/gaoyi/domain_adapation/DANN/data/mnist_m'

# params for training network
num_gpu = 1
num_epochs = 30
log_step = 20
save_step = 5
manual_seed = None
image_size = 28
# params for optimizing models
learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

#
model_restore = None
model_trained = True
model_root = '/home/gaoyi/domain_adapation/DANN/checkpoints'
