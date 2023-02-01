# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:45:30 2022

@author: Mohammad
"""

import numpy as np
import torch
import pandas as pd
import argparse
from model import *
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from piqa import SSIM

class SSIMLoss(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.SSIM = SSIM( window_size = window_size).cuda()
        
    def forward(self, x, y):
        x = (x+1)/2
        y = (y+1)/2
        return 1-self.SSIM(x, y)



def test(args, net, device, criterian, list_data_test):
    # net.eval()
    Loss = []
    results = []
    Inputs = []
    random.shuffle(list_data_test)
    for i in range(len(list_data_test)):
        
        batch_data = np.load(list_data_test[i])
        batch_data = np.expand_dims(batch_data, 0)
        
        batch_input = batch_data[:, :args.input_length, :]
        batch_output = batch_data[:, args.input_length :args.input_length+ args.out_length, 0]
        
        
        batch_input = np.asarray(batch_input, dtype= 'float32' )
        batch_output = np.asarray(batch_output, dtype= 'float32' )

        batch_input = torch.from_numpy(batch_input).to(device)
        batch_output = torch.from_numpy(batch_output).to(device)
        batch_input = batch_input.permute(0,2,1)
        
        pred = net(batch_input)
        loss = criterian(pred, batch_output)
        
        Loss.append(loss.cpu().detach().numpy())
    
    avrloss = np.asarray(Loss).mean()
    
    # print()
    
    target = batch_data[0,:,0]
    pred = pred[0,:].cpu().detach().numpy()
    batch_input = batch_input[0,0,:].cpu().detach().numpy()
    
    prediction = np.concatenate( (batch_input, pred) )
    
    # print(prediction.shape)
    # print(pred.shape)
    # print(target.shape)
    
    # plt.figure(1)
    # plt.cla()
    # plt.plot(prediction)
    # plt.plot(target)
    # plt.pause(0.01)
    
    
    return avrloss

# def test(args, net, device, criterian, data_test, dropout_value ):
#     Loss = []
    
#     num_patient = len(data_test)
    
#     patients = list(data_test.keys())
#     Loss = []
#     for i in range(num_patient):
        
#         data = data_test[patients[i]]

        
#         length_data = np.shape(data)[0]
#         num_itr = int((length_data - args.window)/args.out_length)
#         for itr in range( num_itr ):
#             start_point_input = itr*args.out_length
#             stop_point_input = itr*args.out_length + args.input_length
            
#             start_point_target = itr*args.out_length + args.input_length
#             stop_point_target = itr*args.out_length + args.input_length \
#                                                         + args.out_length
            
            
#             sample_input = data[start_point_input: stop_point_input , :].T
#             sample_target = data[start_point_target: stop_point_target , 0]
            
#             # print(sample_target.shape)
            
            
#             sample_input = np.expand_dims(sample_input, 0)
#             sample_target = np.expand_dims(sample_target, 0)
            
#             sample_input = np.asarray(sample_input, dtype= 'float32' )
#             sample_target = np.asarray(sample_target, dtype= 'float32' )
            
            
#             sample_input = torch.from_numpy(sample_input).to(device)
#             sample_target = torch.from_numpy(sample_target).to(device)
            
#             pred = net(sample_input)
#             # print(pred.shape)
#             loss = criterian(pred, sample_target)
#             Loss.append(loss.detach().cpu().numpy())
            
#     avrloss = np.asarray(Loss).mean()
    
#     sample_target = sample_target.detach().cpu().numpy()
#     sample_input = sample_input.detach().cpu().numpy()
#     pred = pred.detach().cpu().numpy()
#     return avrloss,  sample_input,  sample_target, pred