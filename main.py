# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 07:22:39 2022

@author: Mohammad
"""
    # %%
import numpy as np
import torch
import pandas as pd
import argparse
from model import *
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils import *
import glob
import os
import tqdm 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='lstm')
    parser.add_argument("--epochs",dest= 'epochs', default= 500)
    parser.add_argument("--input_channels",dest= 'input_channels', default= 4)
    parser.add_argument("--input_length",dest= 'input_length', default= 30) # 60 *3 mins
    parser.add_argument("--out_length",dest= 'out_length', default= 3) # 30 *3 mins
    parser.add_argument("--num_layers",dest= 'num_layers', default= 1)
    parser.add_argument("--hidden_size",dest= 'hidden_size', default= 100)
    parser.add_argument("--batch_size",dest= 'batch_size', default= 32)
    parser.add_argument("--learning_rate",dest= 'learning_rate', default= 0.001)
    parser.add_argument("--dropout",dest= 'dropout', default= 0.0)
    # parser.add_argument("--window",dest= 'window', default= 40*5 + 3*5)
    # parser.add_argument("--length_data",dest= 'length_data', default= 10000)
    
    return parser.parse_args()

def add_kernel(x, kernel_size, kernel_enercy):
    
    kernel = np.asarray([ np.exp(-kernel_enercy*x)  for x in range(kernel_size)])
    kernel = np.flip(kernel)
    
    x_padded =np.zeros( (len(x)+ len(kernel)) )
    x_padded [len(kernel):] = x
    out = []
    for i in range(len(x_padded) - len(kernel)):
        d = int(len(kernel)/2)
        a = x_padded[i: i + len(kernel) ]
        value =  kernel@ a 
        out.append(value)

    out = np.asarray(out)
    return out
    


def load_data(list_batch):
    batch_data = []
    for sample_path in list_batch:
        data = np.load(sample_path)
        batch_data.append(data) 
        
    batch_data = np.asarray(batch_data)
    return batch_data
    

if __name__ == '__main__':
    print('Main')
    args = arg_parse()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'


    
    
    # df['set_of_numbers'] = pd.to_numeric(df['set_of_numbers'], errors='coerce')
    
    data_path = "diabetes datasets/Shanghai_T1DM/"
    list_patients = glob.glob(os.path.join(data_path ,'*.xlsx'))
    
    print(len(list_patients))
    
    dataall_dic = {}
    for i, patient in enumerate(list_patients):
        # print(patient)
        df = pd.read_excel(patient)
        for column in list(df):
            df[column] = pd.to_numeric(df[column], errors='coerce')
        
        df = df.replace(np.nan, 0)
        
        input_columns1 = list(df)[1]
        input_columns2 = list(df)[7]
        input_columns3 = list(df)[8]
        
        data_column1 = np.expand_dims(np.asarray(df[input_columns1]), 1)
        data_column2 = np.expand_dims(np.asarray(df[input_columns2]), 1)
        data_column3 = np.expand_dims(np.asarray(df[input_columns3]), 1)
        data = data_column1
        data = np.concatenate( (data, data_column2),  1)
        data = np.concatenate( (data, data_column3),  1)
        
        patient_name = "patient_{:d}".format(i)
        dataall_dic[patient_name] = data
    
    all_data= []
    for patient_name in dataall_dic.keys():
        patient = np.asarray(dataall_dic[patient_name])
        all_data.append(patient)
        
    all_data = np.vstack(all_data)
    scaler_all = MinMaxScaler()
    scaler_all.fit(all_data)
    # all_data = scaler_all.transform(all_data)
    
    #####################
    ## Producing train and test dictionary of all patients
    ##  80 percent of each patient data is selected for train and the rest is for test
    ## 
    #####################
    datatrain_dic = {}
    datatest_dic = {}
    
    for i, patient_name in enumerate(dataall_dic.keys()):
        print(patient_name)
        patient = dataall_dic[patient_name]
        patient = scaler_all.transform(patient)
        threshold = int( 0.8* len(patient))
        datatrain_dic[patient_name] = patient[:threshold, :]
        datatest_dic[patient_name] = patient[threshold:, :]
        
     

    print("Number of patient in train: ", len(datatrain_dic))
    print("Number of patient in test: ", len(datatest_dic))
    patients_train = list(datatrain_dic.keys())
    patients_test = list(datatest_dic.keys())
    print(patients_train)
    print(patients_test)
    

    #####################
    ## Producing sampples for training from each patient
    ## The window step is set to 5
    ## samples are saved into test and train folders
    #####################
    # step = 5
    # for i in range(len(datatrain_dic)):
    #     data = datatrain_dic[patients_train[i]]
    #     length_data_train = len(data)
    #     print(length_data_train)
    #     num_itr = int((length_data_train - (args.input_length + args.out_length) ))-1
    #     for itr in range(0, num_itr,5 ):
    #         sample = data[itr:itr + args.input_length + args.out_length,:]
    #         save_path = 'train'+'/'+ str(i) + str(itr) +'.npy'
    #         np.save(save_path, sample)
            
    # for i in range(len(datatest_dic)):
    #     data = datatest_dic[patients_test[i]]
    #     length_data_test = len(data)
    #     print(length_data_test)
    #     num_itr = int((length_data_test - (args.input_length + args.out_length) ))-1
    #     for itr in range(0, num_itr, 5 ):
    #         sample = data[itr:itr + args.input_length + args.out_length,:]
    #         save_path = 'test'+'/'+ str(i) + str(itr) +'.npy'
    #         np.save(save_path, sample)

    # # %%
    # ###########################
    
    list_data_train = glob.glob(os.path.join('train/' ,'*.npy'))
    list_data_test = glob.glob(os.path.join('test/' ,'*.npy'))
    

    net = new(args, device).to(device)
    
    
    # net.apply(weight_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterian = nn.MSELoss()
    best_result = 0.1
    loss_history = []
    for epoch in range(args.epochs):
        Loss_epoch = [] 
        random.shuffle(list_data_train)
        max_itr = int(len(list_data_train)/args.batch_size)
        # net.train()
        for i in range(max_itr):
            
            batch_list = list_data_train[i*args.batch_size : (i+1)*args.batch_size]
            
            batch_data = load_data(batch_list)
            
            
            batch_input = batch_data[:, :args.input_length, :]
            batch_output = batch_data[:, args.input_length :args.input_length+ args.out_length, 0]
            batch_input = np.asarray(batch_input, dtype= 'float32' )
            batch_output = np.asarray(batch_output, dtype= 'float32' )
            batch_input = torch.from_numpy(batch_input).to(device)
            batch_output = torch.from_numpy(batch_output).to(device)
            batch_input = batch_input.permute(0,2,1)
            # batch_input = batch_data.copy()

            pred = net(batch_input)
            loss = criterian(pred, batch_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss_epoch.append(loss.cpu().detach().numpy())
            # print(loss.cpu().detach().numpy())
        
        avrloss_peoch_train = np.asarray(Loss_epoch).mean()
        

        
        avrloss_peoch_test = test(args, net, device, criterian, list_data_test)
        
        
        if avrloss_peoch_test < best_result:
            best_result = avrloss_peoch_test
            print(epoch, avrloss_peoch_train, avrloss_peoch_test)
        
        
        

        
            
            
            
            
            
            
            
            
            
            
            

