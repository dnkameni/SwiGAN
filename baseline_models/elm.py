# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:44:19 2022

@author: Wang
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import argparse


torch.set_default_tensor_type(torch.DoubleTensor)

class ELM(nn.Module):
    def __init__(self, input_SM, imput_W, hidden_dim, output_dim):
        super(ELM, self).__init__()

        self.hidden1_dim = hidden_dim
        self.output_dim = output_dim

        self.hidden1 = nn.Linear(input_SM + 1, hidden_dim)
        self.hidden2 = nn.Linear(imput_W, 1)
        self.beta = nn.Linear(hidden_dim, output_dim, bias=False)

        # initialize hidden layer with Gaussian random weights + biases of mean 0 and var 1
        self.hidden1.weight.data = torch.randn_like(self.hidden1.weight.data)
        self.hidden1.bias.data = torch.randn_like(self.hidden1.bias.data)
        self.hidden2.weight.data = torch.randn_like(self.hidden2.weight.data)
        self.hidden2.bias.data = torch.randn_like(self.hidden2.bias.data)


    def fit(self, sm, weather, label, batchsize, flatten=False):
        H_outputs = torch.empty(batchsize*20, self.hidden1_dim)                              
        labels = torch.empty(batchsize*20, self.output_dim) # will be Nxoutput_dim

        # compute and append hidden layer output for each sample
        dataset_count = 0 # keep track of index of current training sample
        for index, data_batch in enumerate(sm): #, labelbatch, weatherbatch
            labelbatch = label[index]
            weatherbatch = weather[index]
            batch_count = 0 # keep track of index in this batch
            for x in data_batch:
                x = torch.cat((x, self.hidden2(weatherbatch[batch_count])))
                # compute and append hidden layer output for this sample
                if flatten:
                    H_outputs[dataset_count] = F.sigmoid(self.hidden1(torch.flatten(x)))
                else:
                    H_outputs[dataset_count] = F.sigmoid(self.hidden1(x))

                # append one-hot encoded label to labels tensor
                # if one_hot:
                #     labels[dataset_count] = F.one_hot(label_batch[batch_count], num_classes=self.output_dim)
                # else:
                labels[dataset_count] = labelbatch[batch_count]

                dataset_count += 1 
                batch_count += 1
        
        H_outputs_pinv = torch.linalg.pinv(H_outputs)

        # apply beta layer = pseudoinverse(H) * labels
        
        self.beta.weight.data = torch.matmul(H_outputs_pinv, labels)

    def forward(self, x, w, flatten=False):
        H_out = F.sigmoid(self.hidden1(torch.cat((x, self.hidden2(w)),1)))
        pred = torch.matmul(H_out, self.beta.weight.data)
        return pred

    


# read data
parser = argparse.ArgumentParser()
parser.add_argument('--station', type = str, default = 'AAMU', help = '')
parser.add_argument('--depth', type = str, default = '0.0508', help = '')
parser.add_argument('--ismn', type = str, default = 'SCAN', help = '')
parser.add_argument('--path', type = str, default = 'E:/data', help = '')
arg = parser.parse_args()
for time in range(10):
    begin = 0
    end = -1
    b_size = 100
    
    station_name = arg.station
    de = arg.depth
    ismn = arg.ismn
    path = arg.path
    data = pd.read_csv(f"{path}/{station_name}.csv")#D:/sm_data/a_Final/SN-Dhr/SN-Dhr.csv
    theta_star_5 = data[f'sm_{de}00_{de}00'].values[:,None][begin:end]*100
    choose = theta_star_5 > -998
    theta_star_5 = np.where(choose, theta_star_5, math.nan)
    
    feature = 7
    time = theta_star_5.shape[0]-1
    
    max_value_theta = np.nanmax(theta_star_5.flatten())
    min_value_theta = np.nanmin(theta_star_5.flatten())    
    theta_train_5 = theta_star_5[1:]
    theta_train_5 = 2*(theta_train_5 - min_value_theta) / (max_value_theta- min_value_theta) - 1
    theta_before = theta_star_5[:-1]
    theta_before = 2*(theta_before - min_value_theta) / (max_value_theta- min_value_theta) - 1
    
    # normalization
    data_timeseries = data[['p', 'ta', 'SW', 'LW', 'WS', f'ts_{de}00_{de}00','RH']][begin:end][1:]
    for va in ['p', 'ta', 'SW', 'LW', 'WS', f'ts_{de}00_{de}00','RH']:
        choose = data_timeseries[f'{va}'] > -998
        data_timeseries[f'{va}'] = np.where(choose, data_timeseries[f'{va}'], math.nan)
        
    max_value = np.max(data_timeseries)
    min_value = np.min(data_timeseries)
    data_timeseries = 2*(data_timeseries - min_value) / (max_value- min_value) - 1
    data_timeseries = np.array((data_timeseries))
    
    p_timeseries = data_timeseries
    # formation of tensor
    train_size = int(len(p_timeseries)*0.6)
    x_train = p_timeseries[:train_size] #input p and T-------------------------------------------------------------imput weather
    y1_train = theta_train_5[:train_size] #output theta

    #-------------------------------------------------------------------------------------------------------------------------------------------
    x_train = x_train.reshape(-1,feature)
    y1_train = y1_train.reshape(-1, 1)
    
    # form tensor
    train_size = int(len(data_timeseries)*0.6)
    var_size = int(len(data_timeseries)*0.8)
    x_train = data_timeseries[:train_size].reshape(-1, feature) #input p and T-------------------------------------------------------------imput weather
    y1_train = theta_train_5[:train_size].reshape(-1, 1) #output theta
    
    tlist = []
    for j in range(x_train.shape[0] - 4):
        if not np.isnan(np.sum(x_train[j:j+5])) and not np.isnan(np.sum(y1_train[j:j+5])) :
            tlist.append(j)
    a = x_train[tlist]
    
    
    x_var = data_timeseries[train_size:var_size].reshape(-1, feature)
    y_var = theta_train_5[train_size:var_size].reshape(-1, 1)
    vlist = []
    for j in range(x_var.shape[0] - 4 -7):
        if not np.isnan(np.sum(x_var[j:j+5+3])) and not np.isnan(np.sum(y_var[j:j+5+3])) :
            vlist.append(j)     
    x_var = torch.from_numpy(x_var)
    y_var = torch.from_numpy(y_var)
    
    x_test = data_timeseries[var_size:].reshape(-1, feature)
    y_test = theta_train_5[var_size:].reshape(-1, 1)
    telist = []
    for j in range(x_test.shape[0] - 4 - 7):
        if not np.isnan(np.sum(x_test[j:j+5+7])) and not np.isnan(np.sum(y_test[j:j+5+7])) :
            telist.append(j+var_size)    
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)        
    
    # form tensor
    y1_train = torch.from_numpy(y1_train)
    x_train = torch.from_numpy(x_train)
    # generate_batch
    def generate_batch_elm(batchnum, batchsize, nlist):
        sm_batch = []
        label_batch = []
        weather_batch = []
        for i in range(batchnum):
            b_size = batchsize
            indexlist = np.random.choice(nlist, b_size, replace=False)
            weather = torch.cat([x_train[x+4].reshape(-1,feature) for x in indexlist], 0)
            label = torch.cat([y1_train[x+4].reshape(-1,1) for x in indexlist], 0)
            sm = torch.cat([y1_train[x:x+4].reshape(-1,4) for x in indexlist], 0)
            sm_batch.append(sm)
            label_batch.append(label)
            weather_batch.append(weather)
        return sm_batch, label_batch, weather_batch
    
    
    smbatch, labelbatch, weatherbatch = generate_batch_elm(20, 40, tlist)
    
    # load model
    timestep = 4
    model = ELM(timestep, feature, 20, 1)
    model.fit(smbatch, weatherbatch, labelbatch, 40, flatten=False)

    # test
    dataX1 = p_timeseries#[train_size-4:] #(188, 2)
    dataX2 = theta_train_5

    dataX1 = torch.from_numpy(dataX1)
    dataX2 = torch.from_numpy(dataX2)
    pred_theta = []
    for i in range(dataX2.shape[0]-4):
        var2 = dataX2[i:i+4].reshape(-1,4)
        var1 = dataX1[i+4].reshape(-1,feature)
        theta_pre = model(var2, var1)
        pred_theta.append(theta_pre)
    
    pred_theta = np.array(pred_theta)
    
    
    fig, axs = plt.subplots(1, 1, figsize=(10,2))
    # fig.figure(figsize=(10,2))
    axs.plot(((theta_train_5+1)/2*(max_value_theta- min_value_theta)+min_value_theta)[4:], label='real', alpha=1)
    axs.plot(((pred_theta+1)/2*(max_value_theta- min_value_theta)+min_value_theta), label='ELM', alpha=0.7)
    axs.set_xlim(len(p_timeseries)*0.6,end-begin)#P.shape[0]
    axs.set_xlabel('time')
    axs.set_ylabel('sm')
    axs.grid(True)
    axs.legend(loc='best')
    
     # test
    indexlist = np.random.choice(telist, len(telist), replace=False)
    batch_test = np.concatenate([p_timeseries[x+4] for x in indexlist], 0).reshape(-1, feature)
    theta_test = np.concatenate([theta_train_5[x:x+4] for x in indexlist], 0).reshape(-1, 4)
    label_test = np.concatenate([theta_train_5[x+4] for x in indexlist], 0).reshape(-1, 1)
    theta_test = torch.from_numpy(theta_test)
    batch_test =torch.from_numpy(batch_test)
    
    theta_pre1 = model(theta_test, batch_test)

    theta1 = (label_test+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta_pre1 = (theta_pre1.data.numpy()+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    
    rmse_1 = np.sqrt(mean_squared_error(theta1, theta_pre1)) / np.mean(theta1)
    L1_1 = mean_absolute_error(theta1, theta_pre1)/np.mean(theta1)

    rmse_1 = np.sqrt(mean_squared_error(theta1, theta_pre1))
    mae_1 =  mean_absolute_error(theta1, theta_pre1)
    r2_1 = r2_score(theta1, theta_pre1)
    # 3 step
    indexlist_test3 = indexlist
    batch_test3 = np.concatenate([p_timeseries[x+4] for x in indexlist_test3], 0).reshape(-1, feature) # 150, 2
    theta_test3 = np.concatenate([theta_train_5[x:x+4] for x in indexlist_test3], 0).reshape(-1, 4) # 150, 4
    label_test3 = np.concatenate([theta_train_5[x+4:x+7] for x in indexlist_test3], 0).reshape(-1, 3) # 150, 3
    
    theta_test3 = torch.from_numpy(theta_test3)
    batch_test3 =torch.from_numpy(batch_test3)
    for i in range(3):
        theta_pre1_33 = model(theta_test3[:,i:i+4], batch_test3).reshape(-1,1) #1,4 1,2
        batch_test3 = torch.from_numpy(np.concatenate([p_timeseries[x+4+i+1] for x in indexlist_test3], 0).reshape(-1, feature))
        theta_test3 = torch.cat((theta_test3, theta_pre1_33),1)
        # batch_test3 = np.concatenate((batch_test3, theta_pre1_33),1)
        if i == 0:
            theta_pre1_3 = theta_pre1_33
        else:
            theta_pre1_3 = torch.cat((theta_pre1_3, theta_pre1_33),1)
            
    theta1_3 = (label_test3+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta_pre1_3 = (theta_pre1_3.data.numpy()+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    
    rmse_3 = np.sqrt(mean_squared_error(theta1_3, theta_pre1_3))
    mae_3 =  mean_absolute_error(theta1_3, theta_pre1_3)
    r2_3 = r2_score(theta1_3, theta_pre1_3)
    
    # 7 step
    indexlist_test7 = indexlist
    batch_test7 = np.concatenate([p_timeseries[x+4] for x in indexlist_test7], 0).reshape(-1, feature)
    theta_test7 = np.concatenate([theta_train_5[x:x+4] for x in indexlist_test7], 0).reshape(-1, 4) # 150, 4
    label_test7 = np.concatenate([theta_train_5[x+4:x+11] for x in indexlist_test7], 0).reshape(-1, 7) #40,3
    
    theta_test7 = torch.from_numpy(theta_test7)
    batch_test7 =torch.from_numpy(batch_test7)
    
    for i in range(7):
        theta_pre1_77 = model(theta_test7[:,i:i+4], batch_test7).reshape(-1,1)
        batch_test7 = torch.from_numpy(np.concatenate([p_timeseries[x+4+i+1] for x in indexlist_test7], 0).reshape(-1, feature))
        theta_test7 = torch.cat((theta_test7, theta_pre1_77),1)
        if i == 0:
            theta_pre1_7 = theta_pre1_77
        else:
            theta_pre1_7 = torch.cat((theta_pre1_7, theta_pre1_77),1)
           
    theta1_7 = (label_test7+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta_pre1_7 = (theta_pre1_7.data.numpy()+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    
    rmse_7 = np.sqrt(mean_squared_error(theta1_7, theta_pre1_7))
    mae_7 =  mean_absolute_error(theta1_7, theta_pre1_7)
    r2_7 = r2_score(theta1_7, theta_pre1_7)
