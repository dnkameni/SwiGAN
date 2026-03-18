# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:28:57 2022

@author: Wang
"""

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import argparse
import time as t

#models
from basic_deep_learning_models import oned_cnn, LSTM, Transformer
from CNN_LSTM_hybrid_models import cnn_lstm, lstm_cnn, cnn_with_lstm
from attention_and_lstm_hybrid_models import FA_lstm, TA_lstm, FTA_lstm

# gpu or cpu
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = 'cpu'

torch.set_default_tensor_type(torch.DoubleTensor)
feature = 8
timestep =4
parser = argparse.ArgumentParser()
parser.add_argument('--station', type = str, default = 'Monahans-6-ENE', help = '')
parser.add_argument('--depth', type = str, default = '0.0500', help = '')
parser.add_argument('--ismn', type = str, default = 'USCRN', help = '')
parser.add_argument('--path', type = str, default = 'E:/data', help = '')
arg = parser.parse_args()
for time in range(15): # training times
    begin = 0
    end = -1
    station_name = arg.station
    de = arg.depth
    ismn = arg.ismn
    path = arg.path
    data = pd.read_csv(f"{path}/{station_name}.csv")#D:/sm_data/a_Final/SN-Dhr/SN-Dhr.csv
    theta_star_5 = data[f'sm_{de}00_{de}00'].values[:,None][begin:end]*100
    choose = theta_star_5 > -998
    theta_star_5 = np.where(choose, theta_star_5, math.nan)
    time = theta_star_5.shape[0]-1
    
    # normalize
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
    data_timeseries = np.hstack((data_timeseries, theta_before))    
    # print(np.isnan(data_timeseries).sum())
    
    # formation of tensor
    train_size = int(len(data_timeseries)*0.6)
    var_size = int(len(data_timeseries)*0.8)
    x_train = data_timeseries[:train_size].reshape(-1, feature) #input p and T-------------------------------------------------------------imput weather
    y1_train = theta_train_5[:train_size].reshape(-1, 1) #output theta
    
    tlist = []
    for j in range(x_train.shape[0] - timestep):
        if not np.isnan(np.sum(x_train[j:j+5])) and not np.isnan(np.sum(y1_train[j:j+5])) :
            tlist.append(j)
    a = x_train[tlist]
    
    
    x_var = data_timeseries[train_size:var_size].reshape(-1, feature)
    y_var = theta_train_5[train_size:var_size].reshape(-1, 1)
    vlist = []
    for j in range(x_var.shape[0] - timestep -7):
        if not np.isnan(np.sum(x_var[j:j+5+3])) and not np.isnan(np.sum(y_var[j:j+5+3])) :
            vlist.append(j)     
    x_var = torch.from_numpy(x_var).to(device)
    y_var = torch.from_numpy(y_var).to(device)
    
    x_test = data_timeseries[var_size:].reshape(-1, feature)
    y_test = theta_train_5[var_size:].reshape(-1, 1)
    telist = []
    for j in range(x_test.shape[0] - timestep - 7):
        if not np.isnan(np.sum(x_test[j:j+5+7])) and not np.isnan(np.sum(y_test[j:j+5+7])) :
            telist.append(j)    
    x_test = torch.from_numpy(x_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)     
    
    # form tensor
    y1_train = torch.from_numpy(y1_train).to(device)
    x_train = torch.from_numpy(x_train).to(device)
    
    # generate_batch
    def generate_batch(x_train, y_train, batchnum, batchsize, nlist):#indexlen
        batchlist = []
        thetalist = []
        for i in range(batchnum):
            b_size = batchsize
            indexlist = np.random.choice(nlist, b_size, replace=False)
            batch = torch.cat([x_train[x:x+timestep].unsqueeze(0) for x in indexlist], 0)
            theta = torch.cat([y_train[x+timestep-1].reshape(1,1) for x in indexlist], 0)
            batchlist.append(batch)
            thetalist.append(theta)
        return batchlist, thetalist, indexlist
    # model
    model_name = 'lstm' # cnn, transformer, cnn_lstm, lstm_cnn, cnn_with_lstm, fa-lstm, ta-lstm, fta_lstm
    model = LSTM().to(device)
    # model = oned_cnn().to(device)
    # model = Transformer(d_k=4, d_v=4, d_model=feature, d_ff=10, n_heads=1, num = timestep).to(device) 
    # model = cnn_lstm().to(device)
    # model = lstm_cnn().to(device)
    # model = cnn_with_lstm().to(device)
    # model = FA_lstm(d1 = feature).to(device)
    # model = TA_lstm(d2 = timestep).to(device)
    # model = FTA_lstm(d1 = feature,d2 = timestep).to(device)
    
    
    # optimizer
    optimizer = torch.optim.Adam([{'params':model.parameters(), 'lr':0.001}])
    loss_MSE = nn.MSELoss()                                             
    
    # train
    epoch = 1500
    batchsize = 50
    batchnum = 20
    min_loss_val = 10  
    best_model = None
    min_epoch = 100  
    if batchsize < len(vlist):
        batch_var, theta_var,_ = generate_batch(x_var, y_var, epoch, batchsize, vlist)
    else:
        batch_var, theta_var,_ = generate_batch(x_var, y_var, epoch, len(vlist), vlist)
    
    batch_test, theta_test, indexlist = generate_batch(x_test, y_test, 1, len(telist), telist)
    # print(batch_test[0].shape)
    time_now = 0
    for i in range(epoch):
        
        batchlist, thetalist,_ = generate_batch(x_train, y1_train, batchnum, batchsize, tlist)
            
        for each in range(batchnum):
            theta_5 = model(batchlist[each])
            loss = loss_MSE(theta_5, thetalist[each])
            optimizer.zero_grad()              
            loss.backward()
            optimizer.step()  
            
        theta_var_pre = model(batch_var[i])
        loss_val = loss_MSE(theta_var_pre, theta_var[i])
        
        if i > min_epoch and loss_val <= min_loss_val:
            min_loss_val = loss_val
            print(loss_val)
            best_model = copy.deepcopy(model)
                                                    
        if (i+1)%10==0:
            print('Epoch:{}, Loss:{:.10f}'.format(i+1, loss.item()))
            
        if (i+1)%100==0:
            print(t.time()-time_now)
            time_now = t.time()
            
    
    # test  one by one
    dataX1 = data_timeseries
    dataX2 = torch.from_numpy(dataX1).to(device)
    dataY = torch.from_numpy(theta_train_5).to(device)
    pred_theta = []
    for i in range(dataX2.shape[0]-3):
        var = dataX2[i:i+4].unsqueeze(0)
        theta_pre = best_model(var)
        pred_theta.append(theta_pre.cpu().data.numpy().flatten())
    
    pred_theta = np.array(pred_theta)
    
    
    fig, axs = plt.subplots(1, 1, figsize=(10,2))
    ind = np.arange(0,data_timeseries.shape[0],1)
    
    axs.plot(((theta_train_5+1)/2*(max_value_theta- min_value_theta)+min_value_theta)[timestep:], label='real', alpha=1)
    axs.plot(((pred_theta+1)/2*(max_value_theta- min_value_theta)+min_value_theta)[1:], label=f'{model_name}', alpha=0.7)
    axs.set_xlim(len(data_timeseries)*0.6,time)
    axs.set_xlabel('time')
    axs.set_ylabel('sm')
    axs.grid(True)
    axs.legend(loc='best')

    # 1step
    batch_test1 = batch_test[0]
    theta = batch_test[0][:,:,feature-1].unsqueeze(2)
    for i in range(1):
        theta_pre1_11 = best_model(batch_test1).reshape(-1,1,1) 
        theta = torch.cat((theta,theta_pre1_11 ),1)
        batch_test1 = torch.cat([x_test[x+1+i:x+1+timestep+i] for x in indexlist]).reshape(-1, timestep, feature)[:,:,:feature-1]
        # batch_test3 = torch.cat((batch_test3[:, :, :2], torch.cat((batch_test3[:, :3, 2].reshape(-1,3,1), theta_pre1_33), 1)),2)
        batch_test1 = torch.cat((batch_test1, theta[:, i+1:i+1+timestep, :]),2)
        # batch_test3 = np.concatenate((batch_test3, theta_pre1_33),1)
        if i == 0:
            theta_pre1_1 = theta_pre1_11
        else:
            theta_pre1_1 = torch.cat((theta_pre1_1, theta_pre1_11),2)
            
    theta1_1 = (torch.cat([y_test[x+timestep-1] for x in indexlist], 0).reshape(-1, 1)+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta_pre1_1 = (theta_pre1_1.squeeze(2).cpu().data.numpy()+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    
    theta1_1 = theta1_1.cpu().data.numpy()
    
    l2_1 = np.sqrt(mean_squared_error(theta1_1, theta_pre1_1)) / np.mean(theta1_1)
    l1_1 = mean_absolute_error(theta1_1, theta_pre1_1)/np.mean(theta1_1)
    
    rmse_1 = np.sqrt(mean_squared_error(theta1_1, theta_pre1_1))
    mae_1 =  mean_absolute_error(theta1_1, theta_pre1_1)
    r2_1 = r2_score(theta1_1, theta_pre1_1)
    
    batch_test3 = batch_test[0]
    theta = batch_test[0][:,:,feature-1].unsqueeze(2)
    for i in range(3):
        theta_pre1_33 = best_model(batch_test3).reshape(-1,1,1) 
        theta = torch.cat((theta,theta_pre1_33 ),1)
        batch_test3 = torch.cat([x_test[x+1+i:x+timestep+1+i] for x in indexlist]).reshape(-1, timestep, feature)[:,:,:feature-1]
        # batch_test3 = torch.cat((batch_test3[:, :, :2], torch.cat((batch_test3[:, :3, 2].reshape(-1,3,1), theta_pre1_33), 1)),2)
        batch_test3 = torch.cat((batch_test3, theta[:, i+1:i+1+timestep, :]),2)
        # batch_test3 = np.concatenate((batch_test3, theta_pre1_33),1)
        if i == 0:
            theta_pre1_3 = theta_pre1_33
        else:
            theta_pre1_3 = torch.cat((theta_pre1_3, theta_pre1_33),1)
            
    theta1_3 = (torch.cat([y_test[x+timestep-1:x+timestep-1+3] for x in indexlist], 0).reshape(-1, 3)+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta_pre1_3 = (theta_pre1_3.squeeze(2).cpu().data.numpy()+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    
    theta1_3 = theta1_3.cpu().data.numpy()
    l2_3 = np.sqrt(mean_squared_error(theta1_3, theta_pre1_3)) / np.mean(theta1_3)
    l1_3 = mean_absolute_error(theta1_3, theta_pre1_3)/np.mean(theta1_3)
    
    rmse_3 = np.sqrt(mean_squared_error(theta1_3, theta_pre1_3))
    mae_3 =  mean_absolute_error(theta1_3, theta_pre1_3)
    r2_3 = r2_score(theta1_3, theta_pre1_3)        
    
    batch_test7 = batch_test[0]
    theta = batch_test[0][:,:,feature-1].unsqueeze(2)
    for i in range(7):
        theta_pre1_77 = best_model(batch_test7).reshape(-1,1,1) 
        theta = torch.cat((theta,theta_pre1_77),1)
        batch_test7 = torch.cat([x_test[x+1+i:x+1+timestep+i] for x in indexlist]).reshape(-1, timestep, feature)[:,:,:feature-1]
        batch_test7 = torch.cat((batch_test7, theta[:, i+1:i+1+timestep, :]),2)
        if i == 0:
            theta_pre1_7 = theta_pre1_77
        else:
            theta_pre1_7 = torch.cat((theta_pre1_7, theta_pre1_77),1)
            
    theta1_7 = (torch.cat([y_test[x+timestep-1:x+7+timestep-1] for x in indexlist], 0).reshape(-1, 7)+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta_pre1_7 = (theta_pre1_7.squeeze(2).cpu().data.numpy()+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta1_7 = theta1_7.cpu().data.numpy()
    l2_7 = np.sqrt(mean_squared_error(theta1_7, theta_pre1_7)) / np.mean(theta1_7)
    l1_7 = mean_absolute_error(theta1_7, theta_pre1_7)/np.mean(theta1_7)

    rmse_7 = np.sqrt(mean_squared_error(theta1_7, theta_pre1_7))
    mae_7 =  mean_absolute_error(theta1_7, theta_pre1_7)
    r2_7 = r2_score(theta1_7, theta_pre1_7)   
 




