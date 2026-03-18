# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:25:29 2023

@author: Wang
"""
import torch.nn as nn
import torch
import torch.nn.init as init
import numpy as np

feature = 8
timestep = 4

#CNN-LSTM--------------------------------------------------------------------------------------------------------------------   
class cnn_lstm(nn.Module):  # 多尺度
    def __init__(self): 
        super(cnn_lstm, self).__init__()  
        self.conv = nn.Sequential(nn.Conv1d(in_channels=feature,out_channels=32,kernel_size=2,stride = 1),
        nn.Tanh(),
        nn.Conv1d(in_channels=32,out_channels=64,kernel_size=2,stride = 1),
        nn.Tanh(),
        )
        self.lstm = nn.LSTM(64, 16, 2)
        self.fc = nn.Linear(16,1)
        
    def forward(self,x): 
        self.lstm.flatten_parameters()
        output = self.conv(x.transpose(1,2))
        output = self.lstm(output.permute(2,0,1))[0]    
        output = self.fc(output)
        return output[1,:,:]

#LSTM-CNN--------------------------------------------------------------------------------------------------------------------    
class lstm_cnn(nn.Module):
    def __init__(self): 
        super(lstm_cnn, self).__init__()                                            #  expand
        self.lstm = nn.LSTM(feature, 16, 2)
        self.conv = nn.Sequential(nn.Conv1d(in_channels=16,out_channels=32,kernel_size=2,stride = 1),
        nn.Tanh(),
        nn.Conv1d(in_channels=32,out_channels=64,kernel_size=2,stride = 1),
        nn.Tanh(),
        )
        self.tanh = nn.Tanh()
        self.fnn_end = nn.Linear(128, 1) 
    def forward(self,x): 
        self.lstm.flatten_parameters()
        output = self.lstm(x.permute(1,0,2))[0]   
        output = self.conv(output.permute(1,2,0))        
        output = torch.flatten(output,1,2)
        output = self.fnn_end(output)
        output = self.tanh(output)
        return output.reshape(-1,1)
    
#CNN-with-LSTM--------------------------------------------------------------------------------------------------------------------   
class cnn_with_lstm(nn.Module):
    def __init__(self): 
        super(cnn_with_lstm, self).__init__()    
        self.conv = nn.Sequential(nn.Conv1d(in_channels=feature,out_channels=32,kernel_size=2,stride = 1),
        nn.Tanh(),
        nn.Conv1d(in_channels=32,out_channels=64,kernel_size=2,stride = 1),
        nn.Tanh(),
        )        
        
        self.lstm = nn.LSTM(feature, 16, 2)
        self.fc = nn.Linear(16, 1)
        
        self.fnn_cnn_1 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()
        
        self.fnn_end1 = nn.Linear(2, 10) 
        self.fnn_end2 = nn.Linear(10, 1)        
        
    def forward(self,x): 
        self.lstm.flatten_parameters()
        cnn_output  = self.conv(x.transpose(1,2))
        cnn_output = torch.flatten(cnn_output,1,2)
        cnn_output = self.fnn_cnn_1(cnn_output)              
        cnn_output = self.tanh(cnn_output)

        lstm_output = self.fc(self.lstm(x.permute(1,0,2))[0])
        lstm_output = lstm_output[timestep-1,:,:]

        output = self.fnn_end1(torch.cat((cnn_output, lstm_output),1))
        output = self.tanh(output)
        output = self.fnn_end2(output)
        output = self.tanh(output)
        return output