# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:53:05 2023

@author: Wang
"""

import torch.nn as nn
import torch
import torch.nn.init as init
import numpy as np

feature =8
timestep = 4
class Feature_attention(nn.Module):                                       
    def __init__(self, d):
        super(Feature_attention, self).__init__()
        self.fn = nn.Linear(d, d)
        self.scale_factor = np.sqrt(d)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, enc_inputs):
        outputs = self.fn(enc_inputs)
        outputs = self.sigmoid(outputs)
        attn = self.softmax(outputs)
        outputs = torch.mul(enc_inputs, attn)
        return outputs, attn
    
class Temporal_attention(nn.Module):                                      
    def __init__(self, d):
        super(Temporal_attention, self).__init__()
        self.fn = nn.Linear(d, d)
        self.scale_factor = np.sqrt(d)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, enc_inputs):
        outputs = self.fn(enc_inputs)
        outputs = self.sigmoid(outputs)
        attn = self.softmax(outputs)
        outputs = torch.mul(enc_inputs, attn)
        return outputs, attn    
    
#FA-LSTM--------------------------------------------------------------------------------------------------------------------    
class FA_lstm(nn.Module):
    def __init__(self, d1): 
        super(FA_lstm, self).__init__()
        self.feature_Attn = Feature_attention(d1)
        self.lstm = nn.LSTM(feature, 16, 2)
        self.fc = nn.Linear(16, 1)
      
    def forward(self, inputs):
        self.lstm.flatten_parameters()
        spa_outputs, spa_attn = self.feature_Attn(inputs) 
        spa_outputs =spa_outputs                           
        lstm_outputs = self.lstm(spa_outputs.permute(1,0,2))[0]
        lstm_outputs = self.fc(lstm_outputs)
        lstm_outputs = lstm_outputs.permute(1,0,2)
        return lstm_outputs[:, timestep-1, :]
    
#TA-LSTM--------------------------------------------------------------------------------------------------------------------    
class TA_lstm(nn.Module):
    def __init__(self, d2): 
        super(TA_lstm, self).__init__()
        self.lstm = nn.LSTM(feature,16,2)
        self.temporal_Attn = Temporal_attention(d2)
        self.fc = nn.Linear(16, 1)
        
    def forward(self, inputs):
        self.lstm.flatten_parameters()
        lstm_outputs = self.lstm(inputs.permute(1,0,2))[0]
        lstm_outputs = lstm_outputs.permute(1,0,2)
        tem_outputs, tem_attn = self.temporal_Attn(lstm_outputs.transpose(1,2))
        tem_outputs = torch.sum(tem_outputs,dim =2)
        fn_outputs = self.fc(tem_outputs)
        return fn_outputs
    
#FTA-LSTM--------------------------------------------------------------------------------------------------------------------    
class FTA_lstm(nn.Module):
    def __init__(self, d1, d2): 
        super(FTA_lstm, self).__init__()
        self.feature_Attn = Feature_attention(d1)
        self.lstm = nn.LSTM(d1, 16, 2)
        self.fc1 = nn.Linear(16,1)
        self.fc2 = nn.Linear(d2,1)
        self.temporal_Attn = Temporal_attention(d2)

        
    def forward(self, inputs):
        self.lstm.flatten_parameters()
        spa_outputs, spa_attn = self.feature_Attn(inputs) 
        lstm_outputs = self.lstm(spa_outputs.permute(1,0,2))[0]
        lstm_outputs = self.fc1(lstm_outputs.permute(1,0,2))
        tem_outputs, tem_attn = self.temporal_Attn(lstm_outputs.transpose(1,2))
        fn_outputs = self.fc2(tem_outputs.squeeze(1))
        return fn_outputs
    
