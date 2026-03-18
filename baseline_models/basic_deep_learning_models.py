# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:46:11 2023

@author: Wang
"""

import torch.nn as nn
import torch
import torch.nn.init as init
import numpy as np

feature = 8
timestep = 4
#CNN--------------------------------------------------------------------------------------------------------------------
class oned_cnn(nn.Module):
    def __init__(self): 
        super(oned_cnn, self).__init__()   
        self.conv = nn.Sequential(nn.Conv1d(in_channels=feature,out_channels=32,kernel_size=2,stride = 1),
        nn.Tanh(),
        nn.Conv1d(in_channels=32,out_channels=64,kernel_size=2,stride = 1),
        nn.Tanh(),) 
        self.fc = nn.Sequential(nn.Linear(128, 1), nn.Tanh())
        
    def forward(self,x): 
        convout = self.conv(x.transpose(1,2))
        flattenout = convout.view(-1,128)
        out = self.fc(flattenout)
        return out
    
    
    
    
#LSTM--------------------------------------------------------------------------------------------------------------------   
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__() 
        self.lstm = nn.LSTM(feature,16,2)
        self.fc =nn.Linear(16, 1)

    def forward(self,x):
        self.lstm.flatten_parameters()
        x1,_ = self.lstm(x.permute(1,0,2))
        out = self.fc(x1.permute(1,0,2))                         
        return out[:, timestep-1, :]




#Encoder-----------------------------------------------------------------------------------------------------------------------------
class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
            for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        self.pos_enc = pos_enc

    def forward(self, ifnot):
        if ifnot:
            return self.pos_enc
        
class Linear(nn.Module): # w,b initialized
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)
    
class ScaledDotProductAttention(nn.Module):                                 
    def __init__(self, d_k, dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]                               
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.softmax(scores)

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn   
    
class _MultiHeadAttention(nn.Module):                                        
    def __init__(self, d_k, d_v, d_model, n_heads):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear(d_model, d_k * n_heads)
        self.w_k = Linear(d_model, d_k * n_heads)
        self.w_v = Linear(d_model, d_v * n_heads)

        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        # return the context and attention weights
        return context, attn
class MultiHeadAttention(nn.Module):                                       
    def __init__(self, d_k, d_v, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn = _MultiHeadAttention(d_k, d_v, d_model, n_heads)
        self.proj = Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)                               

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        context, attn = self.multihead_attn(q, k, v, attn_mask=attn_mask)

        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output = self.proj(context)
        return self.layer_norm(residual + output), attn    


class Transformer(nn.Module):                                            
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, num):
        super(Transformer, self).__init__()
        self.self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads)
        self.cls_token = nn.Parameter(torch.randn(1,  1, d_model))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Tanh(),
            nn.Linear(d_ff, 1))
        self.posmodel = nn.Parameter(torch.randn(1,  num+1, d_model))
        # sine and cosine positional encoding
        # self.sincos_posmodel = PosEncoding(num+1,d_model)
        # self.posmodel = self.sincos_posmodel(True)
        self.d_model = d_model

    def forward(self, enc_inputs):
        enc_inputs = torch.cat((enc_inputs, self.cls_token.expand(enc_inputs.shape[0], 1, self.d_model)), dim=1)
        enc_inputs += self.posmodel
        score1, attn = self.self_attn(enc_inputs, enc_inputs,
                                              enc_inputs, attn_mask=None)
        return self.ffn(score1[:,timestep,:])