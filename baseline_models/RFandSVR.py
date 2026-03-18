# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:50:44 2022

@author: Wang
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import math
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser()
parser.add_argument('--station', type = str, default = 'AAMU', help = '')
parser.add_argument('--depth', type = str, default = '0.0508', help = '')
parser.add_argument('--ismn', type = str, default = 'SCAN', help = '')
parser.add_argument('--path', type = str, default = 'E:/data', help = '')
arg = parser.parse_args()
for time in range(10):
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
    
    
    feature = 8
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
    data_timeseries = np.hstack((data_timeseries, theta_before))

    
    # formation of tensor
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
    
    x_test = data_timeseries[var_size:].reshape(-1, feature)
    y_test = theta_train_5[var_size:].reshape(-1, 1)
    telist = []
    for j in range(x_test.shape[0] - 4 - 7):
        if not np.isnan(np.sum(x_test[j:j+5+7])) and not np.isnan(np.sum(y_test[j:j+5+7])) :
            telist.append(j+var_size)    
    # form tensor
    
    traindata = x_train[tlist]
    label = y1_train[tlist]
    
    
#RF----------------------------------------------------------------------------------------------------------------------------------------------
    rf0 = RandomForestRegressor(random_state=0)
    rf0.fit(traindata, label)

   
    # search parameters
    # param_grid = {
    #     'bootstrap': [True], 
    #     # 'max_depth': [1], 
    #     # 'max_features': ['auto'], 
    #     # 'min_samples_leaf': [20],  
    #     # 'min_samples_split': [2, 11, 22],
    #     'n_estimators': range(1,101,10)
    #     # 'min_weight_fraction_leaf':[0,0.5],
    # }
    # grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
    #                               param_grid=param_grid, scoring='neg_mean_squared_error',
    #                               cv=5)
    # grid_search_rf.fit(traindata, label)
    
    
    # rf1 = RandomForestRegressor(**grid_search_rf.best_params_)
    
    # rf1.fit(traindata, label)
    
    
#SVR-------------------------------------------------------------------------------------------------------------------------------------------------------------
    clf = SVR(kernel='poly', C=1.0) # kernel='linear', C=1.5
    # search parameters
    # param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # param_grid = {"C":param_range, 'kernel':kernels}
    
    # grid_search_svr = GridSearchCV(estimator=clf,
    #                               param_grid=param_grid, scoring='neg_mean_squared_error',
    #                               cv=5)
    # grid_search_svr.fit(traindata, label)
    
    clf.fit(traindata, label)
    # grid_search_svr.fit(traindata, label)


    model = rf0 #/clf
    indexlist = np.random.choice(telist, len(telist), replace=False)
    batch = np.concatenate([data_timeseries[x] for x in indexlist], 0).reshape(-1, feature)
    theta = np.concatenate([theta_train_5[x] for x in indexlist], 0)
    
    #1 step
    theta_pre1 = model.predict(batch)
    theta1 = (theta+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta_pre1 = (theta_pre1+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    
    rmse1R = np.sqrt(mean_squared_error(theta1, theta_pre1)) / np.mean(theta1)
    L1r = mean_absolute_error(theta1, theta_pre1)/np.mean(theta1)
    
    theta_pre2 = model.predict(batch)
    theta2 = (theta+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta_pre2 = (theta_pre2+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    
    rmse_1 = np.sqrt(mean_squared_error(theta1, theta_pre1))
    mae_1 =  mean_absolute_error(theta1, theta_pre1)
    r2_1 = r2_score(theta1, theta_pre1)
    
    # 3 step
    indexlist_3 = indexlist
    batch_3 = np.concatenate([data_timeseries[x] for x in indexlist_3], 0).reshape(-1, feature)
    theta_3 = np.concatenate([theta_train_5[x:x+3] for x in indexlist_3], 0).reshape(-1, 3) #40,3
    
    
    for i in range(3):
        theta_pre1_33 = model.predict(batch_3).reshape(-1,1)
        batch_3 = np.concatenate([data_timeseries[x+1+i] for x in indexlist_3], 0).reshape(-1, feature)[:, :feature-1]
        batch_3 = np.concatenate((batch_3, theta_pre1_33),1)
        if i == 0:
            theta_pre1_3 = theta_pre1_33
        else:
            theta_pre1_3 = np.concatenate((theta_pre1_3, theta_pre1_33),1)
            
    theta1_3 = (theta_3+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta_pre1_3 = (theta_pre1_3+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    
    rmse_3 = np.sqrt(mean_squared_error(theta1_3, theta_pre1_3))
    mae_3 =  mean_absolute_error(theta1_3, theta_pre1_3)
    r2_3 = r2_score(theta1_3, theta_pre1_3)
    
    # 7 step
    indexlist_7 = indexlist
    batch_7 = np.concatenate([data_timeseries[x] for x in indexlist_7], 0).reshape(-1, feature)
    theta_7 = np.concatenate([theta_train_5[x:x+7] for x in indexlist_7], 0).reshape(-1, 7) #40,3
    
    
    for i in range(7):
        theta_pre1_77 = model.predict(batch_7).reshape(-1,1)
        batch_7 = np.concatenate([data_timeseries[x+1+i] for x in indexlist_7], 0).reshape(-1, feature)[:, :feature-1]
        batch_7 = np.concatenate((batch_7, theta_pre1_77),1)
        if i == 0:
            theta_pre1_7 = theta_pre1_77
        else:
            theta_pre1_7 = np.concatenate((theta_pre1_7, theta_pre1_77),1)
            
    theta1_7 = (theta_7+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    theta_pre1_7 = (theta_pre1_7+1)/2*(max_value_theta- min_value_theta) + min_value_theta
    
    rmse_7 = np.sqrt(mean_squared_error(theta1_7, theta_pre1_7))
    mae_7 =  mean_absolute_error(theta1_7, theta_pre1_7)
    r2_7 = r2_score(theta1_7, theta_pre1_7)


