# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:57:07 2021

@author: Akshay
"""

LOGFILE = '../logs/project_logs.log'

TRAIN_CSV_PATH = '../data/csvs/train.csv'
TEST_CSV_PATH = '../data/csvs/test.csv'

SAVE_MODEL_PATH = '../models/rf_model.pkl'

HYPER_PARAM_RIDGE = {'alpha':[0.0001,0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,20.0,30.0,40.0,50.0,100.0,500.0,1000.0]}
HYPER_PARAM_LASSO = {'alpha':[0.0001,0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,10.0,50.0,100.0,500.0,1000.0]}
HYPER_PARAM_DT = {
    'max_depth': [2,3,5,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000],
    'min_samples_leaf': [5,10,20,50,100,200]
}
HYPER_PARAM_RF = {
    'max_depth': [5,10,20,30,40],
    'min_samples_leaf': [30,40,50,60,100,200],
    'n_estimators': [20, 30, 40, 50, 100]
}