# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:11:35 2021

@author: Akshay
"""

import numpy as np
from sklearn import metrics
import config
import pickle


def evaluate_model(y_val, y_pred):
    mae = metrics.mean_absolute_error(y_val, y_pred)
    mse = metrics.mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    return mae, mse, rmse


def get_sales_value(model, X_val):
    y_pred = model.predict(X_val)
    return y_pred


def save_model(model):
    # Open a file, where we must store the data
    model_file = open(config.SAVE_MODEL_PATH, 'wb')

    # Dump information to that file
    pickle.dump(model,model_file)     