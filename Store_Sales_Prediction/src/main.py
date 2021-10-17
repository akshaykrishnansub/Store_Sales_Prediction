# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:32:20 2021

@author: Akshay
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import config
import sp_logger
import preprocess,eda,model,evaluate

warnings.filterwarnings('ignore')
logger=sp_logger.print_loggers(__name__)


def store_sales_prediction():
    train = pd.read_csv(config.TRAIN_CSV_PATH)
    test = pd.read_csv(config.TEST_CSV_PATH)
    #train.shape
    #test.shape
    #train.dtypes
    #test.dtypes
    train['source']='train'
    test['source']='test'
    combined=pd.concat([train,test], ignore_index=True)
    #print(train.shape,test.shape,combined.shape)
    
    combined=preprocess.remove_null_values(combined)
    logger.info("Null values removed")
    
    eda.univariate_analysis(combined)
    logger.info("Univariate analysis done")
    eda.bivariate_analysis(combined)
    logger.info("Bivariate analysis done")
    eda.correlation_matrix(combined)
    logger.info("Correlation matrix done")
    
    combined=preprocess.feature_engineering(combined)
    logger.info("Feature engineering done")
    
    train = combined.loc[combined['source']=="train"]
    test = combined.loc[combined['source']=="test"]
    
    train.drop(['source'], axis=1, inplace=True)
    test.drop(['Item_Outlet_Sales','source'], axis=1, inplace=True)
    
    X = train.drop(columns=['Item_Outlet_Sales'])
    y = train['Item_Outlet_Sales']
    print(list(X.columns))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)
    logger.info("Train Test split done")    
    
    logger.info("Calling Linear Regression")
    lr = model.linear_regression(X_train, y_train)
    y_pred = evaluate.get_sales_value(lr, X_val)
    mae, mse, rmse = evaluate.evaluate_model(y_val, y_pred)
    logger.info("Performance of linear regression model: MAE = %.4f, MSE = %.4f, RMSE = %.4f", mae, mse, rmse)
    
    logger.info("Calling Ridge Regression")
    ridge = model.ridge_regression(X_train, y_train)
    y_pred = evaluate.get_sales_value(ridge, X_val)
    mae, mse, rmse = evaluate.evaluate_model(y_val, y_pred)
    logger.info("Performance of ridge regression model: MAE = %.4f, MSE = %.4f, RMSE = %.4f", mae, mse, rmse)
    
    logger.info("Calling Lasso Regression")
    lasso = model.lasso_regression(X_train, y_train)
    y_pred = evaluate.get_sales_value(lasso, X_val)
    mae, mse, rmse = evaluate.evaluate_model(y_val, y_pred)
    logger.info("Performance of lasso regression model: MAE = %.4f, MSE = %.4f, RMSE = %.4f", mae, mse, rmse)
    
    logger.info("Calling Decision Tree Regression")
    dt = model.decision_tree(X_train, y_train)
    y_pred = evaluate.get_sales_value(dt, X_val)
    mae, mse, rmse = evaluate.evaluate_model(y_val, y_pred)
    logger.info("Performance of decision tree regression model: MAE = %.4f, MSE = %.4f, RMSE = %.4f", mae, mse, rmse)
    
    logger.info("Calling Random Forest Regression")    
    rf = model.random_forest(X_train, y_train)
    y_pred = evaluate.get_sales_value(rf, X_val)
    mae, mse, rmse = evaluate.evaluate_model(y_val, y_pred)
    logger.info("Performance of random forest regression model: MAE = %.4f, MSE = %.4f, RMSE = %.4f", mae, mse, rmse)

    logger.info("Saving the best model")
    evaluate.save_model(rf)


if __name__ == "__main__":
    store_sales_prediction()   