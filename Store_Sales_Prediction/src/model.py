# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:29:55 2021

@author: Akshay
"""

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import config
import sp_logger

logger=sp_logger.print_loggers(__name__)


def hyperparameter_tuning(model, model_key, X_train, y_train):
    logger.info("Performing hyperparameter tuning")
    folds = KFold(n_splits=10,shuffle=True,random_state=0)
    try:
        if model_key=='ridge':
            hyper_param=config.HYPER_PARAM_RIDGE
        elif model_key=='lasso':
            hyper_param=config.HYPER_PARAM_LASSO
        elif model_key=='dt':
            hyper_param=config.HYPER_PARAM_DT
        elif model_key=='rf':
            hyper_param=config.HYPER_PARAM_RF
    
        model_cv = GridSearchCV(estimator = model,
                        param_grid = hyper_param,
                        scoring = 'r2',
                        cv = folds,
                        verbose = 1,
                        return_train_score = True
                       )
        model_cv.fit(X_train, y_train)
    except:
        logger.error("Hyperparameter tuning error", exc_info=True)
    return model_cv


def linear_regression(X_train, y_train):
    try:
        logger.info("Training Linear Regression")
        lr = LinearRegression(normalize = True)
        lr.fit(X_train, y_train)
    except:
        logger.error("Training linear regression error", exc_info=True)
    return lr


def ridge_regression(X_train, y_train):
    try:
        logger.info("Training Ridge Regression")
        ridge = Ridge(alpha = 5.0,normalize = True)
        model_cv = hyperparameter_tuning(ridge,'ridge', X_train, y_train)
        ridge.fit(X_train, y_train)
    except:
        logger.error("Training ridge regression error", exc_info=True)    
    return ridge
    

def lasso_regression(X_train, y_train):
    try:
        logger.info("Training Lasso Regression")
        lasso = Lasso(alpha = 2.0, normalize = True)
        model_cv = hyperparameter_tuning(lasso, 'lasso', X_train, y_train)
        lasso.fit(X_train,y_train)
    except:
        logger.error("Training lasso regression error", exc_info=True)    
    return lasso


def decision_tree(X_train, y_train):
    try:
        logger.info("Training Decision Tree Regression")
        dt = DecisionTreeRegressor(criterion = 'mse',max_depth = 5,min_samples_leaf = 100,random_state = 0)
        model_cv = hyperparameter_tuning(dt, 'dt', X_train, y_train)
        dt.fit(X_train, y_train)
    except:
        logger.error("Training decision tree regression error", exc_info=True)
    return dt
    

def random_forest(X_train, y_train):
    try:
        logger.info("Training Random Forest Regression")
        rf = RandomForestRegressor(n_estimators = 40, max_depth = 10, min_samples_leaf = 40, random_state = 0)
        model_cv = hyperparameter_tuning(rf,'rf',X_train,y_train)
        rf.fit(X_train, y_train)
    except:
        logger.error("Training random forest regression error", exc_info=True)
    return rf