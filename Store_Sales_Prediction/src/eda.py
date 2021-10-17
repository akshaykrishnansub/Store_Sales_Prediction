# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:32:20 2021

@author: Akshay
"""
import seaborn as sns
import matplotlib.pyplot as plt
import sp_logger

logger=sp_logger.print_loggers(__name__)


def dist_plots(combined, column):
    logger.info("Distribution plot for %s", column)
    plt.figure(figsize=(12,7))
    sns.distplot(combined[column], bins=25)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(column)
    plt.savefig("../data/plots/dist_plot_" + column + ".jpg")
    
    
def bar_plots(combined, column):
    plt.figure(figsize=(12,7))
    logger.info("Bar plot for %s ", column)
    if(column=='Item_Fat_Content'):
        fatcontent={'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'}
        combined.Item_Fat_Content.replace(fatcontent, inplace=True)
    combined[column].value_counts().plot.bar()
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title("Count Plot")
    plt.savefig("../data/plots/bar_plot_" + column + ".jpg")
    
    
def scatter_plots(combined, column, y):
    logger.info("Scatter plot for %s vs %s", column, y)
    plt.figure(figsize=(20,10))
    plt.scatter(combined[column], combined[y])
    plt.xlabel(column)
    plt.ylabel('Item_Outlet_Sales')
    plt.savefig("../data/plots/scatter_plot_" + column + ".jpg")
    
    
def violin_plots(combined, column, y):
    logger.info("Violin plot for %s vs %s", column, y)
    plt.figure(figsize=(20,10))
    sns.violinplot(combined[column], combined[y])
    plt.xticks(rotation='vertical')
    plt.savefig("../data/plots/violin_plot_" + column + ".jpg")
    
    
    
def univariate_analysis(combined):
    columns=['Item_Outlet_Sales','Item_Weight','Item_Visibility','Item_MRP']
    for column in columns:
        dist_plots(combined, column)
    columns=['Item_Type','Outlet_Size','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Type','Item_Fat_Content']
    for column in columns:
        bar_plots(combined, column)
    

def bivariate_analysis(combined):
    columns=['Item_Weight','Item_Visibility','Item_MRP']
    y='Item_Outlet_Sales'
    for X in columns:
        scatter_plots(combined, X, y)
        
    columns=['Item_Type','Outlet_Identifier','Item_Fat_Content','Outlet_Size','Outlet_Location_Type','Outlet_Type']
    y='Item_Outlet_Sales'
    for X in columns:
        violin_plots(combined, X, y)

    
def correlation_matrix(combined):
    logger.info("Plotting correlation matrix")
    corrmatrix=combined.corr()
    top_correlated_features=corrmatrix.index
    plt.figure(figsize=(20,20))
    g=sns.heatmap(combined[top_correlated_features].corr(),annot=True,cmap='RdYlGn')
    
    