# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 15:35:35 2021

@author: Akshay
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sp_logger

logger=sp_logger.print_loggers(__name__)


def remove_null_values(combined):
    # Checking for missing values
    logger.info("Checking for missing values: %s", combined.isnull().sum())
    
    # Handling missing values in Item_Weight
    combined['Item_Weight'].fillna(combined['Item_Weight'].mean(),inplace=True)
    
    # Handling missing value for Outlet_Size
    combined['Outlet_Size'].fillna(combined['Outlet_Size'].mode()[0],inplace=True)
    
    # Handling missing value for Item_Outlet_Sales
    combined['Item_Outlet_Sales'].fillna(combined['Item_Outlet_Sales'].mean(),inplace=True)
    
    logger.info("Checking for missing values: %s", combined.isnull().sum())
    
    return combined
    

def feature_engineering(combined):
    # Creating a new feature Item_Type_new
    logger.info("Creating a new feature Item_Type_new")
    perishable = ['Breads','Breakfast','Dairy','Fruits and Vegetables','Meat','Seafood']
    non_perishable = ['Baking Goods','Canned','Frozen Foods','Hard Drinks','Health and Hygiene','Household','Soft Drinks']
    combined['Item_Type_New']=None
    for i in range(len(combined)):
        if combined['Item_Type'][i] in perishable:
            combined['Item_Type_New'][i] = 'Perishable'
        elif combined['Item_Type'][i] in non_perishable:
            combined['Item_Type_New'][i] = 'Non Perishable'
        else:
            combined['Item_Type_New'][i] = 'Not sure'
            
    # Creating Item_Category
    logger.info("Creating a new feature Item_Category")
    # step 1: we get the first two characters of ID
    combined['Item_Category'] = combined['Item_Identifier'].str.slice(0,2)
    # step 2: we will rename these into more understandable categories
    combined['Item_Category'] = combined['Item_Category'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
    combined.loc[combined['Item_Category'] == "Non-Consumable",'Item_Fat_Content'] = "Non_Edible"
    
    # Creating the Outlet_Years_Op feature
    logger.info("Creating a new feature Outlet_Years_Op")
    combined['Current_Year'] = 2021
    combined['Outlet_Years_Op'] = combined['Current_Year']-combined['Outlet_Establishment_Year']
    
    # Creating price_per_unit_wt feature
    logger.info("Creating a new feature price_per_unit_wt feature")
    combined['price_per_unit_wt'] = combined['Item_MRP']/combined['Item_Weight']
    
    logger.info("Performing label encoding")
    combined = label_encoding(combined)
    
    logger.info("Performing one hot encoding")
    combined = one_hot_encoding(combined)
    
    return combined
    
    
def label_encoding(combined):
    Item_MRP_Clusters = []
    for i in combined['Item_MRP']:
        if i < 69:
            Item_MRP_Clusters.append(1)
        elif i >= 69 and i < 136:
            Item_MRP_Clusters.append(2)
        elif i >= 136 and i < 203:
            Item_MRP_Clusters.append(3)
        else:
            Item_MRP_Clusters.append(4)
    combined['Item_MRP_Clusters'] = Item_MRP_Clusters
    le = LabelEncoder()
    var_mod = ['Outlet_Size','Outlet_Location_Type']
    for i in var_mod:
        combined[i] = le.fit_transform(combined[i])
    return combined
    
    
def one_hot_encoding(combined):
    combined = pd.get_dummies(combined, columns = ['Item_Fat_Content','Outlet_Type','Item_Type_New','Item_Category'])
    combined['Outlet_Type_Grocery_Store'] = combined['Outlet_Type_Grocery Store']
    combined['Outlet_Type_Supermarket_Type1'] = combined['Outlet_Type_Supermarket Type1']
    combined['Outlet_Type_Supermarket_Type2'] = combined['Outlet_Type_Supermarket Type2']
    combined['Outlet_Type_Supermarket_Type3'] = combined['Outlet_Type_Supermarket Type3']
    combined['Item_Type_New_Non_Perishable'] = combined['Item_Type_New_Non Perishable']
    combined['Item_Type_New_Not_Sure'] = combined['Item_Type_New_Not sure']
    combined['Item_Fat_Content_Low_Fat'] = combined['Item_Fat_Content_Low Fat']
    combined['Item_Category_Non_Consumable'] = combined['Item_Category_Non-Consumable']
    combined.drop(['Outlet_Type_Grocery Store','Outlet_Type_Supermarket Type1','Outlet_Type_Supermarket Type2','Outlet_Type_Supermarket Type3','Item_Type_New_Non Perishable','Item_Type_New_Not sure','Item_Fat_Content_Low Fat','Item_Category_Non-Consumable'],axis=1,inplace=True)
    combined.drop(['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Establishment_Year','Current_Year'],axis=1,inplace=True)
    return combined       