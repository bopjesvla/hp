# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:28:30 2018

@author: rbisschops
"""
import pandas as pd
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
result=df.describe()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(result)

#df.boxplot(column='SalePrice')
corrs=df.corr(method='pearson')
print(df.dtypes)
nonCont=[key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['float64','object']]
print(nonCont)
#print(corrs['SalePrice'])
var = df['Neighborhood']
salePrice=df['SalePrice']
print var
#varSale=[]
#nonVarSale=[]
#for index in range(0,len(salePrice)):
#    if var[index]!=0:
#        varSale.append(salePrice[index])
#    else:
#        nonVarSale.append(salePrice[index])
#dfPoolSale=pd.DataFrame(poolSale)
#dfNonPoolSale=pd.DataFrame(nonPoolSale)
#plt.boxplot([varSale,nonVarSale])
#plotVar=df['OverallQual']
#df.plot(x='OverallQual',y='SalePrice', kind='scatter')
#df.plot(x='Neighborhood',y='SalePrice',kind='box',subplots='True')
#df.boxplot(['SalePrice'], 'Neighborhood')
#
#unique, counts = np.unique(df['Neighborhood'], return_counts=True)
#print  np.asarray((unique, counts)).T

#cols_to_use = ['SalePrice'] # or [0,1,2,3]
#salePrice = pd.read_csv('train.csv', usecols= cols_to_use)
#variables = pd.read_csv('train.csv')
#variables = variables.drop(labels='SalePrice', axis=1)
#for column in variables:
#    if df[column].dtype=='int64' or df[column].dtype=='float64':
#        df.plot(x=column,y='SalePrice',kind='scatter',subplots='True')
#    else:
#        df.fillna('missing')
#        df.boxplot(['SalePrice'], column)

    