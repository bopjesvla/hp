# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:28:30 2018

@author: rbisschops
"""
import pandas as pd
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search
from sklearn import tree
from sklearn import preprocessing
import matplotlib.pyplot as plt
import graphviz 

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
#var = df['Neighborhood']
salePrice=df['SalePrice']

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

#cols_to_use = ['SalePrice']
#salePrice = pd.read_csv('train.csv', usecols= cols_to_use)
#variables = pd.read_csv('train.csv')
#variables = variables.drop(labels='SalePrice', axis=1)
#for column in variables:
#    if df[column].dtype=='int64' or df[column].dtype=='float64':
#        df.plot(x=column,y='SalePrice',kind='scatter',subplots='True')
#    else:
#        df.fillna('missing')
#        df.boxplot(['SalePrice'], column)

variables = pd.read_csv('train.csv')
variables = variables.drop(labels='SalePrice', axis=1)
for column in variables:
    variables=variables.fillna(0)
    if not (variables[column].dtype=='int64' or variables[column].dtype=='float64'):
        le = preprocessing.LabelEncoder()
        le.fit(variables[column])
        variables[column]=le.transform(variables[column])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(variables, salePrice)
dot_data = tree.export_graphviz(clf, #out_file=None, 
                         feature_names=variables.columns.tolist(),  
                         #class_names=[""],  
                         filled=True, rounded=True  ,
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
print graph
