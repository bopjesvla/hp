# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:28:30 2018

@author: rbisschops
"""
import pandas as pd
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search

df = pd.read_csv('train.csv')
result=df.describe()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(result)