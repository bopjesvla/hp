import pandas as pd
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search, model_selection

df = pd.read_csv('train.csv')
X = df[[c for c in df.columns if c != 'SalePrice']]
X_float = X.select_dtypes(exclude=['object']).fillna(0)
X_one_hot = pd.get_dummies(X.select_dtypes(include=['object']).fillna('None'))
# print(X_float[Xfloat.isnull().any(axis=1)])
# print(Xstr.columns)
X = pd.concat((X_float, X_one_hot), axis=1)
y = df['SalePrice']

X.fillna('None', inplace=True)
one_hot = pd.get_dummies(df)

for c in one_hot.columns:
    if len(one_hot[one_hot[c].isnull()]) > 0:
        print()

model = linear_model.LinearRegression()
scores = model_selection.cross_validate(model, X, y, scoring=['neg_mean_squared_error'])
print(scores)
