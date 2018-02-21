import pandas as pd
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search, model_selection


def rmsle(y, h):
    # print(y)
    score = np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())
    return score

scorer = metrics.make_scorer(rmsle)

df = pd.read_csv('train.csv')
date = pd.to_datetime({'year': df['YrSold'], 'month': df['MoSold'], 'day': 1}).astype(int).rename('date')
df = df.assign(date=date).drop('YrSold', 1)
X = df[[c for c in df.columns if c != 'SalePrice']]
X_float = X.select_dtypes(exclude=['object']).fillna(0)
X_one_hot = pd.get_dummies(X.select_dtypes(include=['object']).fillna('None'))
y = df['SalePrice']
extra = pd.concat((y, date), axis=1)
hist = extra.groupby('date').mean().shift(1).fillna(200000).reset_index()

# print(X_float[Xfloat.isnull().any(axis=1)])
# print(Xstr.columns)
X = pd.concat((X_float, X_one_hot), axis=1)
X = pd.merge(X, hist, on=['date'])
y = df['SalePrice']

X.fillna('None', inplace=True)
one_hot = pd.get_dummies(df)

# for c in one_hot.columns:
#     if len(one_hot[one_hot[c].isnull()]) > 0:
#         print()

model = linear_model.LinearRegression()
scores = model_selection.cross_validate(model, X.values, y.values, scoring=scorer, cv=10)
# print(scores)
# print(len(df))
# print(df.isnull().sum())
