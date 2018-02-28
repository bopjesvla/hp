import pandas as pd
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search, model_selection

def rmsle(y, h):
    # if not (h + 1).all() or not (y + 1).all():
    #     raise 'nope'
    score = np.sqrt(np.square(np.log(h + 1.) - np.log(y + 1.)).mean())
    return score

scorer = metrics.make_scorer(rmsle)

df = pd.read_csv('train.csv')

# date as unix timestamp
date = pd.to_datetime({'year': df['YrSold'], 'month': df['MoSold'], 'day': 1}).astype(int).rename('date')
df = df.assign(date=date).drop(['YrSold', 'Id'], 1)
X = df[[c for c in df.columns if c != 'SalePrice']]
X_float = X.select_dtypes(exclude=['object']).fillna(0)

y = df['SalePrice']
extra = pd.concat((y, X_float['date']), axis=1)
hist = extra.groupby('date').mean().shift(1).fillna(200000).reset_index()
hist.columns = ['date', 'hist_price']
# X_float.merge()

# min-max scaling
# X_float -= X_float.mean()
# X_float /= X_float.std()
# X_float = pd.merge(X, hist, on=['date'])

X_one_hot = pd.get_dummies(X.select_dtypes(include=['object']).fillna('None'))

hist.columns = ['date', 'hist_price']
# X_float.merge()

# normalizing
X_float -= X_float.mean()
X_float /= X_float.std()
# X_float = pd.merge(X, hist, on=['date'])

X_one_hot = pd.get_dummies(X.select_dtypes(include=['object']).fillna('None'))

# print(X_float[Xfloat.isnull().any(axis=1)])
# print(Xstr.columns)
X = pd.concat((X_float, X_one_hot), axis=1)
# print(X['hist_price'])
y = df['SalePrice']

X.fillna('None', inplace=True)
one_hot = pd.get_dummies(df)

# for c in one_hot.columns:
#     if len(one_hot[one_hot[c].isnull()]) > 0:
#         print()

scores = []

for train, test in model_selection.KFold(10).split(X, y):
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    model = linear_model.LinearRegression(normalize=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    score = rmsle(y_test, y_pred)
    scores.append(score)

# scores = model_selection.cross_validate(model, X.values, y.values, scoring=scorer, cv=5)
# print(model.coef_)
print(scores)
# print(len(df))
# print(df.isnull().sum())
