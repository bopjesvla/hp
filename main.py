# TODO: convert many categorical variables to mean price per category if large difference

import pandas as pd
import random
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search, model_selection
from scipy import interpolate, stats

def rmsle(y, y_pred):
    y_pred = np.maximum(0, y_pred)
    # if not (h + 1).all() or not (y + 1).all():
    #     raise 'nope'
    score = np.sqrt(np.square(np.log(y_pred + 1.) - np.log(y + 1.)).mean())
    return score

scorer = metrics.make_scorer(rmsle)

df = pd.read_csv('train.csv')
# hood_price = df.groupby('Neighborhood')['SalePrice'].mean().reset_index()
# hood_price.columns = ['Neighborhood', 'hood_price']
# df = df.merge(hood_price, on=['Neighborhood'])

# date as unix timestamp
date = pd.to_datetime({'year': df['YrSold'], 'month': df['MoSold'], 'day': 1}).astype(int).rename('date')
df = df.assign(date=date).drop(['YrSold', 'Id'], 1)
X = df[[c for c in df.columns if c != 'SalePrice']]
X_float = X.select_dtypes(exclude=['object']).fillna(0)

y = df['SalePrice']
# date_price = pd.concat((y, X_float['date']), axis=1)
# mean_price_by_date = date_price.groupby('date').mean()
# hist = mean_price_by_date.shift(1).fillna(200000).reset_index()
# hist.columns = ['date', 'hist_price']
# print(stats.spearmanr(mean_price_by_date.values, hist['hist_price']))
# X_float = X_float.reset_index().merge(hist, on=['date']).set_index('index')

# normalizing
# X_float = pd.merge(X, hist, on=['date'])

X_cat = X.select_dtypes(include=['object']).fillna('None')
# is_ordinal = 

X_one_hot = pd.get_dummies(X_cat)

# print(X_float[Xfloat.isnull().any(axis=1)])
# print(Xstr.columns)
# X = pd.concat((X_float, X_one_hot), axis=1)
X = X_float
X -= X.mean()
X /= X.std()
# print(X['hist_price'])
y = df['SalePrice']

X.fillna('None', inplace=True)
one_hot = pd.get_dummies(df)

# for c in one_hot.columns:
#     if len(one_hot[one_hot[c].isnull()]) > 0:
#         print()

scores = []

def trainModel(model,X,y):
	scores = model_selection.cross_validate(model,X,y,scoring=scorer,cv=10)
	return np.mean(scores['test_score'])

def OnePlusOne( seq,p,model,X,y,current_fitness ):
	new_seq = [not bit if random.random() < p else bit for bit in seq]
	new_fitness = trainModel( model, X[[X.columns[i] for i in range(n) if new_seq[i]]], y)
	return (new_seq,new_fitness) if new_fitness < current_fitness else (seq, current_fitness)

K     = 10
E     = 100

perf_hist = np.zeros((E,K))
k = 0

for train, test in model_selection.KFold(K).split(X, y):
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    performance = 0
    n = len(X_train.columns)
    # seq = [random.randint(0,1) for b in X_train.columns]
    seq = [1 for b in X_train.columns]
    c_fit = 10000

#     for i in range(E):
#         model = linear_model.LinearRegression(normalize=False)
#         (seq,c_fit) = OnePlusOne(seq,1/n,model,X_train,y_train,c_fit)
#         model.fit(X_train[X.columns[seq]], y_train  )
#         y_pred = model.predict(X_test[X.columns[seq]])
#         y_pred = np.maximum(y_pred, 0)
#         performance = rmsle(y_test, y_pred)

#         if i == 0:
#             print('Init test performance:', performance)

# #		performance = trainModel(model, X_test[[X.columns[i] for i in range(n) if seq[i]]],y_test)
#         perf_hist[i][k] = performance
#         print( 'Fold: ' + str(k) + ' Epoch: ' + str(i) + ' Train perf: ' + str(c_fit) + ' Test perf: ' + str(performance) + "%                 ", end='\r')
#     print('Fold: ' + str(k) + ' Performance: ' + str(performance) )

    model = linear_model.Ridge(1000)
    # model = linear_model.LinearRegression()
    # model.fit(X_train[X.columns[seq]], y_train)
    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test[X.columns[seq]])
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    score = rmsle(y_test, y_pred)
    scores.append(score)

    k += 1

# scores = model_selection.cross_validate(model, X.values, y.values, scoring=scorer, cv=5)
# print(model.coef_)
print(np.mean(scores))
# print(len(df))
# print(df.isnull().sum())
