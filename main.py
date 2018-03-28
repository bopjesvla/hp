# TODO: convert many categorical variables to mean price per category if large difference

import pandas as pd
# import xgboost
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search, model_selection
#%%
def preprocess():

    df = pd.read_csv('train.csv')
    date = pd.to_datetime({'year': df['YrSold'], 'month': df['MoSold'], 'day': 1}).astype(int).rename('date')
    df = df.assign(date=date).drop(['YrSold', 'Id', 'MiscVal', '3SsnPorch', 'LowQualFinSF', 'Utilities'], 1)
    df = df.drop(['EnclosedPorch', 'BsmtExposure', 'MasVnrArea', 'OpenPorchSF', 'LotShape', 'BsmtFinSF2', 'PoolArea'], 1)
    X = df[[c for c in df.columns if c != 'SalePrice']]
    X_float = X.select_dtypes(exclude=['object']).fillna(0)
    X_float.fillna('None', inplace=True)
    y = df['SalePrice']

    X_cat = pd.concat((X.select_dtypes(include=['object']).fillna('None'), y), axis=1)
    X = X_float
    X -= X.mean()
    X /= X.std()
    
    return X, X_cat, y
#%%
def train_some_model(X, X_cat, y):
    scores = []
    k = 0
    
    K = 10
    for train, test in model_selection.KFold(K, shuffle=True).split(X, y):
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    
        X_cat_train, X_cat_test = X_cat.iloc[train], X_cat.iloc[test]
        sale_price_mean = X_cat_train['SalePrice'].mean()
    
        for c in X_cat.columns:
            if c == 'SalePrice':
                continue
            hood_price = X_cat_train.groupby(c)['SalePrice'].mean().reset_index()
            hood_price.columns = [c, c + '_mean_price']
            merged_train = X_cat_train.reset_index().merge(hood_price,  how='left', on=[c]).set_index('index')[c + '_mean_price']
            X_train = pd.concat((X_train, merged_train), axis=1)
            merged_test = X_cat_test.reset_index().merge(hood_price, how='left',  on=[c]).set_index('index')[c + '_mean_price'].fillna(sale_price_mean)
            X_test = pd.concat((X_test, merged_test), axis=1)

    
        # model = linear_model.Ridge(250)
        model = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        # model = linear_model.LinearRegression()
        
        y_train = np.log(y_train)
        y_test = np.log(y_test)
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 1e-20)
        y_pred = np.minimum(755000, y_pred)
        score = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        scores.append(score)
#        print(dict(zip(X_train.columns, model.feature_importances_)))
        print(X_train.columns[np.argpartition(model.feature_importances_, -4)[-4:]])
        print(model.feature_importances_[np.argpartition(model.feature_importances_, -4)[-4:]])
    
        # print(model.feature_importances_)
    
        k += 1
        
    return scores, k
#%% 
X, X_cat, y = preprocess()
scores, k = train_some_model(X, X_cat, y)
print(np.mean(scores))
