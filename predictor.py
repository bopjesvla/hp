#Has to be cleaned up: duplicate code!
#Makes predictions using gbr. Other models are in comments.

import pandas as pd
# import xgboost
import numpy as np
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search, model_selection
#%%
def preprocess(train=True):
    if train:
        df = pd.read_csv('train.csv')
        y = df['SalePrice']
    else:
        df = pd.read_csv('test.csv')
        y=None
    date = pd.to_datetime({'year': df['YrSold'], 'month': df['MoSold'], 'day': 1}).values.astype(int)
    df = df.assign(date=date).drop(['YrSold', 'Id', 'MiscVal', '3SsnPorch', 'LowQualFinSF', 'Utilities'], 1)
    df = df.drop(['EnclosedPorch', 'BsmtExposure', 'MasVnrArea', 'OpenPorchSF', 'LotShape', 'BsmtFinSF2', 'PoolArea'], 1)
    X = df[[c for c in df.columns if c != 'SalePrice']]
    X_float = X.select_dtypes(exclude=['object']).fillna(0)
    X_float.fillna('None', inplace=True)

    X_cat = pd.concat((X.select_dtypes(include=['object']).fillna('None'), y), axis=1)
    X = X_float
    X -= X.mean()
    X /= X.std()
    
    return X, X_cat, y
#%% 
X, X_cat, y = preprocess()

#%%
def get_score(model, split):
    
    X_train = split[0] 
    y_train = split[1] 
    X_test = split[2]
    y_test = split[3]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 1e-20)
    y_pred = np.minimum(755000, y_pred)
    
    score = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    return score
#%%
def train_some_model(X, X_cat, y, model='linear', params=None):
    ada_scores = []
    ridge_scores = []
    gbr_scores = []
    linear_scores = []
    knn_scores= []
    
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

        y_train = np.log(y_train)
        y_test = np.log(y_test)
        
        split = (X_train, y_train, X_test, y_test)
        
        dtr = DTR(criterion='mse', max_depth=None)
        ada = ensemble.AdaBoostRegressor(base_estimator=dtr, n_estimators=100, learning_rate=1.0, loss='exponential')
        score = get_score(ada, split)
        ada_scores.append(score)
        
        ridge = linear_model.Ridge(250)
        score = get_score(ridge, split)
        ridge_scores.append(score)
        
        gbr = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        score = get_score(gbr, split)
        gbr_scores.append(score)
        
        linear = linear_model.LinearRegression()
        score = get_score(linear, split)
        linear_scores.append(score)
        
#        is bad: 
#        knn = KNR(n_neighbors=10)
#        score = get_score(knn, split)
#        knn_scores.append(score)
        
#        print(dict(zip(X_train.columns, model.feature_importances_)))
#        print(X_train.columns[np.argpartition(model.feature_importances_, -4)[-4:]])
#        print(model.feature_importances_[np.argpartition(model.feature_importances_, -4)[-4:]])
    
        # print(model.feature_importances_)
    
        k += 1
    scores = [ada_scores, ridge_scores, gbr_scores, linear_scores, knn_scores]    
    return scores, k

def train_test(X_train, X_cat_train, y, model='linear', params=None):
    gbr_scores = []
    X_test, X_cat_test, _ = preprocess(False)

    sale_price_mean = X_cat['SalePrice'].mean()
    
    for c in X_cat.columns:
        if c == 'SalePrice':
            continue
        hood_price = X_cat_train.groupby(c)['SalePrice'].mean().reset_index()
        hood_price.columns = [c, c + '_mean_price']
        merged_train = X_cat_train.reset_index().merge(hood_price,  how='left', on=[c]).set_index('index')[c + '_mean_price']
        X_train = pd.concat((X_train, merged_train), axis=1)
        merged_test = X_cat_test.reset_index().merge(hood_price, how='left',  on=[c]).set_index('index')[c + '_mean_price'].fillna(sale_price_mean)
        X_test = pd.concat((X_test, merged_test), axis=1)

    y = np.log(y)
        
    split = (X_train, y, X_test)
        
    #dtr = DTR(criterion='mse', max_depth=None)
    #ada = ensemble.AdaBoostRegressor(base_estimator=dtr, n_estimators=100, learning_rate=1.0, loss='exponential')
    #score = get_score(ada, split)
    #ada_scores.append(score)
        
    #ridge = linear_model.Ridge(250)
    #score = get_score(ridge, split)
    #ridge_scores.append(score)
        
    gbr = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    X_train = split[0] 
    y_train = split[1] 
    X_test = split[2]

    
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    y_pred = np.maximum(y_pred, 1e-20)
    y_pred = np.minimum(755000, y_pred)
    y_pred=np.exp(y_pred)
        
    #linear = linear_model.LinearRegression()
    #score = get_score(linear, split)
    #linear_scores.append(score)
 
    return y_pred

#%%

predictions=train_test(X,X_cat,y)
pd.DataFrame({'Id': range(1461, 2920), 'SalePrice': predictions}).to_csv("Predictions_of_test.csv", index=False, header=True)
print(pd.DataFrame([predictions]))

