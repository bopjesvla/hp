import pandas as pd
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search, model_selection
from random import *
import matplotlib.pyplot as plt
import warnings

gen = lambda n: [randint(0,1) for b in range(1,n+1)]

def rmsle(y,h):
	score = np.sqrt(np.square(np.log(np.maximum(0,h)+1) - np.log(np.maximum(0,y)+1)).mean())
	return score

scorer = metrics.make_scorer(rmsle)

def initData( filename ):
        df = pd.read_csv(filename)
        X = df[[c for c in df.columns if c != 'SalePrice']]
        X_float = X.select_dtypes(exclude=['object']).fillna(0)
        X_one_hot = pd.get_dummies(X.select_dtypes(include=['object']).fillna('None'))

        X = pd.concat((X_float, X_one_hot), axis=1)
        y = df['SalePrice']

        X.fillna('None', inplace=True)
        X = X.drop(['Id'],axis=1)

        return (X,y)

"""
Initialize folds for k-fold cross-validation
"""
def makeFolds( X,y,K ):
	folds = {}
	xfolds = []
	yfolds = []
	for k in range(K):
		xfolds.append([])
		yfolds.append([])
	for i in range(len(X)):
		fold = randint(0,K-1)
		xfolds[fold].append(X[i:i+1])
		yfolds[fold].append(y[i:i+1])
	folds = ([],[])
	for k in range(K):
		folds[0].append(pd.concat(xfolds[k]))
		folds[1].append(pd.concat(yfolds[k]))

#		folds[k] = (pd.concat(xfolds[k]),pd.concat(yfolds[k]))
	return folds

def trainModel(model,X,y):
	scores = model_selection.cross_validate(model,X,y,scoring=scorer,cv=5)
	return np.mean(scores['test_score'])

def OnePlusOne( seq,p,model,X,y,current_fitness ):
	new_seq = [not bit if random() < p else bit for bit in seq]
	new_fitness = trainModel( model, X[[X.columns[i] for i in range(n) if new_seq[i]]], y)
	return (new_seq,new_fitness) if new_fitness < current_fitness else (seq, current_fitness)

def plotFitness( array_of_fitness, title, xlabel, ylabel ):
	plt.plot(array_of_fitness)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

(X,y) = initData('train.csv')
K     = 10
E     = 100
folds = makeFolds(X,y,K)
model = linear_model.LinearRegression()

hist = np.zeros((E,K))
for k in range(K):
	X_train = pd.concat( [ folds[0][i] for i in range(K) if i!=k ] )
	y_train = pd.concat( [ folds[1][i] for i in range(K) if i!=k ] )
	X_test  = folds[0][k]
	y_test  = folds[1][k]
	
	n = len(X_train.columns)
	seq = gen(n)
	c_fit = trainModel(model, X_train[[X.columns[i] for i in range(n) if seq[i]]], y_train)
	performance = 0
	for i in range(E):
		(seq,c_fit) = OnePlusOne(seq,1/n,model,X_train,y_train,c_fit)
		model.fit(X_train[[X.columns[i] for i in range(n) if seq[i]]], y_train  )
		y_pred = model.predict(X_test)
		performance = rmsle(y_test,y_pred)

#		performance = trainModel(model, X_test[[X.columns[i] for i in range(n) if seq[i]]],y_test)
		hist[i][k] = performance
		print( 'Fold: ' + str(k) + ' Epoch: ' + str(i) + ': ' + str(performance) + "%                 ", end='\r')
	print('Fold: ' + str(k) + ' Performance: ' + str(performance) )


fitness = [np.mean(hist[i,:]) for i in range(E)]
plotFitness( fitness, "foobar", "xbar","ybar")
