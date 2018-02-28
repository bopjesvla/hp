import pandas as pd
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search, model_selection
from random import *
import matplotlib.pyplot as plt
import warnings

gen = lambda n: [randint(0,1) for b in range(1,n+1)]

def rmsle(y,h):
	score = np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())
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
	#X = X.drop(['Id'],axis=1)

	return (X,y)

def trainModel(model, X, y):
	scores = model_selection.cross_validate(model, X, y, scoring=scorer, cv=10)
	return np.mean(scores['test_score'])

def OnePlusOne(seq,p, model,X,y, current_fitness):
	new_seq = [not bit if random() < p else bit for bit in seq]
	new_fitness = trainModel( model, X[[X.columns[i] for i in range(n) if new_seq[i]]], y)
	return (new_seq,new_fitness) if new_fitness < current_fitness else (seq, current_fitness)

def plot_fitness( array_of_fitness, title, xlabel, ylabel ):
	plt.plot  ( array_of_fitness )
	plt.title ( title )
	plt.xlabel( xlabel )
	plt.ylabel( ylabel )
	plt.show  ()

(X,y) = initData( 'train.csv' )
model = linear_model.LinearRegression()
scores = trainModel( model, X, y )
#print(scores)
#print(len(X.columns))

n = len(X.columns)
hist = []
seq = gen(n)
c_fit = trainModel(model, X[[X.columns[i] for i in range(n) if seq[i]]], y)
baseline = trainModel(model, X, y)
print("baseline:" + str(baseline))
performance = 0
for i in range(100):
	(seq,c_fit) = OnePlusOne(seq,1/n,model,X,y, c_fit)
	performance = baseline / c_fit
	hist.append(c_fit)
	print( str(i) + ': ' + str(performance) + "%                 ", end='\r')

c_fit = trainModel(model, X[[X.columns[i] for i in range(n) if seq[i]]], y)
performance = baseline / c_fit
print("Baseline: " + str(baseline))
print("After training: " + str(c_fit) + "                 ")
print("Relative: " + str(performance))
print([X.columns[i] for i in range(n) if seq[i]])

plot_fitness( hist, 'Generative selecion of parameters', 'iteration', 'fitness')
