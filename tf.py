import time
import random
import pandas as pd
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search, model_selection
from scipy import interpolate, stats
import tensorflow as tf
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def loadData():
    df = pd.read_csv('train.csv')
    X = df[[c for c in df.columns if c != 'SalePrice']]
    X_float = X.select_dtypes(exclude=['object']).fillna(0)
    y = df['SalePrice']
    X.fillna('None', inplace=True)
    one_hot = pd.get_dummies(df)
    return (one_hot, y)

(X,Y) = loadData()

current_row = 160
def next_data_batch(batch_size):
    global current_row
    if current_row + batch_size >= X.shape[0]:
        x_out = X.iloc[current_row:].values.tolist()
        y_out = Y.iloc[current_row:].values.tolist()
        current_row = 160
    else:
        x_out = X.iloc[current_row:(current_row+batch_size)].values.tolist()
        y_out = Y.iloc[current_row:(current_row+batch_size)].values.tolist()
        current_row += batch_size
    x_out = [[0 if math.isnan(e) else e for e in row] for row in x_out]
    y_out = [0 if math.isnan(e) else e for e in y_out]
    return (x_out, y_out)

def test_batch():
    x_out = X.iloc[0:160].values.tolist()
    y_out = Y.iloc[0:160].values.tolist()
    x_out = [[0 if math.isnan(e) else e for e in row] for row in x_out]
    y_out = [0 if math.isnan(e) else e for e in y_out]
    return (x_out, y_out)

n_input     = X.shape[1]
n_nodes_hl1 = 150
n_nodes_hl2 = 50
n_output    = 1

batch_size = 100

x = tf.placeholder('float',[None, n_input])
y = tf.placeholder('float')

def ANN_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.ones([n_input, n_nodes_hl1])),
                      'biases': tf.Variable(tf.ones([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.ones([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.ones([n_nodes_hl2]))}
    
    output_layer   = {'weights':tf.Variable(tf.ones([n_input,n_output])),
                      'biases': tf.Variable(tf.ones([n_output]))}

    #l1 = tf.add( tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    #l1 = tf.nn.relu(l1)

    #l2 = tf.add( tf.matmul( l1 , hidden_2_layer['weights']), hidden_2_layer['biases'])
    #l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul( data, output_layer['weights']), output_layer['biases'])
    output = tf.nn.relu(output)

    return output

def train_ANN(x):
    prediction = ANN_model(x)
    cost = tf.sqrt( tf.reduce_mean( tf.square( tf.subtract( tf.log( tf.nn.relu(prediction) + 1), tf.log( y + 1 ) ) ) ) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 1000

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int( (X.shape[0]-160) / batch_size )):
                (epoch_x, epoch_y) = next_data_batch( batch_size )
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch:',epoch,'completed out of',hm_epochs,'loss:',epoch_loss)

        rmsle = tf.sqrt( tf.reduce_mean( tf.square( tf.subtract( tf.log( tf.nn.relu(prediction) + 1), tf.log( y + 1 ) ) ) ) )
        (x_test,y_test) = test_batch()
        print('RMLSE:',rmsle.eval({x: x_test, y: y_test}))       

train_ANN(x)
