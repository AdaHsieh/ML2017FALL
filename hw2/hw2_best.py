
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import math
import xgboost as xgb

import random
from random import shuffle

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import os, sys
import argparse
from math import log, floor

import csv
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


# If you wish to get the same shuffle result
np.random.seed(2401)

def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)
    
    return (X_train, Y_train, X_test)

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))
    
    X_all, Y_all = _shuffle(X_all, Y_all)
    
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    
    return X_train, Y_train, X_valid, Y_valid

def train(X_all, Y_all):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    clf = XGB(X_train,Y_train,X_valid, Y_valid)
    return clf,X_valid, Y_valid
    
def valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return

def caculate(X,Y):
    x1,x2 = X[Y['label']==0],X[Y['label']==1]
    x1 = np.array(x1)
    x2 = np.array(x2)
    number1 = len(x1)
    number2 = len(x2)
    mean1=x1.mean(axis=0)
    mean2=x2.mean(axis=0)
    var1 = np.cov(x1, rowvar=False)
    var2 = np.cov(x2, rowvar=False)
    var = number1*var1/(number1+number2) + number2*var2/(number1+number2)
    w = np.dot(np.transpose(mean1-mean2),np.linalg.inv(var))
    b = -0.5*np.dot(np.transpose(mean1),np.dot(np.transpose(var),mean1))+0.5*np.dot(np.transpose(mean2),np.dot(np.transpose(var),mean2))+ math.log(number1/number2)
    return w,b

def XGB(trainX,trainY,X_valid, Y_valid):
    #gbm = xgb.XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.5).fit(trainX, trainY)
    gbm = xgb.XGBClassifier(max_depth=6,learning_rate =0.1, n_estimators=200, objective= 'binary:logistic')    
    y_pred = gbm.fit(trainX, trainY).predict(trainX)

    print("Number of mislabeled points out of a total %d points : %d"%
          (X_valid.shape[0],(Y_valid != y_pred).sum()))    
    joblib.dump(gbm, 'model.pkl') 
    #print(pd.crosstab(Y_valid[0], y_pred, rownames=['Actual'], colnames=['Predicted']))
    print(accuracy_score(Y_valid, gbm.predict(X_valid)))
    print(accuracy_score(trainY, gbm.predict(trainX)))
    return gbm

def main(*args):
    OX = args[0][1]
    OY = args[0][2]
    X_all, Y_all, X_test = load_data(args[0][3],args[0][4],args[0][5])
    
    #X_all, Y_all, X_test = load_data("X_train.csv","Y_train.csv","X_test.csv")
    
    # Normalization
    X_all, X_test = normalize(X_all, X_test)
    # clf,X_valid, Y_valid = train(X_all,Y_all)
    
    model = joblib.load('model.pkl')
    ans = model.predict(X_test)
    
    answer = []
    for i in range(len(ans)):
        answer.append([str(i+1)])
        answer[i].append(ans[i])

    filename = args[0][6]
    #filename = "result/predict.csv"
    
    text = open(filename, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(answer)):
        s.writerow(answer[i])
    text.close()

if __name__ == '__main__':
    main(sys.argv)


# In[ ]:




