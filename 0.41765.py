# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:35:59 2020

@author: fanta
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from time import time

random.seed(2020)
np.random.seed(2020)


#%%
data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
le = LabelEncoder()
le.fit(data['target'])
y = le.transform(data['target'])
X = data.drop(['id', 'target'], axis = 1).values

data['features sum'] = X.sum(axis=1)
data['number of non-zeros'] = (X>0).sum(axis =1)
data['max value'] = X.max(axis=1)
X = data.drop(['id', 'target'], axis = 1).values

X_test = data_test.drop(['id'], axis = 1).values
data_test['features sum'] = X_test.sum(axis=1)
data_test['number of non-zeros'] = (X_test>0).sum(axis =1)
data_test['max value'] = X_test.max(axis=1)
X_test = data_test.drop(['id'], axis = 1).values


scaler = StandardScaler()
X = scaler.fit_transform(np.log(1+X))
X_test = scaler.fit_transform(np.log(1+X_test))

boostround = range(50)
prediction_test = np.zeros((len(X_test), max(y)+1))
for i in boostround:
    start = time()
    print('round {} starts, {} rounds remaining...'.format(i, boostround[-1]-i))
    # Bootstrap Sampling
    index_bagged = np.random.choice(range(X.shape[0]) ,size = X.shape[0])
    index_oob = [x for x in range(X.shape[0]) if x not in index_bagged]
    #Train the OOB model
    oob_rf =  RandomForestClassifier(n_estimators=300, criterion='entropy', 
                                     max_features='auto', bootstrap=False, 
                                     oob_score=False, n_jobs=8, verbose=0)
    oob_rf.fit(X[index_oob], y[index_oob])
    prediction_rf_bagged = oob_rf.predict_proba(X[index_bagged])
    prediction_rf_test = oob_rf.predict_proba(X_test)
    print('Random Forest round {} finished'.format(i))
    
    
    oob_xgb = XGBClassifier(basescore=0.5, colsamplebytree=0.8, gamma=0.03,
                    learningrate=0.25, maxdeltastep=0, maxdepth=8,
                    minchildweight=5.2475, n_estimators=50, n_jobs = 8,
                    objective='multi:softprob', silent=True, subsample=0.85)
    oob_xgb.fit(X[index_oob], y[index_oob])
    prediction_xgb_bagged = oob_xgb.predict_proba(X[index_bagged])
    prediction_xgb_test = oob_xgb.predict_proba(X_test)
    print('Xgboost round {} finished'.format(i))
    
    
    oob_cb = CatBoostClassifier(iterations = 500, depth = 8,
                                learning_rate=0.25, task_type = 'CPU', verbose = 0)
    oob_cb.fit(X[index_oob], y[index_oob])
    prediction_cb_bagged = oob_cb.predict_proba(X[index_bagged])
    prediction_cb_test = oob_cb.predict_proba(X_test)
    print('Catboost round {} finished'.format(i))
    
    
    oob_lb = LGBMClassifier(learning_rate=0.01, n_estimators=100)
    oob_lb.fit(X[index_oob], y[index_oob])
    prediction_lb_bagged = oob_lb.predict_proba(X[index_bagged])
    prediction_lb_test = oob_lb.predict_proba(X_test)
    print('Lightboost round {} finished'.format(i))        
    
    X_bagged = np.concatenate((X[index_bagged], prediction_rf_bagged,
                               prediction_xgb_bagged, prediction_cb_bagged,
                               prediction_lb_bagged),
                              axis = 1)
    X_test_bagged = np.concatenate((X_test, prediction_rf_test,
                               prediction_xgb_test, prediction_cb_test,
                               prediction_lb_test),
                              axis = 1)
    
    #Bag model
    model_bag = Sequential()
    model_bag.add(Dropout(0.1))
    model_bag.add(Dense(600, input_dim = X_bagged.shape[1], activation = 'relu'))
    model_bag.add(Dropout(0.3))
    model_bag.add((Dense(600, activation = 'relu')))
    model_bag.add(Dropout(0.1))
    model_bag.add((Dense(600, activation = 'relu')))
    model_bag.add((Dense(9, activation = 'softmax')))
    model_bag.compile(loss='categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
    model_bag.fit(X_bagged, to_categorical(y[index_bagged]), 
                    epochs = 100, batch_size = 64, verbose = 0)
    
    
    #Make predictions on the test set
    prediction_test += model_bag.predict_proba(X_test_bagged)
    print('Round {} complete! {} seconds elapsed'.format(i, time()- start))
    
submission = pd.DataFrame(prediction_test /(i+1), columns = ['Class_' + str(i+1) for i in range(9)])
submission['id'] = np.arange(len(submission)) + 1
cols = ['Class_' + str(i+1) for i in range(9)]
cols.insert(0,'id')
submission = submission[cols]
submission.to_csv('submission_latest.csv', index = False)