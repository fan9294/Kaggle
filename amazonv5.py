# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:02:53 2020

@author: fanta
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from itertools import combinations
from sklearn.linear_model import LogisticRegressionCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from time import time
from numpy import hstack
from scipy import sparse
from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedShuffleSplit
from category_encoders import TargetEncoder
from scipy.sparse import hstack
from sklearn.model_selection import RandomizedSearchCV
from skopt.space import Real, Integer
from scipy.stats import uniform
from sklearn.base import BaseEstimator, TransformerMixin

random.seed(2020)
np.random.seed(2020)

def assign_rnd_integer(dataset, number_of_times = 5, seed = 2020):
    new_dataset = pd.DataFrame()
    np.random.seed(seed)
    for c in dataset.columns:
        for i in range(number_of_times):
            col_name = c+"_"+str(i)
            unique_vals = dataset[c].unique()
            labels = np.array(list(range(len(unique_vals))))
            np.random.shuffle(labels)
            mapping = pd.DataFrame({c: unique_vals, col_name: labels})
            new_dataset[col_name] = (dataset[[c]]
                                     .merge(mapping, on = c, how = 'left')[col_name]
                                    ).values
    return new_dataset


def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return pd.DataFrame(np.array(new_data).T)

def frequency_encoding(column, df, df_test=None):
    frequencies = df[column].value_counts().reset_index()
    df_values = df[[column]].merge(frequencies, how='left', 
                                   left_on=column, right_on='index').iloc[:,-1].values
    if df_test is not None:
        df_test_values = df_test[[column]].merge(frequencies, how='left', 
                                                 left_on=column, right_on='index').fillna(1).iloc[:,-1].values
    else:
        df_test_values = None
    return df_values, df_test_values

def assign_rnd_integer(dataset, number_of_times = 5, seed = 2020):
    new_dataset = pd.DataFrame()
    np.random.seed(seed)
    for c in dataset.columns:
        for i in range(number_of_times):
            col_name = c+"_"+str(i)
            unique_vals = dataset[c].unique()
            labels = np.array(list(range(len(unique_vals))))
            np.random.shuffle(labels)
            mapping = pd.DataFrame({c: unique_vals, col_name: labels})
            new_dataset[col_name] = (dataset[[c]]
                                     .merge(mapping, on = c, how = 'left')[col_name]
                                    ).values
    return new_dataset

class TargetEncodingExpandingMean(BaseEstimator, TransformerMixin):
    def __init__(self, columns_names):
        self.columns_names = columns_names
        self.learned_values = {}
        self.dataset_mean = np.nan
    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        self.learned_values = {}
        self.dataset_mean = np.mean(y)
        X_["__target__"] = y
        for c in [x for x in X_.columns if x in self.columns_names]:
            stats = (X_[[c,"__target__"]]
                     .groupby(c)["__target__"]
                     .agg(['mean', 'size'])) #
            stats["__target__"] = stats["mean"]
            stats = (stats
                     .drop([x for x in stats.columns if x not in ["__target__",c]], axis = 1)
                     .reset_index())
            self.learned_values[c] = stats
        return self
    def transform(self, X, **fit_params):
        transformed_X = X[self.columns_names].copy()
        for c in transformed_X.columns:
            transformed_X[c] = (transformed_X[[c]]
                                .merge(self.learned_values[c], on = c, how = 'left')
                               )["__target__"]
        transformed_X = transformed_X.fillna(self.dataset_mean)
        return transformed_X
    
    def fit_transform(self, X, y, **fit_params):
        self.fit(X,y)
    
        #Expanding mean transform
        X_ = X[self.columns_names].copy().reset_index(drop = True)
        X_["__target__"] = y
        X_["index"] = X_.index
        X_transformed = pd.DataFrame()
        for c in self.columns_names:
            X_shuffled = X_[[c,"__target__", "index"]].copy()
            X_shuffled = X_shuffled.sample(n = len(X_shuffled),replace=False)
            X_shuffled["cnt"] = 1
            X_shuffled["cumsum"] = (X_shuffled
                                    .groupby(c,sort=False)['__target__']
                                    .apply(lambda x : x.shift().cumsum()))
            X_shuffled["cumcnt"] = (X_shuffled
                                    .groupby(c,sort=False)['cnt']
                                    .apply(lambda x : x.shift().cumsum()))
            X_shuffled["encoded"] = X_shuffled["cumsum"] / X_shuffled["cumcnt"]
            X_shuffled["encoded"] = X_shuffled["encoded"].fillna(self.dataset_mean)
            X_transformed[c] = X_shuffled.sort_values("index")["encoded"].values
        return X_transformed



from catboost.datasets import amazon
X_train, X_test = amazon()

y_train = X_train["ACTION"]
X_train = X_train.drop(["ACTION", 'ROLE_CODE', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2'], axis=1)
X_test = X_test.drop(["id", 'ROLE_CODE', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2'], axis = 1)


X_train = pd.concat((X_train,
                     group_data(X_train.values, degree = 2),
                     group_data(X_train.values, degree = 3)), axis = 1)
X_test = pd.concat((X_test,
                     group_data(X_test.values, degree = 2),
                     group_data(X_test.values, degree = 3)), axis = 1)
X_train.columns = [str(i) for i in range(X_train.shape[1])]
X_test.columns = [str(i) for i in range(X_test.shape[1])]
columns = X_train.columns
num_train = X_train.shape[0]
categorical_features = list(range(460))

data_rnd = assign_rnd_integer(pd.concat((X_train,X_test), axis = 0))
X_train_freq = pd.DataFrame()
X_test_freq = pd.DataFrame()
for column in columns:
    train_values, test_values = frequency_encoding(column, X_train, X_test)
    X_train_freq[column+'_counts'] = train_values
    X_test_freq[column+'_counts'] = test_values
    
te = TargetEncodingExpandingMean(columns_names = columns)
X_train_encoded = te.fit_transform(X_train.iloc[:,0:92],y_train)
X_test_encoded = te.transform(X_test.iloc[:,0:92])    
X_train_encoded.columns = [column + '_encoded' for column in X_train_encoded.columns]
X_test_encoded.columns = [column + '_encoded' for column in X_test_encoded.columns]


X_train = pd.concat((data_rnd[:num_train],X_train_freq, X_train_encoded), axis = 1)
X_test= pd.concat((data_rnd[num_train:].reset_index(drop = True), X_test_freq, X_test_encoded), axis = 1)

X_train.to_pickle('X_train_2.pkl')
X_test.to_pickle('X_test_2.pkl')
y_train.to_pickle('y_train_2.pkl')


one_step_params_dict = {'bagging_temperature': 32.385109004124388,
 'boosting_type': 'Ordered',
 'colsample_bylevel': 0.5,
 'eval_metric': 'AUC',
 'learning_rate': 0.01,
 'loss_function': 'Logloss',
 'max_depth': 2,
 'n_estimators': 5000,
 'one_hot_max_size': 2,
 'random_seed': 2020,
 'random_strength': 100.0,
 'reg_lambda': 1.0,
 'scale_pos_weight': 3.1722273109999,
 'silent': False,
 'use_best_model': False}

cb_one = CatBoostClassifier(n_estimators=5000, # use large n_estimators deliberately to make use of the early stopping
                         max_depth = 5,
                         random_strength=7.5,
                         scale_pos_weight=1,
                         reg_lambda = 45,
                         bagging_temperature=0,
                         colsample_bylevel = .9,
                         one_hot_max_size=2,
                         learning_rate = 0.1,
                         loss_function='Logloss',
                         eval_metric='AUC',
                         boosting_type='Ordered', # use permutations
                         random_seed=2020,
                         silent=False)
cb_one.fit(X_train, y_train, early_stopping_rounds=30)
prediction_cat = cb_one.predict_proba(X_test)


xgb = XGBClassifier(learning_rate =0.01, n_estimators=5000, max_depth=5, 
                    reg_alpha=65, min_child_weight=1, gamma=0, 
                    subsample=0.9, colsample_bytree=0.8,
                    objective= 'binary:logistic',
                    scale_pos_weight=1, seed=2020)
xgb.fit(X_train, y_train, verbose = True)
prediciton_xgb = xgb.predict_proba(X_test)

params = {'colsample_bytree': 0.5280533549534434,
 'lambda_l1': 0.1267270702844549,
 'learning_rate': 0.001,
 'max_bin': 131,
 'max_depth': 18,
 'min_child_weight': 1.1518716916679328,
 'num_leaves': 184}
X_train = pd.read_pickle('X_train.pkl')
X_test = pd.read_pickle('X_test.pkl')
y_train = pd.read_pickle('y_train.pkl')

rus = RandomUnderSampler(random_state=2020)


light = LGBMClassifier(n_estimators=5000,
                       metric='auc',
                       objective='binary',
                       random_state = 2020,
                       **params)
light.fit(X_train,y_train)
prediction_light = light.predict_proba(X_test)
submission = pd.DataFrame(np.arange(len(prediction_light)) + 1, columns = ['id'])
submission['Action'] = (prediction_light[:,-1])
submission.to_csv('submission.csv', index = False)
prediction_final = np.zeros((X_test.shape[0], max(y_train)+1))
for i in range(20):
    X_train_sampled,y_train_sampled = rus.fit_resample(X_train,y_train)
    light = LGBMClassifier(n_estimators=1000,
                       metric='auc',
                       objective='binary',
                       random_state = 2020,
                       **params)
    light.fit(X_train_sampled,y_train_sampled)
    prediction_final += light.predict_proba(X_test)
    print('Round {} complete'.format(i))




    
submission = pd.DataFrame(np.arange(len(prediction_final)) + 1, columns = ['id'])
submission['Action'] = (prediction_final[:,-1]) / (i+1)
submission.to_csv('submission.csv', index = False)
