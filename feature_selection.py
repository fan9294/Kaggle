# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:06:29 2020

@author: fanta
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.base import BaseEstimator, TransformerMixin
from skopt.space import Real, Integer



params = dict( learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)


param_test1 = {
  'reg_alpha':[65,70]
}


search1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.01, n_estimators=140, max_depth=5, reg_alpha=65,
                                                min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.8,
                                                objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=2020),
                      param_grid = param_test1,
                      scoring='roc_auc',n_jobs=3,iid=False, cv=5)
search1.fit(X_train,y_train)
search1.best_params_, search1.best_score_


search2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=1, reg_alpha = 1.15,
                                                min_child_weight=1, gamma=0.3, subsample=0.8, colsample_bytree=0,
                                                objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=2020),
                      param_grid = param_test2,
                      scoring='roc_auc',n_jobs=4,iid=False, cv=5)
search2.fit(prediction_bagged,y_train.iloc[index_bagged])
search2.best_params_, search1.best_score_

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
X_train = X_train.drop(["ACTION", 'ROLE_CODE'], axis=1)
X_test = X_test.drop(["id", 'ROLE_CODE'], axis = 1)


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
X = pd.concat((X_train, X_test), axis = 0)

te = TargetEncodingExpandingMean(columns_names = columns)
test = te.fit_transform(X_train,y_train)
test2 = te.transform(X_test)

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

cb_one = CatBoostClassifier(**one_step_params_dict)
cb_one.fit(test, y_train, early_stopping_rounds=30)
prediction = cb_one.predict_proba(test2)
prediction_one


submission = pd.DataFrame(np.arange(len(prediction)) + 1, columns = ['id'])
submission['Action'] = (prediction[:,-1] + prediction_one[:,-1])/2
submission.to_csv('submission.csv', index = False)

param_test1 = {
    'scale_pos_weight': [1,5,10], 
    }

search1 = GridSearchCV(estimator = CatBoostClassifier(n_estimators=200, # use large n_estimators deliberately to make use of the early stopping
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
                         silent=False),
                       param_grid = param_test1,
                       scoring='roc_auc',n_jobs=1, cv=3)
search1.fit(X_train,y_train)
search1.best_params_, search1.best_score_


param_test = {
     'reg_lambda':[900,1000,1100]
    }
search1 = GridSearchCV(estimator = LGBMClassifier(n_estimators=1000,
                                                  max_depth = 1,
                                                  num_leaves = 10,
                                                  colsample_bytree = .75,
                                                  subsample = .55,
                                                  min_child_weight = 7.5,
                                                  min_child_samples = 60,
                                                  metric='auc',
                                                  scale_pos_weight=2.75,
                                                  objective='binary',
                                                  learning_rate = 0.1,
                                                  random_state = 2020,
                                                  reg_lambda = 1000),
                       param_grid = param_test,scoring='roc_auc',n_jobs=1, cv=3, verbose = True )
search1.fit(X_train, y_train)
search1.best_params_, search1.best_score_
