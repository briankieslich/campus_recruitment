#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read the docs, obey PEP 8 and PEP 20 (Zen of Python, import this)

Build on:    Spyder
Python ver: 3.7.3
Author: Brian Kieslich
"""
"""
I found KNN Algorithm to be very easy. So I have done some coding according to my data science
and machine learning knowledge. I have uploaded my code. Suggestions are always welcome.

Expected Submission
I welcome everyone from kaggle community to upload their codes and see the difference in
output. Try different combinations of columns.
"""

# %%% modules:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import (GridSearchCV,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from itertools import combinations
from scipy.stats import zscore

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)


# %% data
"""
It is easy to predict 'status' on basis of 'salary': if 'nan' the 'Not Placed'.
So we will eliminate 'salary' and try to predict 'status' by the other features.
"""

df = (pd.read_csv('../data/Placement_Data_Full_Class.csv')
      .drop(columns=['sl_no', 'salary'])
      .pipe(pd.get_dummies, drop_first=True)
      )
print(df.info())


knn= KNeighborsClassifier()
y = df.pop('status_Placed')

# df = StandardScaler().fit_transform(df)
knn.fit(df, y)

print(knn.score(df, y))
# Not too good a score. we will expand our search

# %% Choosing the best features for knn to predict 'status'
df = (pd.read_csv('data/Placement_Data_Full_Class.csv')
      .drop(columns=['sl_no', 'salary'])
      .pipe(pd.get_dummies, drop_first=True)
      )

y = df.pop('status_Placed')

dd = pd.DataFrame(columns=['features', 'score', 'params'])

for i in range(4, 6, 1):
    for cols in combinations(df.columns, i):
        X = df.drop(columns=[*cols])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            stratify=y, random_state=1001)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # knn = KNeighborsClassifier(n_neighbors=5, leaf_size=30, algorithm='brute')

        knn = KNeighborsClassifier()

        params = {'n_neighbors': range(6, 13, 2),   # 8
          'p': [1, 2],          # 2
          'leaf_size': [1, 3, 8, 30],     # 30
          'weights': ['uniform', 'distance'],   # uniform
          'algorithm': ['auto']
          }
        skf = StratifiedKFold(n_splits=5, random_state=42)
        grid = GridSearchCV(estimator=knn, param_grid=params, cv=skf, n_jobs=-1)

        grid.fit(X_train, y_train)
        score = grid.score(X_test, y_test)

        dd = dd.append({'features': cols, 'score': score, 'params': grid.best_params_},
                       ignore_index=True)

#%% test and analyze
cols = 	['etest_p', 'hsc_s_Commerce', 'hsc_s_Science', 'degree_t_Sci&Tech', 'specialisation_Mkt&HR']

X = df.drop(columns=[*cols])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    stratify=y, random_state=1001)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=6, leaf_size=1, algorithm='auto', p=1, weights='distance')
# neighbors = 6, weights=distance

knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
# 0.94444444


y_pred = knn.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

# df_test.loc[y_test != y_pred, :]
# df_test.describe()









#%% Featureselection and crossvalidation
df = (pd.read_csv('data/Placement_Data_Full_Class.csv')
      .drop(columns=['sl_no', 'salary'])
      .pipe(pd.get_dummies, drop_first=True)
      )


# y = df.pop('status')
# X = df.drop(columns=['specialisation', 'degree_t', 'hsc_s', 'ssc_p', 'ssc_b'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
#                                                     stratify=y, random_state=1001)

# df = df.drop(columns=['specialisation', 'degree_t', 'hsc_s', 'ssc_p', 'ssc_b'])
# z_score = zscore(df)
# filter = (np.abs(z_score) < 3).all(axis=1)
# df = df.loc[filter, :]

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_train = df.iloc[:-54, :]
y_train = df_train.pop('status_Placed')
df_test = df.iloc[-54:, :]
y_test = df_test.pop('status_Placed')

# scaler = StandardScaler()
# X_train = scaler.fit_transform(df_train)
# X_test = scaler.transform(df_test)

knn = KNeighborsClassifier()

params = {'n_neighbors': range(5, 12, 1),   # 8
          'p': [1, 2],          # 2
          'leaf_size': range(25, 35, 1),     # 30
          'weights': ['uniform', 'distance'],   # uniform
          'algorithm': ['auto']
          }

grid = GridSearchCV(estimator=knn, param_grid=params, cv=5, n_jobs=15)

grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
print(grid.score(X_test, y_test))

y_pred = grid.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

df_test.loc[y_test != y_pred, :]
df_test.describe()
