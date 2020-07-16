#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read the docs, obey PEP 8 and PEP 20 (Zen of Python, import this)

Build on:    Spyder
Python ver: 3.7.3

Created on %(date)s

@author: %(username)s
"""

# %%% modules:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)


# %% data

df = (pd.read_csv('../data/Placement_Data_Full_Class.csv')
      .drop(columns=['sl_no', 'salary'])
      )

df.head()

# First some histograms
# without hue='status
fig, ax = plt.subplots(4, 2, sharey=True, figsize=(14, 10))
sns.countplot(data=df, x='gender', ax=ax[0, 0])
sns.countplot(data=df, x='ssc_b', ax=ax[0, 1])
sns.countplot(data=df, x='hsc_b', ax=ax[1, 0])
sns.countplot(data=df, x='hsc_s', ax=ax[1, 1])
sns.countplot(data=df, x='degree_t', ax=ax[2, 0])
sns.countplot(data=df, x='workex', ax=ax[2, 1])
sns.countplot(data=df, x='specialisation', ax=ax[3, 0])
sns.countplot(data=df, x='status', ax=ax[3, 1])
plt.tight_layout()
plt.show()

# with hue='status'
fig, ax = plt.subplots(4, 2, sharey=True, figsize=(14, 10))
sns.countplot(data=df, x='gender', hue='status', ax=ax[0, 0])
sns.countplot(data=df, x='ssc_b', hue='status', ax=ax[0, 1])
sns.countplot(data=df, x='hsc_b', hue='status', ax=ax[1, 0])
sns.countplot(data=df, x='hsc_s', hue='status', ax=ax[1, 1])
sns.countplot(data=df, x='degree_t', hue='status', ax=ax[2, 0])
sns.countplot(data=df, x='workex', hue='status', ax=ax[2, 1])
sns.countplot(data=df, x='specialisation', hue='status', ax=ax[3, 0])
sns.countplot(data=df, x='status', ax=ax[3, 1])
plt.tight_layout()
plt.show()
# In 'degree_t' 'Comm&Mgmt' is dominant, while in 'hsc_s' 'Arts' is underrepresented.


# Some relations between 'status' and the rest
fig, ax = plt.subplots(3, 2, sharey=True, figsize=(12, 16))
sns.boxenplot(data=df, x='ssc_b', y='ssc_p', hue='status', ax=ax[0, 0])
sns.boxenplot(data=df, x='degree_t', y='degree_p', hue='status', ax=ax[0, 1])
sns.boxenplot(data=df, x='specialisation', y='etest_p', hue='status', ax=ax[1, 0])
sns.boxenplot(data=df, x='hsc_b', y='hsc_p', hue='status', ax=ax[1, 1])
sns.boxenplot(data=df, x='hsc_s', y='hsc_p', hue='status', ax=ax[2, 0])
sns.boxenplot(data=df, x='hsc_b', y='hsc_p', hue='hsc_s', ax=ax[2, 1])
plt.tight_layout()
plt.show()


(pd.read_csv('data/Placement_Data_Full_Class.csv')
    .drop(columns=['sl_no', 'salary'])
    .replace('Not Placed', 'Not_Placed')
    .pipe(pd.get_dummies, columns=['gender', 'status'])
    .groupby(['degree_t', 'specialisation']).agg({
        'gender_F' : 'sum',
        'gender_M' : 'sum',
        'status_Not_Placed' : 'sum',
        'status_Placed' : 'sum',
        'mba_p' : ['min', 'mean', 'max']
          })
    )

#
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, figsize=(14, 8))
sns.boxenplot(data=df, x='status', y='ssc_p', ax=ax1)
sns.boxenplot(data=df, x='status', y='hsc_p', ax=ax2)
sns.boxenplot(data=df, x='status', y='degree_p', ax=ax3)
sns.boxenplot(data=df, x='status', y='etest_p', ax=ax4)
sns.boxenplot(data=df, x='status', y='mba_p', ax=ax5)
plt.tight_layout()
plt.show()




#%%
df_p = df.select_dtypes(include='float')
df_p.head()
lr = LinearRegression()

y_p = df_p.pop('mba_p')
df_p = StandardScaler().fit_transform(df_p)
lr.fit(df_p, y_p)
lr.score(df_p, y_p)



