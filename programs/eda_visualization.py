#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read the docs, obey PEP 8 and PEP 20 (Zen of Python, import this)

Build on:    Spyder
Python ver: 3.7.3
Author: Brian Kieslich
"""

# %%% modules:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)
pd.set_option('plotting.backend', 'holoviews')
sns.set(font_scale = 1.4, rc={'legend.fontsize': 10})


# %% data

df = (pd.read_csv('../data/Placement_Data_Full_Class.csv',
                  dtype= {'gender': 'category',
                          'ssc_b': 'category',
                          'hsc_b': 'category',
                          'hsc_s': 'category',
                          'hsc_b': 'category',
                          'degree_t': 'category',
                          'workex': 'category',
                          'specialisation': 'category',
                          'status': 'category'})
      .drop(columns=['sl_no'])
      .replace('Not Placed', 'Not_Placed')

      )

df.head()
df.nunique()
df.info()


"""
Many relations to investigate, but I will focus on relations that can help answer if:
    1. Can 'mba_p' be predicted from the previous features,
    2. Can 'status' be predicted from the previous features, and
    3. Can 'salary' be predicted from the previous features, with or without 'status'.

"""

# We start with the categorical features
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
# In 'degree_t' 'Comm&Mgmt' is dominant, while in 'hsc_s' 'Arts' is almost missing.


# Some relations between 'category' features and their respective scores
fig, ax = plt.subplots(3, 2, sharey=True, figsize=(12, 16))
sns.boxenplot(data=df, x='ssc_b', y='ssc_p', ax=ax[0, 0])
sns.boxenplot(data=df, x='degree_t', y='degree_p', ax=ax[0, 1])
sns.boxenplot(data=df, x='specialisation', y='etest_p', ax=ax[1, 0])
sns.boxenplot(data=df, x='hsc_b', y='hsc_p', ax=ax[1, 1])
sns.boxenplot(data=df, x='hsc_s', y='hsc_p', ax=ax[2, 0])
sns.boxenplot(data=df, x='hsc_b', y='hsc_p', hue='hsc_s', ax=ax[2, 1])
plt.tight_layout()
plt.show()


# And these relations split on 'status'
fig, ax = plt.subplots(3, 2, sharey=True, figsize=(12, 16))
sns.boxenplot(data=df, x='ssc_b', y='ssc_p', hue='status', ax=ax[0, 0])
sns.boxenplot(data=df, x='degree_t', y='degree_p', hue='status', ax=ax[0, 1])
sns.boxenplot(data=df, x='specialisation', y='etest_p', hue='status', ax=ax[1, 0])
sns.boxenplot(data=df, x='hsc_b', y='hsc_p', hue='status', ax=ax[1, 1])
sns.boxenplot(data=df, x='hsc_s', y='hsc_p', hue='status', ax=ax[2, 0])
sns.boxenplot(data=df, x='hsc_b', y='hsc_p', hue='hsc_s', ax=ax[2, 1])
plt.tight_layout()
plt.show()

pd.pivot_table(df.loc[df.hsc_b == 'Others', :],
               values='hsc_p',
               index='hsc_s',
               columns='status')


sns.catplot(data=df, y='hsc_p', hue='status', x='hsc_b', col='hsc_s', kind='boxen', orient='v')



(pd.read_csv('data/Placement_Data_Full_Class.csv')
   .drop(columns=['sl_no'])
   .replace('Not Placed', 'Not_Placed')
   .pipe(pd.get_dummies, columns=['gender', 'status'])
   .groupby(['degree_t', 'specialisation']).agg({
       'gender_F' : 'sum',
       'gender_M' : 'sum',
       'status_Not_Placed' : 'sum',
       'status_Placed' : 'sum',
       'mba_p' : ['mean'],
       'salary' : ['mean']
       })
)



# 'status' versus scores(points)
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, figsize=(14, 8))
sns.boxenplot(data=df, x='status', y='ssc_p', ax=ax1)
sns.boxenplot(data=df, x='status', y='hsc_p', ax=ax2)
sns.boxenplot(data=df, x='status', y='degree_p', ax=ax3)
sns.boxenplot(data=df, x='status', y='etest_p', ax=ax4)
sns.boxenplot(data=df, x='status', y='mba_p', ax=ax5)
plt.tight_layout()
plt.show()
# 'ssc_p', 'hsc_p' and 'degree_p' might be usefull in prediction 'status'.

# relations between categorical and mba_p / salary ????

# Relations between the scores
sns.pairplot(df)
# No obvious relations, at least not linear.
# And the distributions are nearly(without glasses) normal.
sns.pairplot(df.loc[df.status == 'Not Placed', :])
sns.pairplot(df.loc[df.status != 'Not Placed', :])
# And a big No to any (linear) relations
# implement df.skew()




#%% lmplot test
df.columns
cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
sns.lmplot(x='mba_p', y='salary', data=df, row='hsc_s', col='workex')
