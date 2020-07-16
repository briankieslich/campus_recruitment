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
import category_encoders as ce

from sklearn.pipeline import Pipeline
from sklearn.linear_model import (LinearRegression,
                                  Lasso,
                                  LassoLarsCV,
                                  )
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error,
                             r2_score,
                             )
from sklearn.feature_selection import (SelectFromModel,
                                       SelectKBest,
                                       mutual_info_regression,
                                       f_regression,
                                       )
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import (boxcox,
                         zscore,
                         )
from scipy.special import inv_boxcox
from custom_transformers import (DFStandardScaler,
                                 DFMinMaxScaler,
                                 TypeExtractor,
                                 DFFeatureUnion,
                                 DFPolynomial,
                                 MixCatNum,
                                 ColumnExtractor,
                                 )

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)
sns.set(font_scale = 1, rc={'legend.fontsize': 10})



# %% data

# Setting up for predicting 'salary'
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
      .dropna()
      .assign(outlier=lambda x: (np.abs(zscore(x.select_dtypes('number'))) < 3).all(axis=1))
      .dropna()
      .drop(columns=['status', 'outlier'])
      )

y = df.pop('salary')

# df.head()
# df.shape

#%% Preprocessor functions
ohe = ce.OneHotEncoder(drop_invariant=True,
                       return_df=True,
                       use_cat_names=True,
                       handle_missing='return_nan') # Remember replace(np.nan, 0)

tge = ce.TargetEncoder(drop_invariant=True,
                       return_df=True,
                       handle_missing='value',
                       # min_samples_leaf=3,
                       # smoothing=0.4,
                       )

num_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
new_cat_cols = ['gender_M', 'gender_F',
                'ssc_b_Others', 'ssc_b_Central',
                'hsc_b_Others', 'hsc_b_Central',
                'hsc_s_Commerce', 'hsc_s_Science', 'hsc_s_Arts',
                'degree_t_Sci&Tech', 'degree_t_Comm&Mgmt', 'degree_t_Others',
                'workex_No', 'workex_Yes',
                'specialisation_Mkt&HR', 'specialisation_Mkt&Fin']

# Preprocessing with a Pipeline
preprocess = Pipeline([
    ('features', DFFeatureUnion([
        ('categorical_o', Pipeline([
            ('extract', TypeExtractor('category')),
            ('onehot', ohe),
            # ('imputer', DFImputer())
        ])),
        ('categorical_t', Pipeline([
            ('extract', TypeExtractor('category')),
            ('target', tge),
            ('catscale', DFMinMaxScaler())
        ])),
        ('numerics', Pipeline([
            ('extract', TypeExtractor('number')),
            ('polynomial', DFPolynomial()),
            ('catscale', DFMinMaxScaler())
        ])),
    ])),
    ('mixcatnum', MixCatNum(new_cat_cols, num_cols))
])

params = {
    'features__categorical_t__target__min_samples_leaf': 7,
    'features__categorical_t__target__smoothing': 0.1,
    }

preprocess.set_params(**params)

X = preprocess.fit_transform(df, y)

#%% Feature selection with DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            random_state=42)
dtr.fit(X, y)
y_pred = dtr.predict(X)
print(f'Root mean: {mean_squared_error(y, y_pred)**0.5:.2f}')
print(f'r2_score : {r2_score(y, y_pred):.2f}')

dd = pd.DataFrame({'name': X.columns, 'imp': dtr.feature_importances_})
dd = dd.sort_values('imp', ascending=False)
dd.tail()
dd = dd.loc[dd.imp > 0, :]
sns.catplot(data=dd, y='name', x='imp', kind='bar', height=10, orientation='horizontal')

short_cols = dd.name.values

#%% Correlation and other things

params = {
    'features__categorical_t__target__min_samples_leaf': 7,
    'features__categorical_t__target__smoothing': 0.1,
    }

preprocess.set_params(**params)

X = preprocess.fit_transform(df, y)

X['y'] = y
X_cor = (np.abs(X.corr())
         .sort_values('y', ascending=False)
         .sort_values('y', axis=1, ascending=False)
         )

#Correlation with output variable
X_cor_y = X_cor['y']

relevant_features = X_cor_y[X_cor_y>0.1].index.to_numpy()[1:]
X = X.loc[:, relevant_features]

corr_columns = np.full((relevant_features.size), True, dtype=bool) # Make an array of True's

for i in range(relevant_features.size - 1):
    for j in range(i+1, relevant_features.size):
        if X_cor.iloc[i,j] >= 0.9:
            corr_columns[j] = False

X = X.loc[:, corr_columns]


short_cols = X.columns.to_list()
# 1, 0.5, ['etest_p_mba_p', 'gender_M_mba_p', 'gender_M_etest_p', 'etest_p_etest_p', 'mba_p', 'mba_p_mba_p', 'specialisation_Mkt&Fin_mba_p', 'hsc_p_mba_p', 'workex_Yes_hsc_p', 'gender_M_degree_p']
# same    ['etest_p_mba_p', 'gender_M_mba_p', 'gender_M_etest_p', 'etest_p_etest_p', 'mba_p', 'mba_p_mba_p', 'specialisation_Mkt&Fin_mba_p', 'hsc_p_mba_p', 'workex_Yes_hsc_p', 'gender_M_degree_p']


#%% choose features
# lasso_1 = Lasso(max_iter=100_000_000, random_state=42,
#                 alpha=0.1, selection='cyclic', tol=1e-3)
# select_feat = SelectFromModel(lasso_1, threshold=1e-5, max_features=None)

# params = {
#     # Target_encoder
#     'features__categorical_t__target__min_samples_leaf': 5,
#     'features__categorical_t__target__smoothing': 0.8,
#     # SelectFromModel Lasso
#     # 'select_feat__estimator__alpha': 0.5,
#     # 'select_feat__estimator__selection': 'random',
#     # 'select_feat__estimator__tol': 1e-3,
#     # # SelectFromModel
#     # 'select_feat__max_features': 100,
#     # 'select_feat__threshold': 'mean',
#     }

# preprocess.set_params(**params)

# X = preprocess.fit_transform(df, y)
# sel = select_feat.fit_transform(X, y)
# dd = pd.DataFrame({'name': X.columns,
#                    'good': select_feat.get_support()})

# print(f'NUmber of choosen features: {dd.good.sum()}')
# short_cols = X.columns[select_feat.get_support()]

# %% Crossval
lr = LinearRegression()
lasso_1 = Lasso(max_iter=100_000_000, random_state=42)
lasso_2 = Lasso(max_iter=100_000_000, random_state=42)
select_feat = SelectFromModel(lasso_1)
selectkbest = SelectKBest(score_func=f_regression)

pipe = Pipeline(steps=[
    ('preprocess', preprocess),
    ('select_feat', ColumnExtractor(short_cols)),
    # ('select_feat', select_feat),
    ('classifier', lr)
    ])
# pipe.get_params().keys()

params = {
    # Target_encoder
    'preprocess__features__categorical_t__target__min_samples_leaf': [1],
    'preprocess__features__categorical_t__target__smoothing': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ],
    # SelectFromModel Lasso
    # 'select_feat__estimator__alpha': [0.5],
    # 'select_feat__estimator__selection': ['random'],
    # 'select_feat__estimator__tol': [1e-3],
    # SelectFromModel
    # 'select_feat__max_features': [None],
    # 'select_feat__threshold': [1e-5],
    # SelectKBest
    # 'selectkbest__k': [120],
    # Classifier Lasso
    # 'classifier__alpha': [0.1, 0.3, 0.5],
    # 'classifier__selection': ['cyclic', 'random'],
    # 'classifier__tol': np.logspace(-5, -3, num=3),
    }

gscv = GridSearchCV(estimator=pipe,
                    cv=10,
                    n_jobs=-1,
                    scoring='neg_mean_squared_error',
                    param_grid=params,
                    )

gscv.fit(df, y)
y_pred = gscv.predict(df)
print(f'Root mean: {mean_squared_error(y, y_pred)**0.5:.2f}')
print(f'r2_score : {r2_score(y, y_pred)**0.5:.2f}')

gscv.best_params_
# pipe.get_params().keys()

# X = preprocess.fit_transform(df, y)
# sel = select_feat.fit_transform(X, y)
# dd = pd.DataFrame({'name': X.columns,
#                     'good': select_feat.get_support()})

## tests
# X = preprocess.fit_transform(df, y)
# X.columns
# X.isnull().sum().sum()
# sel = select_feat.fit_transform(X, y)
# select_feat.get_support().shape
# dd = pd.DataFrame({'name': X.columns, 'in_out': select_feat.get_support()})
# X.shape
# sel.shape

# pipe.get_params()

# print(f'Number of features: {clf.n_features_in_}')

# y_s, lmbda = boxcox(y)

# pipe.fit(df, y_s)
# y_pred_s = pipe.predict(df)
# y_pred = inv_boxcox(y_pred_s, lmbda)
# print('With boxcox:')
# print(f'Root mean: {mean_squared_error(y, y_pred)**0.5:.2f}')
# print(f'r2_score : {r2_score(y, y_pred)**0.5:.2f}')
# print(f'Number of features: {clf.n_features_in_}')

# scaler = StandardScaler()
# y_scaled = scaler.fit_transform(y.to_numpy().reshape(-1, 1))
# pipe.fit(df, y_scaled)
# y_pred_s = pipe.predict(df)
# y_pred = scaler.inverse_transform(y_pred_s)
# print('With StandardScaler:')
# print(f'Root mean: {mean_squared_error(y, y_pred)**0.5:.2f}')
# print(f'r2_score : {r2_score(y, y_pred)**0.5:.2f}')
# print(f'Number of features: {clf.n_features_in_}')

