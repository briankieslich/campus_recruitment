#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read the docs, obey PEP 8 and PEP 20 (Zen of Python, import this)

Build on:    Spyder
Python ver: 3.7.3
Author: Brian Kieslich
"""
"""
1) Develop an estimated multiple linear regression equation with mbap as response variable
and sscp & hscp as the two predictor variables.
Interpret the regression coefficients and check whether they are significant based on
the summary output.

2) Estimate a multiple regression equation for each of the below scenarios and based
on the model’s R-square comment which model is better.
    (i) Use mbap as outcome variable and sscp & degreep as the two predictor variables.
    (ii) Use mbap as outcome variable and hscp & degreep as the two predictor variables.

3) Show the functional form of a multiple regression model. Build a regression model with
mbap as dependent variable and sscp, hscp and degree_p as three independent variables.
Divide the dataset in the ratio of 80:20 for train and test set (set seed as 1001) and
use the train set to build the model.
Show the model summary and interpret the p-values of the regression coefficients.
Remove any insignificant variables and rebuild the model.
Use this model for prediction on the test set and show the first few observations’ actual
value of the test set in comparison to the predicted value.



"""
# %%% modules:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import (LinearRegression,
                                  ElasticNet)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             mean_squared_error)
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 4)



# %% Task 1:
"""
Develop an estimated multiple linear regression equation with mbap as response variable
and sscp & hscp as the two predictor variables.
Interpret the regression coefficients and check whether they are significant based on
the summary output.
"""

data = pd.read_csv('../data/Placement_Data_Full_Class.csv',
                 usecols=['ssc_p', 'hsc_p', 'degree_p', 'mba_p'])
data.info()
sns.pairplot(data=data, kind='reg', diag_kind='kde')
sns.boxenplot(data=data)
# Very few outliers, but let's leave it as it is.
df = data.drop(columns='degree_p').copy()
y = df.pop('mba_p')

lr = LinearRegression()
lr.fit(df, y)

score_1 = lr.score(df, y)
print(score_1)
print(lr.coef_)
print(lr.intercept_)

X = sm.add_constant(df)
print(sm.OLS(y, X).fit().summary())
"""
Model:  mba_p = 0.15103*ssc_p + 0.11318*hsc_p + 44.60525
But from the R2 scores and t values we see that this is not a good model,
but the best we can do with these data :(
"""

#%% Task 2
"""
Estimate a multiple regression equation for each of the below scenarios and based
on the model’s R-square comment which model is better.
    (i) Use mbap as outcome variable and sscp & degreep as the two predictor variables.
    (ii) Use mbap as outcome variable and hscp & degreep as the two predictor variables.
"""
df = data.copy()
y = df.pop('mba_p')

lr = LinearRegression()

lr.fit(df.loc[:, ['ssc_p', 'degree_p']], y)
score_2_a = lr.score(df.loc[:, ['ssc_p', 'degree_p']], y)
print(score_2_a)
lr.fit(df.loc[:, ['hsc_p', 'degree_p']], y)
score_2_b = lr.score(df.loc[:, ['hsc_p', 'degree_p']], y)
print(score_2_b)

# 'ssc_p' and 'degree_p' is a better model with a slighty higher r2 score

#%% Task 3
"""
Show the functional form of a multiple regression model.
Build a regression model with mbap as dependent variable and sscp, hscp and degree_p as
three independent variables.
Divide the dataset in the ratio of 80:20 for train and test set (set seed as 1001) and
use the train set to build the model.
Show the model summary and interpret the p-values of the regression coefficients.
Remove any insignificant variables and rebuild the model.
Use this model for prediction on the test set and show the first few observations’ actual
value of the test set in comparison to the predicted value.
"""
df = data.copy()
y = df.pop('mba_p')

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1001)
lr = LinearRegression()
lr.fit(X_train, y_train)
score_3_a = lr.score(X_test, y_test)
print(score_3_a)
print(lr.coef_, lr.intercept_)
X = sm.add_constant(df)
print(sm.OLS(y, X).fit().summary())

# it looks like ssc_p and hsc_p doesn't give that much information, too high p-values

X_train = X_train.loc[:, 'degree_p'].to_numpy().reshape(-1, 1)
X_test = X_test.loc[:, 'degree_p'].to_numpy().reshape(-1, 1)

lr.fit(X_train, y_train)
score_3_b = lr.score(X_test, y_test)
print(score_3_b)

# Summary of scores:
print(f'{score_1:.4f} for ssc_p and hsc_p')
print(f'{score_2_a:.4f} for ssc_p and degree_p')
print(f'{score_2_b:.4f} for hsc_p and degree_p')
print(f'{score_3_a:.4f} for ssc_p and hsc_p and degree_p')
print(f'{score_3_b:.4f} for degree_p')

comp = pd.DataFrame({'y_test': y_test, 'y_pred': lr.predict(X_test)})
print(comp.head())

# Not too bad :(
"""
Try
pipe = Pipeline([
    ('select', SelectFromModel(Lasso())),
    ('lnr', LinearRegression())
    ])
Alexander: 1:30:00 -->




"""