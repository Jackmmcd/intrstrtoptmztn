#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 21:03:41 2019

@author: jackmcdonald
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd 
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df= pd.read_csv("loan.csv.zip")

X = df[['loan_amnt', 'funded_amnt', 'installment', 'annual_inc']]
y = df[['int_rate']]
toEncode= df.iloc[:, 13:101].values


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
