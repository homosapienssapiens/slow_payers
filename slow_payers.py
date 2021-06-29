# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:18:00 2021

@author: Miguel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Learning function
def learn(X, wh, wo, a = 0.5, alpha = 0.5, E = 0.001, L = 6):
    Y = np.random.random((len(X), 1))
    while True:
        for i in range(L):
            #Forward**********************************************************
            neth = wh @ X[i]
            yh = 1/(1+np.e**(-a*neth))
            neto = wo @ yh
            Y[i] = 1/(1+np.e**(-a*neto))
            #Backward*********************************************************
            deltao = (D[i] - Y[i]) * Y[i] * (1 - Y[i])
            deltah = yh * (1 - yh) * (np.transpose(wo) @ deltao)
            wo += np.transpose(np.atleast_2d(alpha * deltao)) @ np.atleast_2d(yh)
            wh += np.transpose(np.atleast_2d(alpha * deltah)) @ np.atleast_2d(X[i])
        print(np.linalg.norm(deltao))
        if np.linalg.norm(deltao) <= E:
            return wh, wo
        
# Function function     
def funct(X, wh, wo, a = 0.5, L = 6):
    Y = np.random.random((len(X), 1))
    for i in range(len(X)):    
        #Forward**************************************************************
        neth = wh @ X[i]
        yh = 1/(1+np.e**(-a*neth))
        neto = wo @ yh
        Y[i] = 1/(1+np.e**(-a*neto))
    return Y

data = pd.read_excel('users.xlsx')

# Normalization***************************************************************
# I was asked to use these 3 variables for the prediction:
    # Quantity: The total quantity of the loan.
    # Labor seniority: The time the client has worked in their current job (in months)
    # Salary ratio: The percentage of the Monthly income vs the monthly loan payment.

# We need to normalize the variables so we can use them for prediction. 
# Quantity
data['Quantity norm'] = (data['Quantity'] - min(data['Quantity'])) / (max(data['Quantity']) - min(data['Quantity']))
# Labor seniority
data['Labor seniority norm'] = (data['Labor seniority'] - min(data['Labor seniority'])) / (max(data['Labor seniority']) - min(data['Labor seniority']))
# Salary ratio
data['Salary ratio norm'] = data['Monthly payment']/data['Monthly income']

# Slow pay variable substitution (Si(YES)/NO(NO) for  1/0)********************

data['Slow pay'] = data['Slow pay'].replace('SI', 1)
data['Slow pay'] = data['Slow pay'].replace('NO', 0)

# DF preparedness for training************************************************

# Filtering of the variables needed.
index_list = np.random.choice(data.index, 700, replace=False)
# Data_train is a sample of 700 randomly selected rows of the original DF.
data_train = data.iloc[index_list, :]
# Data_test is the remaining 300 rows. We will use this DF for further testing
# of the effectiveness of the training.
data_test = data.drop(index_list)
X = data_train.iloc[:, [8, 9, 10]].to_numpy()
D = data_train['Slow pay'].to_numpy()

# Deep learning***************************************************************

# Random generation of weights. Because of the randomness of the initial weights
# the final prediction results can vary. 
wh = np.random.random((6, 3))
wo = np.random.random((1, 6))
# Learning
wh, wo = learn(X, wh, wo)

# Testing*********************************************************************
X = data_test.iloc[:, [8, 9, 10]].to_numpy()

# Funtion running and adding of the predictions to data_test.
data_test['Y'] = funct(X, wh, wo)

# New column for the accuracy of the predictions. 
# If the prediction is correct 'True', if it is wrong 'False'.
data_test.loc[(data_test['Slow pay'] == 1) & (data_test['Y'] >= 0.5), 'Result'] = True 
data_test.loc[(data_test['Slow pay'] == 0) & (data_test['Y'] <= 0.5), 'Result'] = True 
data_test.loc[(data_test['Slow pay'] == 1) & (data_test['Y'] < 0.5), 'Result'] = False 
data_test.loc[(data_test['Slow pay'] == 0) & (data_test['Y'] > 0.5), 'Result'] = True

# Pie chart of the results.
Result = data_test['Result'].value_counts(normalize=True) * 100
plt.pie(Result, labels = ['Right', 'Wrong'], shadow = True)


