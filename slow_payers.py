# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:18:00 2021

@author: Miguel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función de aprendizaje
def learn(X, wh, wo, a = 0.5, alpha = 0.5, E = 0.001, L = 6):
    Y = np.random.random((len(X), 1))
    while True:
        for i in range(L):
            #Forward**********************************************************************
            neth = wh @ X[i]
            yh = 1/(1+np.e**(-a*neth))
            neto = wo @ yh
            Y[i] = 1/(1+np.e**(-a*neto))
            #Backward*********************************************************************
            deltao = (D[i] - Y[i]) * Y[i] * (1 - Y[i])
            deltah = yh * (1 - yh) * (np.transpose(wo) @ deltao)
            wo += np.transpose(np.atleast_2d(alpha * deltao)) @ np.atleast_2d(yh)
            wh += np.transpose(np.atleast_2d(alpha * deltah)) @ np.atleast_2d(X[i])
        print(np.linalg.norm(deltao))
        if np.linalg.norm(deltao) <= E:
            return wh, wo
        
# Función de funcionamiento       
def funct(X, wh, wo, a = 0.5, L = 6):
    Y = np.random.random((len(X), 1))
    for i in range(len(X)):    
        #Forward**********************************************************************
        neth = wh @ X[i]
        yh = 1/(1+np.e**(-a*neth))
        neto = wo @ yh
        Y[i] = 1/(1+np.e**(-a*neto))
    return Y


#Leemos el dataframe y generamos nuestra x y nuestra d
data = pd.read_excel('tarea7.xlsx')

# Normalización***************************************************************

# Monto
data['Monto normalizado'] = (data['Monto'] - min(data['Monto'])) / (max(data['Monto']) - min(data['Monto']))
# Ingreso Mensual
data["Antigüedad laboral normalizada"] = (data['Antigüedad laboral (meses)'] - min(data['Antigüedad laboral (meses)'])) / (max(data['Antigüedad laboral (meses)']) - min(data['Antigüedad laboral (meses)']))
# Carga salarial
data['Carga salarial normalizada'] = data['Mensualidad']/data['Ingreso mensual']

# Reemplazo de la variable 'Mora' a 1 y 0*************************************

data['Mora'] = data['Mora'].replace('SI', 1)
data['Mora'] = data['Mora'].replace('NO', 0)

# Preparacion del dataset para el training************************************

# Filtramos las variables que necesitamos.
index_list = np.random.choice(data.index, 700, replace=False)
# Tomamos 700 filas al azar para ser entrenadas.
data_train = data.iloc[index_list, :]
# Tomamos las 300 restantes para hacer pruebas.
data_test = data.drop(index_list)
X = data_train.iloc[:, [8, 9, 10]].to_numpy()
D = data_train['Mora'].to_numpy()

# Aprendizaje*****************************************************************

# Inicializamos los pesos con valores aleatorios
wh = np.random.random((6, 3))
wo = np.random.random((1, 6))
# Corremos el aprendizaje
wh, wo = learn(X, wh, wo)

# Prueba *********************************************************************
X = data_test.iloc[:, [8, 9, 10]].to_numpy()

# Corremos el funcionamiento y lo añadimos a data_test.
data_test['Y'] = funct(X, wh, wo)

#Hacemos una nueva columna con un booleano de aprobación del test.
data_test.loc[(data_test['Mora'] == 1) & (data_test['Y'] >= 0.5), 'Resultado'] = True 
data_test.loc[(data_test['Mora'] == 0) & (data_test['Y'] <= 0.5), 'Resultado'] = True 
data_test.loc[(data_test['Mora'] == 1) & (data_test['Y'] < 0.5), 'Resultado'] = False 
data_test.loc[(data_test['Mora'] == 0) & (data_test['Y'] > 0.5), 'Resultado'] = True

resultado = data_test['Resultado'].value_counts(normalize=True) * 100

plt.pie(resultado, labels = ['Right', 'Wrong'], shadow = True)


