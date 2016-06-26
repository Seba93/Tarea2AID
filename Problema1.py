#!usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn import cross_validation
import matplotlib.pyplot as plt
import scipy.stats as stats

#Se construye dataframe 
url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)

#Se elimina columna sin nombre que denota la posicion de cada registro del dataframe
df = df.drop('Unnamed: 0', axis=1)

#Se almacena por separado la variable train, la cual indica si el registro pertenece o no al conjunto de entrenamiento
istrain_str = df['train']

#Se crea un arreglo que realiza un cambio de notación para los valores que puede tomar la variable train, de
#tal manera que si posee el valor T se cambio por True, y si el valor es F, se cambia por False. 
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])

#Se crea un arreglo para indicar si cada registro pertenece o no al conjunto de prueba
istest = np.logical_not(istrain)

#Se elimina la variable train del dataframe creado
df = df.drop('train',axis=1)

#Se normalizan los datos
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']

#Se construye modelo de regresion lineal
X = df_scaled.ix[: , :-1]
N = X.shape[0]
X.insert(X.shape[1], 'intercept', np.ones(N))
y = df_scaled['lpsa']
Xtrain = X[istrain]
ytrain = y[istrain]
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
linreg = lm.LinearRegression(fit_intercept = False)
linreg.fit(Xtrain, ytrain)

#Ya implementada la regresion, se obtiene el peso asociado a cada variable y su error estandar
weights = linreg.coef_
SEM = np.asarray(Xtrain.std()) / np.sqrt(len(Xtrain))
#A partir de lo anterior, se calculan los Z-score de cada variable
Z_score = weights / SEM

#Se estima error de prediccion del modelo 
#Primero, con K = 5
yhat_test = linreg.predict(Xtest)
mse_test = np.mean(np.power(yhat_test - ytest, 2))
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
k_fold1 = cross_validation.KFold(len(Xm), 5)
mse_cv1 = 0
for k, (train, val) in enumerate(k_fold1):
	linreg = lm.LinearRegression(fit_intercept = False)
	linreg.fit(Xm[train], ym[train])
	yhat_val = linreg.predict(Xm[val])
	mse_fold = np.mean(np.power(yhat_val - ym[val], 2))
	mse_cv1 += mse_fold
mse_cv1 = mse_cv1 / 5

#Luego, con K = 10
k_fold2 = cross_validation.KFold(len(Xm), 10)
mse_cv2 = 0
for k, (train, val) in enumerate(k_fold2):
	linreg = lm.LinearRegression(fit_intercept = False)
	linreg.fit(Xm[train], ym[train])
	yhat_val = linreg.predict(Xm[val])
	mse_fold = np.mean(np.power(yhat_val - ym[val], 2))
	mse_cv2 += mse_fold
mse_cv2 = mse_cv2 / 10

#Se estima error de prediccion por cada dato de entrenamiento
yhat_train = linreg.predict(Xtrain)
ytrain_array = np.asarray(ytrain)
error = yhat_train - ytrain_array
#Se genera grafico de errores
stats.probplot(error, dist='norm', plot=plt)
plt.title('Siguen los errores de prediccion sobre el conjunto de entrenamiento una distribucion normal?')
plt.ylabel('Error dato de entrenamiento')
plt.show()
