#!usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
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

# Seleccion de Atributos
Xm_test = Xtest.as_matrix()
ym_test = ytest.as_matrix()

#(a)
def fss(x, y, names_x, k = 10000):
  mse_train = []
  p = x.shape[1]-1
  k = min(p, k)
  names_x = np.array(names_x)
  remaining = range(0, p)
  selected = [p] # Se parte con el intercepto
  current_score = 0.0
  best_new_score = 0.0
  while [remaining] and len(selected)<=k :
    score_candidates = []
    for candidate in remaining:
      model = lm.LinearRegression(fit_intercept=False)
      indexes = selected + [candidate]
      x_train = x[:,indexes]
      predictions_train = model.fit(x_train, y).predict(x_train)
      residuals_train = predictions_train - y
      mse_candidate = np.mean(np.power(residuals_train, 2))
      score_candidates.append((mse_candidate, candidate))
    score_candidates.sort()
    score_candidates[:] = score_candidates[::-1]
    best_new_score, best_candidate = score_candidates.pop()
    remaining.remove(best_candidate)
    selected.append(best_candidate)
    print "selected = %s ..."%(names_x[best_candidate])
    print "total variables = %d, mse = %f"%(len(indexes),best_new_score)
    mse_train.append(best_new_score)
  return selected, mse_train

names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]
print "FSS, error de entrenamiento:"
seleccionados, mse_train = fss(Xm,ym,names_regressors)

def fss_test(x,x_t, y, y_t, selected, names_x):
  score = []
  indexes = []
  indexes.append(selected[0])
  for i in range(1, len(selected)):
    model = lm.LinearRegression(fit_intercept=False)
    indexes.append(selected[i])
    x_train = x[:,indexes]
    x_test = x_t[:,indexes]
    prediction_test = model.fit(x_train, y).predict(x_test)
    residuals_test = prediction_test - y_t
    mse_test = np.mean(np.power(residuals_test,2))
    score.append(mse_test)
    #print "selected = %s ..."%(names_x[selected[i]])
    #print "total variables = %d, mse = %f"%(i+1,score[i-1])
  return score

mse_test = fss_test(Xm,Xm_test,ym,ym_test,seleccionados,names_regressors)

#Grafico de MSE train y test para FSS
n_predictores = range(1, Xm.shape[1]) #El intercepto se incluye desde la primera iteración
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(n_predictores, mse_train, 'bo-', label='FSS train')
ax.plot(n_predictores, mse_test, 'ro-', label='FSS test')

ax.legend()

plt.xlabel('N variables')
plt.ylabel('MSE')
plt.title('Step-wise Selection (FSS)')

plt.axis([0, 9, 0, 1])
#plt.show()

#(b)
def bss(x, y, names_x):
  mse_train = []
  orden_drop = []
  names_x = np.array(names_x)
  remaining = range(0, x.shape[1])
  selected = range(0,x.shape[1])

  #Se calcula el mse con todos los datos
  model = lm.LinearRegression(fit_intercept=False)
  predictions_train = model.fit(x, y).predict(x)
  residuals_train = predictions_train - y
  mse_train.append(np.mean(np.power(residuals_train, 2)))

  while (len(selected) != 1):
    score_candidates = []
    for candidate in remaining:
      model = lm.LinearRegression(fit_intercept=False)
      selected.remove(candidate)
      indexes = selected
      x_train = x[:,indexes]
      predictions_train = model.fit(x_train, y).predict(x_train)
      residuals_train = predictions_train - y
      mse_candidate = np.mean(np.power(residuals_train, 2))
      score_candidates.append((mse_candidate, candidate))
      selected.append(candidate)
    score_candidates.sort()
    score_candidates[:] = score_candidates[::-1]
    worst_new_score, worst_candidate = score_candidates.pop()
    remaining.remove(worst_candidate)
    selected.remove(worst_candidate)
    print "selected = %s ..."%(names_x[worst_candidate])
    print "total variables = %d, mse = %f"%(len(indexes),worst_new_score)
    mse_train.append(worst_new_score)
    orden_drop.append(worst_candidate)
  orden_drop.append(x.shape[1]-1)

  return orden_drop, mse_train

names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45", "intercept"]
print "BSS, error de entrenamiento:"
seleccionados, mse_train  = bss(Xm,ym,names_regressors)

def bss_test(x,x_t, y, y_t, selected, names_x):
  score = []
  indexes = selected[:]
  for i in selected:
    model = lm.LinearRegression(fit_intercept=False)
    x_train = x[:,indexes]
    x_test = x_t[:,indexes]
    prediction_test = model.fit(x_train, y).predict(x_test)
    residuals_test = prediction_test - y_t
    mse_test = np.mean(np.power(residuals_test,2))
    score.append(mse_test)
    #print "total variables = %d, mse = %f"%(len(indexes), mse_test)
    #print "selected = %s ..."%(names_x[i])
    indexes.remove(i)

  return score

mse_test = bss_test(Xm,Xm_test,ym,ym_test,seleccionados,names_regressors)

#Grafico de MSE train y test para FSS
n_predictores = range(0, Xm.shape[1])
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(n_predictores, mse_train, 'bo-', label='BSS train')
ax.plot(n_predictores, mse_test, 'ro-', label='BSS test')

ax.legend()

plt.xlabel('N variables')
plt.ylabel('MSE')
plt.title('Backward Step-wise Selection (BSS)')
plt.legend(loc=2)
plt.axis([0, 9, 0, 1.5])
plt.show()

# Regularización
# (a)
X = X.drop('intercept', axis=1)
Xtrain = X[istrain]
ytrain = y[istrain]
names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]
alphas_ = np.logspace(4,-1,base=10)
coefs = []
model = Ridge(fit_intercept=True,solver='svd')
for a in alphas_:
  model.set_params(alpha=a)
  model.fit(Xtrain, ytrain)
  coefs.append(model.coef_)
ax = plt.gca()
for y_arr, label in zip(np.squeeze(coefs).T, names_regressors):
  #print alphas_.shape
  #print y_arr.shape
  plt.plot(alphas_, y_arr, label=label)
plt.legend()
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) # reverse axis
plt.xlabel('lambda')
plt.ylabel('weights')
plt.title('Regularization Path RIDGE')
plt.axis('tight')
plt.legend(loc=2)
plt.show()

# (b)
alphas_ = np.logspace(1,-2,base=10)
coefs = []
model = Lasso(fit_intercept=True)
for a in alphas_:
  model.set_params(alpha=a)
  model.fit(Xtrain, ytrain)
  coefs.append(model.coef_)
ax = plt.gca()
for y_arr, label in zip(np.squeeze(coefs).T, names_regressors):
  #print alphas_.shape
  #print y_arr.shape
  plt.plot(alphas_, y_arr, label=label)
plt.legend()
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) # reverse axis
plt.xlabel('lambda')
plt.ylabel('weights')
plt.title('Regularization Path LASSO')
plt.axis('tight')
plt.legend(loc=2)
plt.show()

# (c)
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
alphas_ = np.logspace(2,-2,base=10)
coefs = []
model = Ridge(fit_intercept=True, solver='svd')
mse_test = []
mse_train = []
for a in alphas_:
  model.set_params(alpha=a)
  model.fit(Xtrain, ytrain)
  yhat_train = model.predict(Xtrain)
  yhat_test = model.predict(Xtest)
  mse_train.append(np.mean(np.power(yhat_train - ytrain, 2)))
  mse_test.append(np.mean(np.power(yhat_test - ytest, 2)))
ax = plt.gca()
ax.plot(alphas_,mse_train,label='train error ridge')
ax.plot(alphas_,mse_test,label='test error ridge')
ax.annotate('mse = 0.485847', xy=( 12.648552, 0.485847), xycoords='data', xytext=(12.648552, 30), 
  textcoords='offset points',arrowprops=dict(arrowstyle="->"))
plt.legend(loc=2)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.title('RIDGE Regression')
plt.show()

print "RIDGE"
print "Menor mse-test = %f"% (min(mse_test))
print "Alpha asociado = %f"%(alphas_[mse_test.index(min(mse_test))])

# (d)
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
alphas_ = np.logspace(0.5,-2,base=10)
coefs = []
model = Lasso(fit_intercept=True)
mse_test = []
mse_train = []
for a in alphas_:
  model.set_params(alpha=a)
  model.fit(Xtrain, ytrain)
  yhat_train = model.predict(Xtrain)
  yhat_test = model.predict(Xtest)
  mse_train.append(np.mean(np.power(yhat_train - ytrain, 2)))
  mse_test.append(np.mean(np.power(yhat_test - ytest, 2)))
print "LASSO"
print "Menor mse-test = %f"% (min(mse_test))
print "Alpha asociado = %f"%(alphas_[mse_test.index(min(mse_test))])
ax = plt.gca()
ax.plot(alphas_,mse_train,label='train error ridge')
ax.plot(alphas_,mse_test,label='test error ridge')
ax.annotate('mse = 0.452565', xy=( 0.104811, 0.452565), xycoords='data', xytext=(0.1, 30), 
  textcoords='offset points',arrowprops=dict(arrowstyle="->"))
plt.legend(loc=1)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.title('LASSO')
plt.show()

# (e)
def MSE(y,yhat): return np.mean(np.power(y-yhat,2))
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
alphas_ = np.logspace(0.5,-2,base=10)
k_fold = cross_validation.KFold(len(Xm),10)
best_cv_mse = float("inf")
model = Lasso(fit_intercept=True)
for a in alphas_:
  model.set_params(alpha=a)
  mse_list_k10 = [MSE(model.fit(Xm[train], ym[train]).predict(Xm[val]), ym[val]) for train, val in k_fold]
  if np.mean(mse_list_k10) < best_cv_mse:
    best_cv_mse = np.mean(mse_list_k10)
    best_alpha = a
print "BEST PARAMETER=%f, MSE-LASSO(CV)=%f"%(best_alpha,best_cv_mse)

def MSE_R(y,yhat): return np.mean(np.power(y-yhat,2))
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
alphas_ = np.logspace(2,-2,base=10)
k_fold = cross_validation.KFold(len(Xm),10)
best_cv_mse = float("inf")
model = Ridge(fit_intercept=True, solver='svd')
for a in alphas_:
  model.set_params(alpha=a)
  mse_list_k10 = [MSE_R(model.fit(Xm[train], ym[train]).predict(Xm[val]), ym[val]) for train, val in k_fold]
  if np.mean(mse_list_k10) < best_cv_mse:
    best_cv_mse = np.mean(mse_list_k10)
    best_alpha = a
print "BEST PARAMETER=%f, MSE-RIDGE(CV)=%f"%(best_alpha,best_cv_mse)