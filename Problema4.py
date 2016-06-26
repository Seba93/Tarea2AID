import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

#Se leen y cargan datos de entrenamiento
X = csr_matrix(mmread('train.x.mm'))
y = np.loadtxt('train.y.dat')

#Se construye modelo, utilizando Lasso. Se calcula con alpha que maximiza el coeficiente de determinacion
a = 10**6
model = Lasso(fit_intercept=True)
model.set_params(alpha=a)
model.fit(X, y)

#Se Leen y cargan datos de prueba
X_test = csr_matrix(mmread('test.x.mm'))
y_test = np.loadtxt('test.y.dat')

#Se calcula coeficiente de determinacion
coef = model.score(X_test, y_test)
print 'Coeficiente de determinacion:', coef
