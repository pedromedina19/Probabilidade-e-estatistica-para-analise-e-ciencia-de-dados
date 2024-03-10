# -*- coding: utf-8 -*-

# Intervalos de confiança

import numpy as np
from scipy.stats import norm
from scipy import stats
import seaborn as sns
import math

## Base de dados

dados = np.array([126. , 129.5, 133. , 133. , 136.5, 136.5, 140. , 140. , 140. ,
                  140. , 143.5, 143.5, 143.5, 143.5, 143.5, 143.5, 147. , 147. ,
                  147. , 147. , 147. , 147. , 147. , 150.5, 150.5, 150.5, 150.5,
                  150.5, 150.5, 150.5, 150.5, 154. , 154. , 154. , 154. , 154. ,
                  154. , 154. , 154. , 154. , 157.5, 157.5, 157.5, 157.5, 157.5,
                  157.5, 157.5, 157.5, 157.5, 157.5, 161. , 161. , 161. , 161. ,
                  161. , 161. , 161. , 161. , 161. , 161. , 164.5, 164.5, 164.5,
                  164.5, 164.5, 164.5, 164.5, 164.5, 164.5, 168. , 168. , 168. ,
                  168. , 168. , 168. , 168. , 168. , 171.5, 171.5, 171.5, 171.5,
                  171.5, 171.5, 171.5, 175. , 175. , 175. , 175. , 175. , 175. ,
                  178.5, 178.5, 178.5, 178.5, 182. , 182. , 185.5, 185.5, 189., 192.5])

n = len(dados)
n

media = np.mean(dados)
media

desvio_padrao = np.std(dados)
desvio_padrao

## Cálculo do intervalo de confiança - manual

alpha = 0.05 / 2
alpha

1 - alpha

z = norm.ppf(1 - alpha)
z

x_inferior = media - z * (desvio_padrao / math.sqrt(n))
x_inferior

x_superior = media + z * (desvio_padrao / math.sqrt(n))
x_superior

margem_erro = abs(media - x_superior)
margem_erro

sns.distplot(dados);

## Cálculo do intervalo de confiança - scipy


stats.sem(dados)

desvio_padrao / math.sqrt(n - 1)

intervalos = norm.interval(0.95, media, stats.sem(dados))
intervalos

margem_erro = media - intervalos[0]
margem_erro

## Diferentes níveis de confiança

intervalos = norm.interval(0.99, media, stats.sem(dados))
intervalos

margem_erro = media - intervalos[0]
margem_erro

intervalos = norm.interval(0.8, media, stats.sem(dados))
intervalos

margem_erro = media - intervalos[0]
margem_erro

## Exercício

dados_salario = np.array([82.1191, 72.8014, 79.1266, 71.3552, 59.192 , 79.1952, 56.518 ,
                          70.3752, 73.5364, 61.0407, 64.3902, 66.4076, 63.5215, 71.9936,
                          60.1489, 78.5932, 76.0459, 67.7726, 64.6149, 80.1948, 76.7998,
                          76.1831, 80.7065, 62.4953, 57.204 , 62.5408, 80.0982, 63.287 ,
                          66.5826, 79.3674])

media = dados_salario.mean()
media

desvio_padrao = np.std(dados_salario)
desvio_padrao

intervalos = norm.interval(0.95, media, stats.sem(dados_salario))
intervalos

# Temos 95% de confiança de que a média salarial das pessoas está
# no intervalo entre 67.26 e 73.01

## Distribuição T Student

dados = np.array([149. , 160., 147., 189., 175., 168., 156., 160., 152.])

n = len(dados)
n

media = dados.mean()
media

desvio_padrao = np.std(dados)
desvio_padrao

from scipy.stats import t

intervalos = t.interval(0.95, n - 1, media, stats.sem(dados, ddof = 0))
intervalos

margem_erro = media - intervalos[0]
margem_erro

## Intervalos de confiança e classificação

## Accuracy


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
dataset = pd.read_csv('credit_data.csv')
dataset.dropna(inplace=True)
dataset.head()

X = dataset.iloc[:, 1:4].values
X

y = dataset.iloc[:, 4].values
y

resultados_naive_bayes_cv = []
resultados_naive_bayes_cv_300 = []
resultados_logistica_cv = []
resultados_logistica_cv_300 = []
resultados_forest_cv = []
resultados_forest_cv_300 = []
for i in range(30):
  kfold = KFold(n_splits = 10, shuffle = True, random_state = i)

  naive_bayes = GaussianNB()
  scores = cross_val_score(naive_bayes, X, y, cv = kfold)
  resultados_naive_bayes_cv_300.append(scores)
  resultados_naive_bayes_cv.append(scores.mean())

  logistica = LogisticRegression()
  scores = cross_val_score(logistica, X, y, cv = kfold)
  resultados_logistica_cv_300.append(scores)
  resultados_logistica_cv.append(scores.mean())

  random_forest = RandomForestClassifier()
  scores = cross_val_score(random_forest, X, y, cv = kfold)
  resultados_forest_cv_300.append(scores)
  resultados_forest_cv.append(scores.mean())

len(resultados_naive_bayes_cv), len(resultados_naive_bayes_cv_300)

print(resultados_naive_bayes_cv)

print(resultados_naive_bayes_cv_300)

np.asarray(resultados_naive_bayes_cv_300).shape

resultados_naive_bayes_cv = np.array(resultados_naive_bayes_cv)
resultados_naive_bayes_cv_300 = np.array(np.asarray(resultados_naive_bayes_cv_300).reshape(-1))
resultados_logistica_cv = np.array(resultados_logistica_cv)
resultados_logistica_cv_300 = np.array(np.asarray(resultados_logistica_cv_300).reshape(-1))
resultados_forest_cv = np.array(resultados_forest_cv)
resultados_forest_cv_300 = np.array(np.asarray(resultados_forest_cv_300).reshape(-1))

resultados_naive_bayes_cv_300.shape

sns.distplot(resultados_naive_bayes_cv);

sns.distplot(resultados_naive_bayes_cv_300);

sns.distplot(resultados_logistica_cv);

sns.distplot(resultados_logistica_cv_300);

sns.distplot(resultados_forest_cv);

sns.distplot(resultados_forest_cv_300, bins=5);

resultados_naive_bayes_cv.mean(), resultados_logistica_cv.mean(), resultados_forest_cv.mean()

stats.variation(resultados_naive_bayes_cv) * 100, stats.variation(resultados_logistica_cv) * 100, stats.variation(resultados_forest_cv) * 100

### Intervalos de confiança

from scipy.stats import t
from scipy.stats import norm

## Naïve bayes

intervalos_naive_bayes_t = t.interval(0.95, len(resultados_naive_bayes_cv) - 1,
                                    resultados_naive_bayes_cv.mean(),
                                    stats.sem(resultados_naive_bayes_cv, ddof = 0))
intervalos_naive_bayes_t

abs(resultados_naive_bayes_cv.mean() - intervalos_naive_bayes_t[1])

intervalos_naive_bayes_n = norm.interval(0.95, resultados_naive_bayes_cv_300.mean(),
                                         stats.sem(resultados_naive_bayes_cv_300))
intervalos_naive_bayes_n

abs(resultados_naive_bayes_cv_300.mean() - intervalos_naive_bayes_n[1])

## Regressão logística

intervalos_logistica_t = t.interval(0.95, len(resultados_logistica_cv) - 1,
                                    resultados_logistica_cv.mean(),
                                    stats.sem(resultados_logistica_cv, ddof = 0))
intervalos_logistica_t

abs(resultados_logistica_cv.mean() - intervalos_logistica_t[1])

intervalos_logistica_n = norm.interval(0.95, resultados_logistica_cv_300.mean(),
                                       stats.sem(resultados_logistica_cv_300))
intervalos_logistica_n

abs(resultados_logistica_cv_300.mean() - intervalos_logistica_n[1])

## Random Forest

intervalos_forest_t = t.interval(0.95, len(resultados_forest_cv) - 1,
                                 resultados_forest_cv.mean(),
                                 stats.sem(resultados_forest_cv, ddof = 0))
intervalos_forest_t

abs(resultados_forest_cv.mean() - intervalos_forest_t[1])

intervalos_forest_n = norm.interval(0.95, resultados_forest_cv_300.mean(),
                                    stats.sem(resultados_forest_cv_300))
intervalos_forest_n

abs(resultados_forest_cv_300.mean() - intervalos_forest_n[1])

# Temos 95% de confiança de que a média de acertos do Random Forest está
# no intervalo entre 98,63% e 98,74% - 98,59% e 98,77%

kfold = KFold(n_splits = 10, shuffle = True)
random_forest = RandomForestClassifier()
scores = cross_val_score(random_forest, X, y, cv = kfold)
print(scores.mean())