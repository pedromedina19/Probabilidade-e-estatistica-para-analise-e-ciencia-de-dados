# -*- coding: utf-8 -*-

# Correlação e regressão

import numpy as np
import pandas as pd
import seaborn as sns
import math

## Base de dados

tamanho = np.array([30, 39, 49, 60])
preco = np.array([57000, 69000, 77000, 90000])

dataset = pd.DataFrame({'tamanho': tamanho, 'preco': preco})
dataset

media_tamanho = dataset['tamanho'].mean()
media_preco = dataset['preco'].mean()
media_tamanho, media_preco

dp_tamanho = dataset['tamanho'].std()
dp_preco = dataset['preco'].std()
dp_tamanho, dp_preco

## Correlação - cálculo manual

dataset['dif'] = (dataset['tamanho'] - media_tamanho) * (dataset['preco'] - media_preco)
dataset

soma_dif = dataset['dif'].sum()
soma_dif

covariancia = soma_dif / (len(dataset) - 1)
covariancia

coeficiente_correlacao = covariancia / (dp_tamanho * dp_preco)
coeficiente_correlacao

sns.scatterplot(tamanho, preco);

coeficiente_determinacao = math.pow(coeficiente_correlacao, 2)
coeficiente_determinacao

## Correlação - cálculo com numpy a pandas

np.cov(tamanho, preco)

dataset.cov()

np.corrcoef(tamanho, preco)

dataset.corr()

## Exercício - correlação base de dados preço das casas

dataset = pd.read_csv('house_prices.csv')
dataset.head()

dataset.drop(labels = ['id', 'date', 'sqft_living', 'sqft_lot'], axis = 1, inplace=True)
dataset.head()

dataset.corr()

sns.scatterplot(dataset['sqft_living15'], dataset['price']);

sns.scatterplot(dataset['grade'], dataset['price']);

sns.scatterplot(dataset['long'], dataset['price']);

sns.heatmap(dataset.corr(), annot=True);

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(dataset.corr(), annot=True)

## Correlação com a biblioteca Yellowbrick

from yellowbrick.target import FeatureCorrelation

dataset.columns[1:]

grafico = FeatureCorrelation(labels = dataset.columns[1:])
grafico.fit(dataset.iloc[:, 1:16].values, dataset.iloc[:, 0].values)
grafico.show();

"""## Regressão

- Correlação: relacionamento entre variáveis, uma variável afeta a outra. Duas vias: correlação de price x sqft_living = correlação sqft_living x price
- Regressão é uma via: sqft_living para prever o preço é diferente de utilizar o preço para prever sqft_living
- Espera-se que a correlação seja de moderada a forte para obter um bom modelo (positiva ou negativa)
- Coeficiente de determinação (R2): > 0.7 é um bom valor. Entre 0 e 0.3 é ruim. Entre esses valores é interessante fazer testes
- Regressão linear: existir lineariedade
"""

dataset = pd.read_csv('house_prices.csv')
dataset.head()

dataset.drop(labels = ['id', 'date'], axis = 1, inplace=True)
dataset.head()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(dataset.corr(), annot=True)

math.pow(0.7, 2)

## Regressão linear simples


X = dataset['sqft_living'].values
X.shape

X = X.reshape(-1, 1)
X.shape

y = dataset['price'].values
y

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_treinamento.shape, X_teste.shape

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

# b0
regressor.intercept_

# b1
regressor.coef_

regressor.intercept_ + regressor.coef_ * 900

regressor.predict(np.array([[900]]))

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')

regressor.score(X_treinamento, y_treinamento)

regressor.score(X_teste, y_teste)

## Métricas de erros

previsoes = regressor.predict(X_teste)

previsoes, y_teste

from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(y_teste, previsoes)

mean_squared_error(y_teste, previsoes)

math.sqrt(mean_squared_error(y_teste, previsoes))

## Regressão linear múltipla

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(dataset.corr(), annot=True)

dataset.head()

X = dataset.iloc[:, [2, 3, 9, 10]].values
X

y = dataset.iloc[:, 0].values
y

import matplotlib.pyplot as plt
f, ax = plt.subplots(2, 2)
ax[0, 0].hist(X[0])
ax[0, 1].hist(X[1])
ax[1, 0].hist(X[2])
ax[1, 1].hist(X[3]);

plt.hist(y);

y = np.log(y)

plt.hist(y);

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_treinamento.shape, X_teste.shape

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

regressor.score(X_treinamento, y_treinamento)

regressor.score(X_teste, y_teste)

previsoes = regressor.predict(X_teste)
mean_absolute_error(y_teste, previsoes)

## Exercício

dataset.head()

dataset.drop(labels = ['sqft_living15', 'sqft_lot15'], axis = 1, inplace=True)
dataset.head()

X = dataset.iloc[:, 1:17].values
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

regressor.score(X_treinamento, y_treinamento)

regressor.score(X_teste, y_teste)

### Seleção de atributos

from sklearn.feature_selection import SelectFdr, f_regression
selecao = SelectFdr(f_regression, alpha = 0.0)
X_novo = selecao.fit_transform(X, y)
X.shape, X_novo.shape

selecao.pvalues_

colunas = selecao.get_support()
colunas

dataset.columns[1:17]

dataset.columns[1:17][colunas == True]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X_novo, y, test_size = 0.2, random_state = 1)
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

regressor.score(X_treinamento, y_treinamento)

regressor.score(X_teste, y_teste)