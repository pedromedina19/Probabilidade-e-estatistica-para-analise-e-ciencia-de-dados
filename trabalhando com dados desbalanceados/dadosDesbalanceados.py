# -*- coding: utf-8 -*-

# Classificação com dados desbalanceados

## Carregamento da base de dados

import pandas as pd
import random
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

dataset = pd.read_csv('credit_data.csv')

dataset.shape

dataset.head()

dataset.dropna(inplace=True)
dataset.shape

import seaborn as sns
sns.countplot(x=dataset['c#default']);

X = dataset.iloc[:, 1:4].values

X.shape

X

y = dataset.iloc[:, 4].values

y.shape

y

"""## Base de treinamento e teste"""

from sklearn.model_selection import train_test_split

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, stratify = y)

X_treinamento.shape, y_treinamento.shape

X_teste.shape, y_teste.shape

np.unique(y, return_counts=True)

1714 / len(dataset), 283 / len(dataset)

np.unique(y_treinamento, return_counts=True)

226 / len(y_treinamento)

np.unique(y_teste, return_counts=True)

57 / len(y_teste)

"""## Classificação com Naïve Bayes"""

from sklearn.naive_bayes import GaussianNB

modelo = GaussianNB()
modelo.fit(X_treinamento, y_treinamento)

previsoes = modelo.predict(X_teste)

previsoes

y_teste

from sklearn.metrics import accuracy_score

accuracy_score(y_teste,previsoes) #Corrigido 04/10/2021

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_teste,previsoes) #Corrigido 04/10/2021
cm

sns.heatmap(cm, annot=True);

(336 + 32) / (336 + 25 + 7 + 32)

# Percentual de acerto para pessoas que pagam o empréstimo
336 / (336 + 25)

# Percentual de acerto para pessoas que não pagam o empréstimo
32 / (32 + 7)

# Perdas: 5.000
# Clientes não pagadores: 1.000
1000 * 18 / 100

180 * 5000

"""## Subamostragem (undersampling) - Tomek links

- https://imbalanced-learn.org/stable/introduction.html
"""

from imblearn.under_sampling import TomekLinks

tl = TomekLinks(sampling_strategy='majority')
X_under, y_under = tl.fit_resample(X, y) 

X_under.shape, y_under.shape

np.unique(y, return_counts=True)

np.unique(y_under, return_counts=True)

X_treinamento_u, X_teste_u, y_treinamento_u, y_teste_u = train_test_split(X_under,
                                                                          y_under,
                                                                          test_size=0.2,
                                                                          stratify=y_under)
X_treinamento_u.shape, X_teste_u.shape

modelo_u = GaussianNB()
modelo_u.fit(X_treinamento_u, y_treinamento_u)
previsoes_u = modelo_u.predict(X_teste_u)
accuracy_score(y_teste_u,previsoes_u ) #Corrigido 04/10/2021

cm_u = confusion_matrix(y_teste_u, previsoes_u) #Corrigido 04/10/2021
cm_u

315 / (315 + 26)

31 / (31 + 8)

"""## Sobreamostragem (oversampling) - SMOTE"""

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_over, y_over = smote.fit_resample(X, y) 

X_over.shape, y_over.shape

np.unique(y, return_counts=True)

np.unique(y_over, return_counts=True)

X_treinamento_o, X_teste_o, y_treinamento_o, y_teste_o = train_test_split(X_over, y_over,
                                                                          test_size = 0.2,
                                                                          stratify=y_over)

X_treinamento_o.shape, X_teste_o.shape

modelo_o = GaussianNB()
modelo_o.fit(X_treinamento_o, y_treinamento_o)
previsoes_o = modelo_o.predict(X_teste_o)
accuracy_score(y_teste_o, previsoes_o) #Corrigido 04/10/2021

cm_o = confusion_matrix(y_teste_o, previsoes_o) #Corrigido 04/10/2021
cm_o

305 / (305 + 19)

324 / (324 + 38)

# Perdas: 5.000
# Cliente não pagadores: 1.000
1000 * 11 / 100

110 * 5000

900000 - 550000