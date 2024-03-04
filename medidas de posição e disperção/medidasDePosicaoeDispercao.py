# -*- coding: utf-8 -*-

# Medidas de posição e dispersão


import numpy as np
import statistics
from scipy import stats
import math

dados = np.array([150, 151, 152, 152, 153, 154, 155, 155, 155, 155, 156, 156, 156,
                  157, 158, 158, 160, 160, 160, 160, 160, 161, 161, 161, 161, 162,
                  163, 163, 164, 164, 164, 165, 166, 167, 168, 168, 169, 170, 172,
                  173])

## Média aritmética simples

dados.sum() / len(dados)

dados.mean()

statistics.mean(dados)

## Moda

statistics.mode(dados)

stats.mode(dados)

## Mediana

dados_impar = [150, 151, 152, 152, 153, 154, 155, 155, 155]

## Cálculo manual (ímpar)

posicao = len(dados_impar) / 2
posicao

posicao = math.ceil(posicao)
posicao

dados_impar[posicao - 1]

## Cálculo manual (par)

posicao = len(dados) // 2
posicao

dados[posicao - 1], dados[posicao]

mediana = (dados[posicao - 1] + dados[posicao]) / 2
mediana

## Bibliotecas

np.median(dados_impar)

np.median(dados)

statistics.median(dados_impar)

statistics.median(dados)

## Média aritmética ponderada

notas = np.array([9, 8, 7, 3])
pesos = np.array([1, 2, 3, 4])

(9 * 1 + 8 * 2 + 7 * 3 + 3 * 4) / (1 + 2 + 3 + 4)

media_ponderada = (notas * pesos).sum() / pesos.sum()
media_ponderada

np.average(notas, weights=pesos)

## Média aritmética, moda e mediana com distribuição de frequência (dados agrupados)

dados = {'inferior': [150, 154, 158, 162, 166, 170],
         'superior': [154, 158, 162, 166, 170, 174],
         'fi': [5, 9, 11, 7, 5, 3]}

import pandas as pd
dataset = pd.DataFrame(dados)
dataset

dataset['xi'] = (dataset['superior'] + dataset['inferior']) / 2
dataset

dataset['fi.xi'] = dataset['fi'] * dataset['xi']
dataset

dataset['Fi'] = 0
dataset

frequencia_acumulada = []
somatorio = 0
for linha in dataset.iterrows():
  #print(linha[1])
  #print(linha[1][2])
  somatorio += linha[1][2]
  frequencia_acumulada.append(somatorio)

frequencia_acumulada

dataset['Fi'] = frequencia_acumulada
dataset

## Média

dataset['fi'].sum(), dataset['fi.xi'].sum()

dataset['fi.xi'].sum() / dataset['fi'].sum()

## Moda

dataset['fi'].max()

dataset[dataset['fi'] == dataset['fi'].max()]

dataset[dataset['fi'] == dataset['fi'].max()]['xi'].values[0]

## Mediana

dataset

fi_2 = dataset['fi'].sum() / 2
fi_2

limite_inferior, frequencia_classe, id_frequencia_anterior = 0, 0, 0
for linha in dataset.iterrows():
  #print(linha)
  limite_inferior = linha[1][0]
  frequencia_classe = linha[1][2]
  id_frequencia_anterior = linha[0]
  if linha[1][5] >= fi_2:
    id_frequencia_anterior -= 1
    break

limite_inferior, frequencia_classe, id_frequencia_anterior

Fi_anterior = dataset.iloc[[id_frequencia_anterior]]['Fi'].values[0]
Fi_anterior

mediana = limite_inferior + ((fi_2 - Fi_anterior) * 4) / frequencia_classe
mediana

## Função completa

def get_estatisticas(dataframe):
  media = dataset['fi.xi'].sum() / dataset['fi'].sum()
  moda = dataset[dataset['fi'] == dataset['fi'].max()]['xi'].values[0]

  fi_2 = dataset['fi'].sum() / 2
  limite_inferior, frequencia_classe, id_frequencia_anterior = 0, 0, 0
  for i, linha in enumerate(dataset.iterrows()):
    limite_inferior = linha[1][0]
    frequencia_classe = linha[1][2]
    id_frequencia_anterior = linha[0]
    if linha[1][5] >= fi_2:
      id_frequencia_anterior -= 1
      break
  Fi_anterior = dataset.iloc[[id_frequencia_anterior]]['Fi'].values[0]
  mediana = limite_inferior + ((fi_2 - Fi_anterior) * 4) / frequencia_classe

  return media, moda, mediana

get_estatisticas(dataset)

## Média geométrica, harmônica e quadrática

## Média geométrica


from scipy.stats.mstats import gmean

gmean(dados)

## Média harmônica

from scipy.stats.mstats import hmean

hmean(dados)

## Média quadrática

def quadratic_mean(dados):
  return math.sqrt(sum(n * n for n in dados) / len(dados))

quadratic_mean(dados)

## Quartis

dados_impar = [150, 151, 152, 152, 153, 154, 155, 155, 155]

## Cálculo manual

np.median(dados_impar)

posicao_mediana = math.floor(len(dados_impar) / 2)
posicao_mediana

esquerda = dados_impar[0:posicao_mediana]
esquerda

np.median(esquerda)

direita = dados_impar[posicao_mediana + 1:]
direita

np.median(direita)

## Bibliotecas

## numpy


np.quantile(dados_impar, 0.5)

np.quantile(dados_impar, 0.75)

np.quantile(dados_impar, 0.25)

esquerda2 = dados_impar[0:posicao_mediana + 1]
esquerda2

np.median(esquerda2)

np.quantile(dados, 0.25), np.quantile(dados, 0.50), np.quantile(dados, 0.75)

## scipy

stats.scoreatpercentile(dados, 25), stats.scoreatpercentile(dados, 50), stats.scoreatpercentile(dados, 75)

## pandas

import pandas as pd
dataset = pd.DataFrame(dados)
dataset.head()

dataset.quantile([0.25, 0.5, 0.75])

dataset.describe()

## Quartis com distribuição de frequência (dados agrupados)

dataset

def get_quartil(dataframe, q1 = True):
  if q1 == True:
    fi_4 = dataset['fi'].sum() / 4
  else:
    fi_4 = (3 * dataset['fi'].sum()) / 4

  limite_inferior, frequencia_classe, id_frequencia_anterior = 0, 0, 0
  for linha in dataset.iterrows():
    limite_inferior = linha[1][0]
    frequencia_classe = linha[1][2]
    id_frequencia_anterior = linha[0]
    if linha[1][5] >= fi_4:
      id_frequencia_anterior -= 1
      break
  Fi_anterior = dataset.iloc[[id_frequencia_anterior]]['Fi'].values[0]
  q = limite_inferior + ((fi_4 - Fi_anterior) * 4) / frequencia_classe

  return q

get_quartil(dados), get_quartil(dados, q1 = False)

## Percentis

np.median(dados)

np.quantile(dados, 0.5)

np.percentile(dados, 50)

np.percentile(dados, 5), np.percentile(dados, 10), np.percentile(dados, 90)

stats.scoreatpercentile(dados, 5), stats.scoreatpercentile(dados, 10), stats.scoreatpercentile(dados, 90)

import pandas as pd
dataset = pd.DataFrame(dados)
dataset.head()

dataset.quantile([0.05, 0.10, 0.90])

## Exercício

dataset = pd.read_csv('census.csv')

dataset.head()

dataset['age'].mean()

stats.hmean(dataset['age'])

from scipy.stats.mstats import gmean
gmean(dataset['age'])

quadratic_mean(dataset['age'])

dataset['age'].median()

statistics.mode(dataset['age'])

## Medidas de dispersão

## Amplitude total e diferença interquartil


dados

dados.max() - dados.min()

q1 = np.quantile(dados, 0.25)
q3 = np.quantile(dados, 0.75)
q1, q3

diferenca_interquartil = q3 - q1
diferenca_interquartil

inferior = q1 - (1.5 * diferenca_interquartil)
inferior

superior = q3 + (1.5 * diferenca_interquartil)
superior

## Variância, desvio padrão e coeficiente de variação

dados_impar = np.array([150, 151, 152, 152, 153, 154, 155, 155, 155])

## Cálculo manual

media = dados_impar.sum() / len(dados_impar)
media

desvio = abs(dados_impar - media)
desvio

desvio = desvio ** 2
desvio

soma_desvio = desvio.sum()
soma_desvio

v = soma_desvio / len(dados_impar)
v

dp = math.sqrt(v)
dp

cv = (dp / media) * 100
cv

def get_variancia_desvio_padrao_coeficiente(dataset):
  media = dataset.sum() / len(dataset)
  desvio = abs(dados_impar - media)
  desvio = desvio ** 2
  soma_desvio = desvio.sum()
  variancia = soma_desvio / len(dados_impar)
  dp = math.sqrt(variancia)
  return variancia, dp, (dp / media) * 100

get_variancia_desvio_padrao_coeficiente(dados_impar)

## Bibliotecas

np.var(dados_impar)

np.std(dados_impar)

np.var(dados)

np.std(dados)

statistics.variance(dados)

statistics.stdev(dados)

from scipy import ndimage
ndimage.variance(dados)

stats.tstd(dados, ddof = 0)

stats.variation(dados_impar) * 100

stats.variation(dados) * 100

## Desvio padrão com dados agrupados

dataset

dataset['xi_2'] = dataset['xi'] * dataset['xi']
dataset

dataset['fi_xi_2'] = dataset['fi'] * dataset['xi_2']
dataset

dataset.columns

colunas_ordenadas = ['inferior', 'superior', 'fi', 'xi', 'fi.xi', 'xi_2', 'fi_xi_2', 'Fi']

dataset = dataset[colunas_ordenadas]
dataset

dp = math.sqrt(dataset['fi_xi_2'].sum() / dataset['fi'].sum() - math.pow(dataset['fi.xi'].sum() / dataset['fi'].sum(), 2))
dp

## Testes com algoritmos de classificação

import pandas as pd
dataset = pd.read_csv('credit_data.csv')

dataset.dropna(inplace=True)
dataset.shape

dataset

X = dataset.iloc[:, 1:4].values
X

y = dataset.iloc[:, 4].values
y

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

resultados_naive_bayes = []
resultados_logistica = []
resultados_forest = []
for i in range(30):
  X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2,
                                                                    stratify = y, random_state = i)
  naive_bayes = GaussianNB()
  naive_bayes.fit(X_treinamento, y_treinamento)
  resultados_naive_bayes.append(accuracy_score(y_teste, naive_bayes.predict(X_teste)))

  logistica = LogisticRegression()
  logistica.fit(X_treinamento, y_treinamento)
  resultados_logistica.append(accuracy_score(y_teste, logistica.predict(X_teste)))

  random_forest = RandomForestClassifier()
  random_forest.fit(X_treinamento, y_treinamento)
  resultados_forest.append(accuracy_score(y_teste, random_forest.predict(X_teste)))

print(resultados_naive_bayes)

print(resultados_logistica)

print(resultados_forest)

type(resultados_naive_bayes)

resultados_naive_bayes = np.array(resultados_naive_bayes)
resultados_logistica = np.array(resultados_logistica)
resultados_forest = np.array(resultados_forest)

type(resultados_naive_bayes)

## Média

resultados_naive_bayes.mean(), resultados_logistica.mean(), resultados_forest.mean()

## Moda

statistics.mode(resultados_naive_bayes)

stats.mode(resultados_naive_bayes), stats.mode(resultados_logistica), stats.mode(resultados_forest)

## Mediana

np.median(resultados_naive_bayes), np.median(resultados_logistica), np.median(resultados_forest)

## Variância

np.set_printoptions(suppress=True)
np.var(resultados_naive_bayes), np.var(resultados_logistica), np.var(resultados_forest)

np.min([8.756250000000001e-05, 0.00020933333333333337, 2.9229166666666637e-05])

np.max([8.756250000000001e-05, 0.00020933333333333337, 2.9229166666666637e-05])

resultados_forest

## Desvio padrão

np.std(resultados_naive_bayes), np.std(resultados_logistica), np.std(resultados_forest)

## Coeficiente de variação

stats.variation(resultados_naive_bayes) * 100, stats.variation(resultados_logistica) * 100, stats.variation(resultados_forest) * 100

## Exercício: validação cruzada

from sklearn.model_selection import cross_val_score, KFold

resultados_naive_bayes_cv = []
resultados_logistica_cv = []
resultados_forest_cv = []
for i in range(30):
  kfold = KFold(n_splits = 10, shuffle = True, random_state = i)

  naive_bayes = GaussianNB()
  scores = cross_val_score(naive_bayes, X, y, cv = kfold)
  resultados_naive_bayes_cv.append(scores.mean())

  logistica = LogisticRegression()
  scores = cross_val_score(logistica, X, y, cv = kfold)
  resultados_logistica_cv.append(scores.mean())

  random_forest = RandomForestClassifier()
  scores = cross_val_score(random_forest, X, y, cv = kfold)
  resultados_forest_cv.append(scores.mean())

scores, 10 * 30

scores.mean()

print(resultados_naive_bayes_cv)

print(resultados_logistica_cv)

print(resultados_forest_cv)

stats.variation(resultados_naive_bayes) * 100, stats.variation(resultados_logistica) * 100, stats.variation(resultados_forest) * 100

stats.variation(resultados_naive_bayes_cv) * 100, stats.variation(resultados_logistica_cv) * 100, stats.variation(resultados_forest_cv) * 100

## Seleção de atributos utilizando variância

np.random.rand(50)

np.random.randint(0, 2)

base_selecao = {'a': np.random.rand(20),
                'b': np.array([0.5] * 20),
                'classe': np.random.randint(0, 2, size = 20)}

base_selecao

dataset = pd.DataFrame(base_selecao)
dataset.head()

dataset.describe()

math.sqrt(0.08505323963215053)

np.var(dataset['a']), np.var(dataset['b'])

X = dataset.iloc[:, 0:2].values
X

from sklearn.feature_selection import VarianceThreshold

selecao = VarianceThreshold(threshold=0.07)
X_novo = selecao.fit_transform(X)

X_novo, X_novo.shape

selecao.variances_

indices = np.where(selecao.variances_ > 0.07)
indices

## Exercício seleção de atributos utilizando variância

dataset = pd.read_csv('credit_data.csv')

dataset.dropna(inplace=True)

dataset.head()

dataset.describe()

X = dataset.iloc[:, 1:4].values
X

y = dataset.iloc[:, 4].values
y

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X

selecao = VarianceThreshold(threshold=0.027)
X_novo = selecao.fit_transform(X)

X_novo


np.var(X[:, 0]), np.var(X[:, 1]), np.var(X[:, 2])

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
naive_sem_selecao = GaussianNB()
naive_sem_selecao.fit(X, y)
previsoes = naive_sem_selecao.predict(X)
accuracy_score(previsoes, y)

naive_com_selecao = GaussianNB()
naive_com_selecao.fit(X_novo, y)
previsoes = naive_com_selecao.predict(X_novo)
accuracy_score(previsoes, y)

## Valores faltantes com média e moda

## Média


import pandas as pd
dataset = pd.read_csv('credit_data.csv')

dataset.isnull().sum()

nulos = dataset[dataset.isnull().any(axis=1)]
nulos

dataset['age'].mean(), dataset['age'].median()

dataset['age'] = dataset['age'].replace(to_replace = np.nan, value = dataset['age'].mean())

dataset[dataset.isnull().any(axis=1)]

## Moda

dataset = pd.read_csv('autos.csv', encoding='ISO-8859-1')

dataset.head()

dataset.isnull().sum()

dataset['fuelType'].unique()

stats.mode(dataset['fuelType'])

statistics.mode(dataset['fuelType'])

dataset['fuelType'] = dataset['fuelType'].replace(to_replace = np.nan, value = statistics.mode(dataset['fuelType']))

dataset['fuelType'].unique()