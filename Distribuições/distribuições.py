# -*- coding: utf-8 -*-

# Distribuições

import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats

## Variáveis contínuas

### Distribuição normal - Gaussian distribution (bell)


dados_normal = stats.norm.rvs(size = 1000, random_state = 1)

min(dados_normal), max(dados_normal)

sns.distplot(dados_normal, hist = True, kde = True);

dados_normal.mean(), np.median(dados_normal), stats.mode(dados_normal), np.var(dados_normal), np.std(dados_normal)

np.sum(((dados_normal >= 0.9810041339322116) & (dados_normal <= 0.9810041339322116 + 1)))

np.sum(((dados_normal <= 0.9810041339322116) & (dados_normal >= 0.9810041339322116 - 1)))

(148 + 353) / 1000

### Distribuição normal com dados das alturas

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

len(dados)

min(dados), max(dados)

dados.mean()

np.median(dados)

stats.mode(dados)

159.25 + 13.65, 159.25 - 13.65

np.var(dados), np.std(dados), stats.variation(dados) * 100

sns.distplot(dados);

### Enviesamento

from scipy.stats import skewnorm

dados_normal = skewnorm.rvs(a = 0, size = 1000)

sns.distplot(dados_normal);

dados_normal.mean(), np.median(dados_normal), stats.mode(dados_normal)

dados_normal_positivo = skewnorm.rvs(a = 10, size = 1000)
sns.distplot(dados_normal_positivo);

dados_normal_positivo.mean(), np.median(dados_normal_positivo), stats.mode(dados_normal_positivo)

dados_normal_negativo = skewnorm.rvs(-10, size = 1000)
sns.distplot(dados_normal_negativo);

dados_normal_negativo.mean(), np.median(dados_normal_negativo), stats.mode(dados_normal_negativo)

### Distribuição normal padronizada

dados_normal_padronizada = np.random.standard_normal(size = 1000)

min(dados_normal_padronizada), max(dados_normal_padronizada)

sns.distplot(dados_normal_padronizada);

dados_normal_padronizada.mean(), np.std(dados_normal_padronizada)

dados

media = dados.mean()
media

desvio_padrao = np.std(dados)
desvio_padrao

dados_padronizados = (dados - media) / desvio_padrao
dados_padronizados

dados_padronizados.mean(), np.std(dados_padronizados)

sns.distplot(dados_padronizados);

"""### Teorema central do limite

- Quando o tamanho da amostra aumenta, a distribuição amostral da sua média aproxima-se cada vez mais de uma distribuição normal
"""

alturas = np.random.randint(126, 192, 500)
alturas.mean()

sns.distplot(alturas)

medias = [np.mean(np.random.randint(126, 192, 500)) for _ in range(1000)]

type(medias), len(medias)

print(medias)

sns.distplot(medias);

"""### Distribuição gamma

- Distribuição geral com valores assimétricos à direita
- Análise de tempo de vida de produtos
- Inadimplência com valores agregados
- Quantidade de chuva acumulada em um reservatório
- Tempo de reação de um motorista de acordo com a idade
"""

from scipy.stats import gamma

dados_gama = gamma.rvs(a = 5, size = 1000)

sns.distplot(dados_gama);

min(dados_gama), max(dados_gama)

"""### Distribuição exponencial

- É um tipo da distribuição gama
- Tempo de vida de certos produtos e materiais
- Tempo de vida de óleos isolantes e dielétricos, entre outros
"""

from scipy.stats import expon

dados_exponencial = expon.rvs(size = 1000)

sns.distplot(dados_exponencial);

min(dados_exponencial), max(dados_exponencial)

"""### Distribuição uniforme

- Os números da distribuição possuem a mesma probabilidade
- Probabilidade de peças com defeitos em um lote com determinada quantidade de peças
- Geração de números aleatórios em linguagens de programação
"""

from scipy.stats import uniform

dados_uniforme = uniform.rvs(size = 1000)

sns.distplot(dados_uniforme);

min(dados_uniforme), max(dados_uniforme)

np.unique(dados_uniforme, return_counts=True)

dataset = pd.read_csv('credit_data.csv')
dataset.dropna(inplace=True)
dataset.shape

dataset.head()

X = dataset.iloc[:, 1:4].values
X

y = dataset.iloc[:, 4].values
y

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

resultados_naive_bayes = []
for i in range(30):
  X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2,
                                                                    stratify = y, random_state = i)
  naive_bayes = GaussianNB()
  naive_bayes.fit(X_treinamento, y_treinamento)
  resultados_naive_bayes.append(accuracy_score(y_teste, naive_bayes.predict(X_teste)))

print(resultados_naive_bayes)

sns.distplot(resultados_naive_bayes, bins = 10);

"""## Variáveis discretas

### Distribuição de Bernoulli

- Tipo específico da distribuição binomial
- Possui somente duas respostas: sucesso ou fracasso
"""

from scipy.stats import bernoulli

dados_bernoulli = bernoulli.rvs(size = 1000, p = 0.3)

dados_bernoulli

np.unique(dados_bernoulli, return_counts=True)

sns.distplot(dados_bernoulli, kde=False);

"""### Distribuição binomial

- Somente duas possibilidades (sucesso ou falha)
- Probabilidade de sucesso ou falha
"""

from scipy.stats import binom

dados_binomial = binom.rvs(size = 1000, n = 10, p = 0.8)

np.unique(dados_binomial, return_counts=True)

print(dados_binomial)

sns.distplot(dados_binomial, kde=False);

"""### Distribuição de Poisson

- Número de vezes que um evento aconteceu em um intervalo de tempo
- Exemplo: número de usuários que visitaram um website em um intervalo
- Quantidade de vezes que um evento aconteceu nesse intervalo (número de ligações em um call center)
"""

from scipy.stats import poisson

dados_poisson = poisson.rvs(size = 1000, mu = 1)

min(dados_poisson), max(dados_poisson)

np.unique(dados_poisson, return_counts=True)

sns.distplot(dados_poisson, kde=False);

## Exercício

dataset = pd.read_csv('census.csv')
dataset.head()

dataset.dtypes

sns.distplot(dataset['age']);

sns.distplot(dataset['final-weight']);

sns.distplot(dataset['education-num'], kde=False);

sns.distplot(dataset['capital-gain'], kde=False);

sns.distplot(dataset['capital-loos']);

sns.distplot(dataset['hour-per-week'], kde=False);

sns.countplot(dataset['marital-status']);

sns.countplot(dataset['sex']);

sns.countplot(dataset['income']);

dataset = pd.read_csv('credit_data.csv')
dataset.head()

sns.distplot(dataset['income']);

sns.distplot(dataset['age']);

sns.distplot(dataset['loan']);

sns.distplot(dataset['c#default'], kde = False);

## Naïve Bayes e distribuições

## Bernoulli


from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('census.csv')
dataset.head()

dataset['sex'].unique()

X = dataset['sex'].values
X

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X = label_encoder.fit_transform(X)

X, np.unique(X)

sns.distplot(X, kde=False);

X.shape

X = X.reshape(-1, 1)
X.shape

y = dataset['income'].values
y

bernoulli_naive_bayes = BernoulliNB()
bernoulli_naive_bayes.fit(X, y)

previsoes = bernoulli_naive_bayes.predict(X)

previsoes, y

accuracy_score(y, previsoes)

## Multinomial

from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv('census.csv')
dataset.head()

from sklearn.preprocessing import LabelEncoder
label_encoder0 = LabelEncoder()
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()
label_encoder4 = LabelEncoder()
label_encoder5 = LabelEncoder()
label_encoder6 = LabelEncoder()

dataset['workclass'] = label_encoder0.fit_transform(dataset['workclass'])
dataset['education'] = label_encoder1.fit_transform(dataset['education'])
dataset['marital-status'] = label_encoder2.fit_transform(dataset['marital-status'])
dataset['occupation'] = label_encoder3.fit_transform(dataset['occupation'])
dataset['relationship'] = label_encoder4.fit_transform(dataset['relationship'])
dataset['race'] = label_encoder5.fit_transform(dataset['race'])
dataset['native-country'] = label_encoder6.fit_transform(dataset['native-country'])

dataset.head()

X = dataset.iloc[:, [1,3,5,6,7,8,13]].values
X

y = dataset['income'].values
y

multinomial_naive_bayes = MultinomialNB()
multinomial_naive_bayes.fit(X, y)

previsoes = multinomial_naive_bayes.predict(X)

previsoes, y

accuracy_score(y, previsoes)

## Padronização (z-score) e k-NN

import pandas as pd
import numpy as np
dataset = pd.read_csv('credit_data.csv')
dataset.dropna(inplace=True)
dataset.head()

### Sem padronização

X = dataset.iloc[:, 1:4].values
X

y = dataset['c#default'].values
y

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2,
                                                                  stratify = y)

np.mean(X_treinamento[0]), np.median(X_treinamento[0]), np.std(X_treinamento[0])

np.mean(X_teste[0]), np.median(X_teste[0]), np.std(X_teste[0])

knn = KNeighborsClassifier()
knn.fit(X_treinamento, y_treinamento)

previsoes = knn.predict(X_teste)

accuracy_score(y_teste, previsoes)

## Com padronização

from sklearn.preprocessing import StandardScaler

z_score_treinamento = StandardScaler()
z_score_teste = StandardScaler()

X_treinamento_p = z_score_treinamento.fit_transform(X_treinamento)
X_teste_p = z_score_teste.fit_transform(X_teste)

X_treinamento_p, X_teste_p

min(X_treinamento_p[0]), max(X_treinamento_p[0])

np.mean(X_treinamento_p), np.median(X_treinamento_p), np.std(X_treinamento_p)

np.mean(X_teste_p), np.median(X_teste_p), np.std(X_teste_p)

knn = KNeighborsClassifier()
knn.fit(X_treinamento_p, y_treinamento)
previsoes = knn.predict(X_teste_p)
accuracy_score(y_teste, previsoes)

## Dados enviesados e machine learning


import pandas as pd
dataset = pd.read_csv('house_prices.csv')

dataset.head()

sns.distplot(dataset['price']);

sns.distplot(dataset['sqft_living']);

## Sem tratamento de dados

from sklearn.linear_model import LinearRegression

X = dataset['sqft_living'].values
X

X.shape

X = X.reshape(-1, 1)
X.shape

y = dataset['price'].values
y

regressor = LinearRegression()
regressor.fit(X, y)

previsoes = regressor.predict(X)
previsoes

y

from sklearn.metrics import mean_absolute_error, r2_score
mean_absolute_error(y, previsoes)

r2_score(y, previsoes)

## Com tratamento de dados

X_novo = np.log(X)
X_novo

sns.distplot(X_novo);

y_novo = np.log(y)
y_novo

sns.distplot(y_novo);

regressor = LinearRegression()
regressor.fit(X_novo, y_novo)
previsoes = regressor.predict(X_novo)
mean_absolute_error(y_novo, previsoes)

r2_score(y_novo, previsoes)

## Inicialização de pesos em redes neurais

import tensorflow as tf
tf.__version__

### Inicializadores



from tensorflow.keras import initializers

## Random normal

normal = initializers.RandomNormal()
dados_normal = normal(shape=[1000])

np.mean(dados_normal), np.std(dados_normal)

sns.distplot(dados_normal);

## Random uniform

uniforme = initializers.RandomUniform()
dados_uniforme = uniforme(shape=[1000])

sns.distplot(dados_uniforme);

"""## Glorot normal (Xavier initialization)

- Centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor
"""

import math
math.sqrt(2 / (10 + 100))

math.sqrt(2 / (100 + 100))

math.sqrt(2 / (100 + 1))

# 10 -> 100 -> 100 -> 1

normal_glorot = initializers.GlorotNormal()
dados_normal_glorot = normal_glorot(shape=[1000])

sns.distplot(dados_normal_glorot);

## Glorot uniform

uniforme_glorot = initializers.GlorotUniform()
dados_uniforme_glorot = uniforme_glorot(shape=[1000])

sns.distplot(dados_uniforme_glorot);

"""## Testes de normalidade

- Estatística paramétrica: os dados estão em alguma distribuição, geralmente a distribuição normal
- Estatística não paramétrica: os dados estão em outra distribuição (ou desconhecida)
- Se os dados são "normais", usamos estatística paramétrica. Caso contrário, usamos estatística não paramétrica
"""

from scipy.stats import skewnorm
dados_normais = stats.norm.rvs(size = 1000)
dados_nao_normais = skewnorm.rvs(a = -10, size = 1000)

## Histograma

sns.distplot(dados_normais);

sns.distplot(dados_nao_normais);

## Quantile-quantile plot

from statsmodels.graphics.gofplots import qqplot

qqplot(dados_normais, line = 's');

qqplot(dados_nao_normais, line = 's');

"""### Teste de Shapiro-Wilk

- p-value é usado para interpretar o teste estatístico
- p <= alpha: rejeita a hipótese, não é normal
- p > alpha: não rejeita a hipótese, é normal
"""

from scipy.stats import shapiro

_, p = shapiro(dados_normais)
p

alpha = 0.05
if p > alpha:
  print('Distribuição normal')
else:
  print('Distribuição não normal')

_, p = shapiro(dados_nao_normais)
p

alpha = 0.05
if p > alpha:
  print('Distribuição normal')
else:
  print('Distribuição não normal')