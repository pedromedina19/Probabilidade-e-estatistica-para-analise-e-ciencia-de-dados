# -*- coding: utf-8 -*-
# Testes de hipóteses

import numpy as np
import math
from scipy.stats import norm

## Base de dados

dados_originais = np.array([126. , 129.5, 133. , 133. , 136.5, 136.5, 140. , 140. , 140. ,
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

H0_media = np.mean(dados_originais)
H0_media

H0_desvio_padrao = np.std(dados_originais)
H0_desvio_padrao

dados_novos = dados_originais * 1.03
dados_novos

H1_media = np.mean(dados_novos)
H1_media

H1_desvio_padrao = np.std(dados_novos)
H1_desvio_padrao

H1_n = len(dados_novos)
H1_n

alpha = 0.05

## Teste de hipótese Z

### Teste manual


Z = (H1_media - H0_media) / (H1_desvio_padrao / math.sqrt(H1_n))
Z

norm.cdf(3.398058252427187), norm.ppf(0.9996606701617486)

Z = norm.cdf(Z)
Z

p = 1 - Z
p

if p < alpha:
  print('Hipótese nula rejeitada')
else:
  print('Hipótese alternativa rejeitada')

"""### Teste com o statsmodels

- https://www.statsmodels.org/devel/generated/statsmodels.stats.weightstats.ztest.html
"""

from statsmodels.stats.weightstats import ztest

_, p = ztest(dados_originais, dados_novos,
             value = H1_media - H0_media,
             alternative='larger')

p

## Teste de hipótese T

dados_originais = np.array([149. , 160., 147., 189., 175., 168., 156., 160., 152.])

dados_originais.mean(), np.std(dados_originais)

dados_novos = dados_originais * 1.02
dados_novos

dados_novos.mean(), np.std(dados_novos)

from scipy.stats import ttest_rel

_, p = ttest_rel(dados_originais, dados_novos)
p

alpha = 0.01
if p <= alpha:
  print('Hipótese nula rejeitada')
else:
  print('Hipótese alternativa rejeitada')

"""## Teste qui quadrado

-https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
"""

from scipy.stats import chi2_contingency

#tabela = np.array([[30, 20], [22, 28]])
tabela = np.array([[45, 5], [5, 45]])

tabela.shape

_, p, _, _ = chi2_contingency(tabela)
p

alpha = 0.05
if p <= alpha:
  print('Hipótese nula rejeitada')
else:
  print('Hipótese alternativa rejeitada')

"""## Seleção de atributos com testes de hipóteses - univariate SelectFdr

Testes estatísticos univariados são aqueles que envolvem uma variável dependente, por exemplo, teste t ou teste z para comparação de médias

Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html#sklearn.feature_selection.SelectFpr

False discovery rate: https://en.wikipedia.org/wiki/False_discovery_rate

Proporção esperada de erros do tipo I. Um erro do tipo I é quando a hipótese nula é rejeitada incorretamente, ou seja, é obtido um falso positivo

Erro I: https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/statistics-definitions/type-i-error-type-ii-error-decision/
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('ad.data', header=None)
dataset.head()

dataset.shape

X = dataset.iloc[:, 0:1558].values
X

y = dataset.iloc[:, 1558].values
y

np.unique(y, return_counts=True)

## Sem seleção de atributos

naive1 = GaussianNB()
naive1.fit(X, y)
previsoes1 = naive1.predict(X)
accuracy_score(y, previsoes1)

## Seleção de atributos com Qui Quadrado

selecao = SelectFdr(chi2, alpha=0.01)
X_novo = selecao.fit_transform(X, y)

X.shape, X_novo.shape

selecao.pvalues_, len(selecao.pvalues_)

np.sum(selecao.pvalues_ <= 0.01)

colunas = selecao.get_support()
colunas

indices = np.where(colunas == True)
indices

naive2 = GaussianNB()
naive2.fit(X_novo, y)
previsoes2 = naive2.predict(X_novo)
accuracy_score(y, previsoes2)

## Seleção de atributos com ANOVA

#from sklearn.feature_selection import f_classif -- atualizado 10/08/2021
from sklearn.feature_selection import SelectFdr, f_classif

selecao = SelectFdr(f_classif, alpha = 0.01)
X_novo_2 = selecao.fit_transform(X, y)

X.shape, X_novo.shape, X_novo_2.shape

selecao.pvalues_

np.sum(selecao.pvalues_ < 0.01)

naive3 = GaussianNB()
naive3.fit(X_novo_2, y)
previsoes3 = naive3.predict(X_novo_2)
accuracy_score(y, previsoes3)

## ANOVA

grupo_a = np.array([165, 152, 143, 140, 155])
grupo_b = np.array([130, 169, 164, 143, 154])
grupo_c = np.array([163, 158, 154, 149, 156])

from scipy.stats import f

f.ppf(1 - 0.05, dfn = 2, dfd = 12)

from scipy.stats import f_oneway

_, p = f_oneway(grupo_a, grupo_b, grupo_c)
p

alpha = 0.05
if p <= alpha:
  print('Hipótese nula rejeitada')
else:
  print('Hipótese alternativa rejeitada')

"""### Teste de Tukey"""

dados = {'valores': [165, 152, 143, 140, 155, 130, 169, 164, 143, 154, 163, 158, 154, 149, 156],
         'grupo': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C']}

dados = {'valores': [70, 90, 80, 50, 20, 130, 169, 164, 143, 154, 163, 158, 154, 149, 156],
         'grupo': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C']}

import pandas as pd
dados_pd = pd.DataFrame(dados)
dados_pd

from statsmodels.stats.multicomp import MultiComparison

compara_grupos = MultiComparison(dados_pd['valores'], dados_pd['grupo'])

teste = compara_grupos.tukeyhsd()
print(teste)

teste.plot_simultaneous();

teste.plot_simultaneous();

## Resultados dos algoritmos de machine learning

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
dataset = pd.read_csv('credit_data.csv')
dataset.dropna(inplace=True)
dataset.head()

X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

min(X[0]), max(X[0])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

min(X[0]), max(X[0])

resultados_naive_cv = []
resultados_logistica_cv = []
resultados_forest_cv = []
for i in range(30):
  kfold = KFold(n_splits = 10, shuffle = True, random_state = i)

  naive_bayes = GaussianNB()
  scores = cross_val_score(naive_bayes, X, y, cv = kfold)
  resultados_naive_cv.append(scores.mean())

  logistica = LogisticRegression()
  scores = cross_val_score(logistica, X, y, cv = kfold)
  resultados_logistica_cv.append(scores.mean())

  random_forest = RandomForestClassifier()
  scores = cross_val_score(random_forest, X, y, cv = kfold)
  resultados_forest_cv.append(scores.mean())

resultados_naive_cv = np.array(resultados_naive_cv)
resultados_logistica_cv = np.array(resultados_logistica_cv)
resultados_forest_cv = np.array(resultados_forest_cv)

resultados_naive_cv.mean(), resultados_logistica_cv.mean(), resultados_forest_cv.mean()

"""### Teste de hipótese de Shapiro-Wilk

- https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
"""

alpha = 0.05

from scipy.stats import shapiro
shapiro(resultados_naive_cv), shapiro(resultados_logistica_cv), shapiro(resultados_forest_cv)

import seaborn as sns
sns.distplot(resultados_naive_cv);

sns.distplot(resultados_logistica_cv);

sns.distplot(resultados_forest_cv);

"""### Teste de hipótese de D'Agostinho K^2

- https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test
"""

from scipy.stats import normaltest
normaltest(resultados_naive_cv), normaltest(resultados_logistica_cv), normaltest(resultados_forest_cv)

"""### Teste de hipótese de Anderson-Darling

- https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test
"""

from scipy.stats import anderson
anderson(resultados_naive_cv).statistic, anderson(resultados_logistica_cv).statistic, anderson(resultados_forest_cv).statistic

"""### Testes não paramétricos

- https://www.statisticshowto.com/parametric-and-non-parametric-data/#:~:text=Nonparametric%20tests%20can%20perform%20well,20%20items%20in%20each%20group).

Se possível, você deve usar testes paramétricos, pois eles tendem a ser mais precisos. Os testes paramétricos têm maior poder estatístico, o que significa que é provável que encontrem um efeito verdadeiramente significativo. Use testes não paramétricos apenas se for necessário (ou seja, você sabe que suposições como a normalidade estão sendo violadas). Os testes não paramétricos podem ter um bom desempenho com dados contínuos não normais se você tiver um tamanho de amostra suficientemente grande (geralmente 15 a 20 itens em cada grupo).
### Teste de Wilcoxon Signed-Rank

- https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
"""

alpha = 0.05

from scipy.stats import wilcoxon
_, p = wilcoxon(resultados_naive_cv, resultados_logistica_cv)
p

_, p = wilcoxon(resultados_naive_cv, resultados_forest_cv)
p

_, p = wilcoxon(resultados_logistica_cv, resultados_forest_cv)
p

"""### Teste de Friedman

- https://en.wikipedia.org/wiki/Friedman_test
- Teste de Nemenyi: https://en.wikipedia.org/wiki/Nemenyi_test
"""

from scipy.stats import friedmanchisquare

_, p = friedmanchisquare(resultados_naive_cv, resultados_logistica_cv, resultados_forest_cv)
p

## ANOVA e Tukey - algoritmos

from scipy.stats import f_oneway

_, p = f_oneway(resultados_naive_cv, resultados_logistica_cv, resultados_forest_cv)
p

alpha = 0.05
if p <= alpha:
  print('Hipótese nula rejeitada. Dados são diferentes')
else:
  print('Hipótese alternativa rejeitada')

resultados_algoritmos = {'accuracy': np.concatenate([resultados_naive_cv, resultados_logistica_cv, resultados_forest_cv]),
                         'algoritmo': ['naive', 'naive','naive','naive','naive','naive','naive','naive','naive','naive',
                                       'naive', 'naive','naive','naive','naive','naive','naive','naive','naive','naive',
                                       'naive', 'naive','naive','naive','naive','naive','naive','naive','naive','naive',
                                       'logistic','logistic','logistic','logistic','logistic','logistic','logistic','logistic','logistic','logistic',
                                       'logistic','logistic','logistic','logistic','logistic','logistic','logistic','logistic','logistic','logistic',
                                       'logistic','logistic','logistic','logistic','logistic','logistic','logistic','logistic','logistic','logistic',
                                       'forest','forest','forest','forest','forest','forest','forest','forest','forest','forest',
                                       'forest','forest','forest','forest','forest','forest','forest','forest','forest','forest',
                                       'forest','forest','forest','forest','forest','forest','forest','forest','forest','forest']}

import pandas as pd
resultados_df = pd.DataFrame(resultados_algoritmos)
resultados_df

from statsmodels.stats.multicomp import MultiComparison

compara_grupos = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo'])

teste = compara_grupos.tukeyhsd()
print(teste)

teste.plot_simultaneous();

## Geração do arquivo com os resultados para o teste de Nemenyi

resultados_algoritmos = {'naive_bayes': resultados_naive_cv,
                         'logistica': resultados_logistica_cv,
                         'random_forest': resultados_forest_cv}

resultados_df = pd.DataFrame(resultados_algoritmos)
resultados_df

resultados_df.to_excel('resultados_excel.xlsx', sheet_name='resultados')

## Dados não normais

import pandas as pd
dataset = pd.read_csv('trip_d1_d2.csv', sep = ';')
dataset.head()

import seaborn as sns
sns.distplot(dataset['D1']);

sns.distplot(dataset['D2']);

print(dataset['D2'])

alpha = 0.05
from scipy.stats import shapiro
shapiro(dataset['D1']), shapiro(dataset['D2'])

from scipy.stats import friedmanchisquare
_, p = friedmanchisquare(dataset['D1'], dataset['D2'])
p

from scipy.stats import wilcoxon
_, p = wilcoxon(dataset['D1'], dataset['D2'])
p

dataset['D1'].mean(), dataset['D2'].mean()