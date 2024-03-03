# -*- coding: utf-8 -*-

# Dados absolutos e relativos

import pandas as pd

## Percentuais

dados = {'emprego': ['Adminstrador_banco_dados', 'Programador', 'Arquiteto_redes'],
         'nova_jersey': [97350, 82080, 112840],
         'florida': [77140, 71540, 62310]}

type(dados)

dados

dataset = pd.DataFrame(dados)

dataset

dataset['nova_jersey'].sum()

dataset['florida'].sum()

dataset['%_nova_jersey'] = (dataset['nova_jersey'] / dataset['nova_jersey'].sum()) * 100

dataset

dataset['%_florida'] = (dataset['florida'] / dataset['florida'].sum()) * 100

dataset

## Exercício percentuais

dataset = pd.read_csv('census.csv')

dataset.head()

dataset2 = dataset[['income', 'education']]
dataset2

dataset3 = dataset2.groupby(['education', 'income'])['education'].count()

dataset3

dataset3.index

dataset3[' Bachelors', ' <=50K'], dataset3[' Bachelors', ' >50K']

3134 + 2221

# % >50K
(2221 / 5355) * 100

# % <=50K
(3134 / 5355) * 100

## Exercício coeficientes e taxas

dados = {'ano': ['1', '2', '3', '4', 'total'],
        'matriculas_marco': [70, 50, 47, 23, 190],
        'matriculas_novembro': [65, 48, 40, 22, 175]}

dados

dataset = pd.DataFrame(dados)
dataset

dataset['taxa_evasao'] = ((dataset['matriculas_marco'] - dataset['matriculas_novembro']) / dataset['matriculas_marco']) * 100

dataset