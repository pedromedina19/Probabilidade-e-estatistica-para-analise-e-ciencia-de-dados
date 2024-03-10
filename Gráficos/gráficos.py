# -*- coding: utf-8 -*-

# Gráficos
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Base de dados

dataset = pd.read_csv('census.csv')
dataset.head()

## Gráfico de dispersão

sns.relplot(x = 'age', y = 'final-weight', data=dataset,
            hue = 'income', style = 'sex', size = 'education-num');

## Gráfico de barra e setor (pizza)

sns.barplot(x = 'sex', y = 'final-weight', data=dataset, hue = 'income');

dados_agrupados = dataset.groupby(['income'])['education-num'].sum()
dados_agrupados

dados_agrupados.plot.bar();

dados_agrupados.plot.pie();

## Gráfico de linha

vendas = {'mes': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
          'valor': np.array([100, 200, 120, 300, 500, 198, 200, 209, 130, 500, 300, 120])}

vendas_df = pd.DataFrame(vendas)
vendas_df

sns.relplot(x = 'mes', y = 'valor', kind = 'line', data=vendas_df);

## Boxplot

sns.boxplot(dataset['age']);

sns.boxplot(dataset['education-num']);

dataset2 = dataset.iloc[:, [0, 4, 12]]
dataset2.head()

sns.boxplot(data=dataset2);

## Gráficos com atributos categóricos

sns.catplot(x = 'income', y = 'hour-per-week', data=dataset, hue = 'sex');

sns.catplot(x = 'income', y = 'hour-per-week',
            data=dataset.query('age < 20'), hue = 'sex');

## Subgráficos

g = sns.FacetGrid(dataset, col = 'sex', hue = 'income')
g.map(sns.scatterplot, 'age', 'final-weight');

g = sns.FacetGrid(dataset, col = 'workclass', hue = 'income')
g.map(sns.scatterplot, 'age', 'final-weight');

g = sns.FacetGrid(dataset, col = 'sex', hue = 'income')
g.map(sns.histplot, 'age');

dataset2.head()

g = sns.PairGrid(dataset2)
g.map(sns.scatterplot)

g = sns.PairGrid(dataset2)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot);

## Mapa

#!apt-get install libgeos-3.5.0
#!apt-get install libgeos-dev
#!pip install https://github.com/matplotlib/basemap/archive/master.zip

#!python -m pip install basemap

from mpl_toolkits.basemap import Basemap

dataset = pd.read_csv('house_prices.csv')
dataset.head()

dataset = dataset.sort_values(by = 'price', ascending = True)
dataset.head()

dataset.tail()

dataset_caros = dataset[0:1000]
dataset_caros

dataset_baratos = dataset[0:1000]
dataset_baratos

dataset['lat'].describe()

dataset['long'].describe()

lat1, lat2 = dataset['lat'].min(), dataset['lat'].max()
lon1, lon2 = dataset['long'].min(), dataset['long'].max()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
m = Basemap(projection='cyl', resolution='i',
            llcrnrlat = lat1, urcrnrlat = lat2,
            llcrnrlon = lon1, urcrnrlon = lon2)
m.drawcoastlines()
m.fillcontinents(color = 'palegoldenrod', lake_color='lightskyblue')
m.drawmapboundary(fill_color='lightskyblue')
m.scatter(dataset['long'], dataset['lat'], s = 5, c = 'green', alpha = 0.1, zorder = 2)
m.scatter(dataset_caros['long'], dataset_caros['lat'], s = 10, c = 'red', alpha = 0.1, zorder = 3)
m.scatter(dataset_baratos['long'], dataset_baratos['lat'], s = 10, c = 'blue', alpha = 0.1, zorder = 4)