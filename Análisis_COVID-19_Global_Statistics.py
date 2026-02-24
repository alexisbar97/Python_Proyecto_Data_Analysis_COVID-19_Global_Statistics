# URL DEL ARCHIVO CSV: https://www.kaggle.com/datasets/swatibadola156/covid-19-global-statistics-february-2026

# INSTALACIÓN DE PAQUETES REQUERIDOS

# Ejecutar en CMD: pip install kagglehub

# IMPORTACIÓN DE LIBRERÍAS

from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

# RUTA DEL ARCHIVO CSV

# local_file_path = ''
online_file_path = kagglehub.dataset_download("swatibadola156/covid-19-global-statistics-february-2026")

# CARGA DEL DATAFRAME

# df = pd.read_csv(local_file_path)
df = pd.read_csv(f"{online_file_path}/covid19_global_statistics_2026.csv")

# CONFIGURACIÓN DE VISUALIZACIÓN

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', None)

# SEPARADOR DE SECCIONES

def print_separator():
    print('\n#-------------------------------------#\n')

# TEXTO EN COLOR NEGRO

def print_black(text):
    print("\033[1m" + text + "\033[0m")

# TEXTO EN COLOR MORADO

def print_purple(text):
    print("\033[95m" + text + "\033[0m")

# EXPLORACIÓN INICIAL DEL DATAFRAME

print_separator()
print_black('EXPLORACIÓN INICIAL DEL DATAFRAME')
print_separator()

print_purple('df.columns: Columnas Del DataFrame.\n')
print(list(df.columns))

print_purple('\ndf.head(): Primeras 5 Filas Del DataFrame.\n') 
print(df.head())                                         

print_purple('\ndf.tail(): Últimas 5 Filas Del DataFrame.\n')
print(df.tail())                                        

print_purple('\ndf.sample(5): 5 Filas Aleatorias Del DataFrame.\n')
print(df.sample(5))                                                                                       

print_purple('\ndf.shape: Dimensión Del DataFrame (Filas, Columnas).\n')
print(df.shape)

print_purple('\ndf.info(): Información General Del DataFrame.\n')
df.info()

# LIMPIEZA Y MANIPULACIÓN DE DATOS

print_separator()
print_black('LIMPIEZA Y MANIPULACIÓN DE DATOS')

# LIMPIEZA DE DATOS - DETECCIÓN DE VALORES FALTANTES

# VALORES FALTANTES POR COLUMNA

print_separator()
print_purple('Valores Faltantes Por Columna.\n')
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0],'\n')

# PORCENTAJE DE VALORES FALTANTES POR COLUMNA

print_purple('Porcentaje De Valores Faltantes Por Columna.\n')                                                                               
missing_percentage = (df.isnull().sum() / len(df)) * 100                                                            
print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

# Sustitución de valores NaN con "Sin Información" en las columnas 'population', 'new_cases', 'active_cases', 'cases_per_million', 'new_deaths', 'deaths_per_million', 'total_deaths', 'test_per_million', 'total_tests'.

for col in ['population', 'new_cases', 'active_cases', 'cases_per_million', 'new_deaths', 'deaths_per_million', 'total_deaths', 'tests_per_million', 'total_tests']:
    df[col] = df[col].fillna("Sin Información")

# LIMPIEZA DE DATOS - GESTIÓN DE DATOS BASURA

# Eliminar filas que incluyen valores relacionados con cruceros Fila 5 (Diamond-Princess) y 8 (MS Zaandam).

df = df.drop([5, 8])

# Eliminar filas en que 'continent' == 'country'.

df = df.drop(df[(df['continent'] == df['country'])].index)

# Corregir índice 112 ('country' = 'R&eacute;union') -> 'Reunión'.

df.loc[112, 'country'] = 'Reunión'

# Eliminar columnas con exceso de valores NaN.

df.drop(columns=['new_cases', 'new_deaths'], inplace=True)

# Columnas del Dataframe actualizadas.

print_purple('\ndf.columns: Columnas Del DataFrame Actualizadas.\n')
print(list(df.columns))

# Resetear el index del DataFrame.

df = df.reset_index(drop=True)

# LIMPIEZA DE DATOS - GESTIÓN DE VALORES DUPLICADOS

# NÚMERO DE FILAS DUPLICADAS

num_duplicates = df.duplicated().sum()

print_purple('\nNúmero De Filas Duplicadas\n')                                                                                                                
print(f"Filas Duplicadas Encontradas: {num_duplicates}.")

# LIMPIEZA DE DATOS - CORRECCIÓN DE TIPOS DE DATOS

# TIPOS DE DATOS ORIGINALES

print_purple('\nTipos De Datos Originales:\n')
print(df.dtypes,'\n')

df['population'] = pd.to_numeric(df['population'], errors='coerce')

for col in ['active_cases', 'cases_per_million', 'total_cases', 'deaths_per_million', 'total_deaths', 'tests_per_million', 'total_tests']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# TIPOS DE DATOS CORREGIDOS

print_purple('Tipos De Datos Corregidos:\n')
print(df.dtypes)

# MANIPULACIÓN DE DATOS: CREACIÓN DE COLUMNA TASA DE MORTALIDAD

df['mortality_rate'] = (df['total_deaths'] / df['total_cases']) * 100

# MANIPULACIÓN DE DATOS: CREACIÓN DE COLUMNA TASA DE POSITIVIDAD

df['positivity_rate'] = (df['total_cases'] / df['total_tests']) * 100

# Columnas del Dataframe actualizadas.

print_purple('\ndf.columns: Columnas Del DataFrame Actualizadas.\n')
print(list(df.columns))

# MANIPULACIÓN DE DATOS: AGRUPAR DATOS POR CONTINENTE Y CALCULAR LA SUMA DE 'total_cases' Y 'total_deaths'. ADEMÁS DEL PROMEDIO DE 'mortality_rate'.

continent_data = df.groupby('continent').agg({
    'total_cases': 'sum',
    'total_deaths': 'sum',
    'mortality_rate': 'mean'
}).reset_index()

print_purple('\nDatos Agrupados Por Continente:\n')
print(continent_data)

# ANÁLISIS EXPLORATORIO DE DATOS (EDA)

print_separator()
print_black('ANÁLISIS EXPLORATORIO DE DATOS (EDA)')
print_separator()

# PREGUNTA 1: Genera un gráfico de barras mostrando los 10 países con más total_cases.

# FILTRADO

top_cases = df.sort_values(by='total_cases', ascending=False).head(10)

# GRÁFICO

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

ax = sns.barplot(
    x='total_cases', 
    y='country', 
    data=top_cases, 
    palette="magma",
    hue='country',  
    edgecolor='0.3',
    legend=False    
)

max_val = top_cases['total_cases'].max()

ticks_pos = np.arange(0, max_val + 20e6, 20e6)

ax.set_xticks(ticks_pos)

ax.set_xticklabels(
    [f'{int(t/1e6)}M' for t in ticks_pos],
    fontsize=12,
    fontweight='bold'
)

ax.set_yticks(
    range(len(top_cases))
) 

ax.set_yticklabels(
    top_cases['country'],
    fontsize=12,
    fontweight='bold'
)

for i, v in enumerate(top_cases['total_cases']):
    ax.text(
        v + (max_val * 0.01), 
        i, f'{v/1e6:.1f}M', 
        va='center',
        fontsize=12, 
        fontweight='bold',
        color='#444444'
    )

plt.title(
    'TOP 10 PAÍSES CON MÁS CASOS DE COVID-19', 
    fontsize=22, 
    fontweight='heavy', 
    pad=20
)

plt.xlabel('')
plt.ylabel('') 

sns.despine(
    left=True, 
    bottom=True
)

plt.tight_layout()
plt.show()

# PREGUNTA 2: Genera otro gráfico mostrando los 10 países con mayor deaths_per_million. (Esto suele mostrar una realidad diferente a los números absolutos).

# FILTRADO

top_deaths = df.sort_values(by='deaths_per_million', ascending=False).head(10)

# GRÁFICO

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

ax = sns.barplot(
    x='deaths_per_million', 
    y='country', 
    data=top_deaths, 
    palette="magma",
    hue='country',  
    edgecolor='0.3',
    legend=False    
)

max_val = top_deaths['deaths_per_million'].max()

ticks_pos = np.arange(0, max_val + 1000, 1000)

ax.set_xticks(ticks_pos)

ax.set_xticklabels(
    [f'{int(t)}' for t in ticks_pos],
    fontsize=12,
    fontweight='bold'
)

ax.set_yticks(
    range(len(top_deaths))
) 

ax.set_yticklabels(
    top_deaths['country'],
    fontsize=12,
    fontweight='bold'
)

for i, v in enumerate(top_deaths['deaths_per_million']):
    ax.text(
        v + (max_val * 0.01), 
        i,
        f'{v:.0f} MPM', 
        va='center', 
        fontsize=12, 
        fontweight='bold', 
        color='#444444'
    )

plt.title(
    'TOP 10 PAÍSES CON MÁS MUERTES POR MILLÓN DE HABITANTES A CAUSA DEL COVID-19',
    fontsize=22,
    fontweight='heavy',
    pad=20
)

plt.xlabel('') 
plt.ylabel('') 

sns.despine(
    left=True, 
    bottom=True
)

plt.tight_layout()
plt.show()

# PREGUNTA 3: Genera un Scatter Plot (gráfico de dispersión): Eje X = population, Eje Y = total_cases.

# GRÁFICO

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))

ax = sns.scatterplot(
    x='population', 
    y='total_cases', 
    data=df, 
    palette="magma",
    hue='total_cases',
    size='total_cases',
    sizes=(20, 200),
    edgecolor='0.3',
    alpha=0.7,
    legend=False    
)

ax.set_xscale('log')
ax.set_yscale('log')

def millones_formatter(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:g}M'
    elif x >= 1e3:
        return f'{x*1e-3:g}k'
    else:
        return f'{x:g}'

formatter = FuncFormatter(millones_formatter)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)

ax.set_title(
    'RELACIÓN ENTRE POBLACIÓN Y CASOS DE COVID-19', 
    fontsize=22,
    fontweight='heavy',
    pad=20
)

ax.set_xlabel(
    'POBLACIÓN',
    fontsize=18,
    labelpad=15,
    fontweight='heavy'
)
ax.set_ylabel(
    'TOTAL DE CASOS',
    fontsize=18,
    labelpad=15,
    fontweight='heavy'
) 

sns.despine(
    left=True,
    bottom=True
)

plt.tight_layout()
plt.show()

# PREGUNTA 4: Crea un Histograma de la columna mortality_rate. ¿Sigue una distribución normal o está sesgada?

# GRÁFICO

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

ax = sns.histplot(
    df['mortality_rate'],
    kde=True,
    color='#D6D6D6', 
    edgecolor='white', 
    alpha=1
)

ax.lines[0].set_color('#E50914')
ax.lines[0].set_linewidth(1.5)
ax.set_xlim(left=0)

plt.title(
    'DISTRIBUCIÓN DE LA TASA DE MORTALIDAD',
    fontsize=22,
    fontweight='heavy',
    pad=20
)

plt.xlabel(
    'TASA DE MORTALIDAD',
    fontsize=18,
    labelpad=15,
    fontweight='heavy'
)

plt.ylabel(
    'FRECUENCIA',
    fontsize=18,
    labelpad=15,
    fontweight='heavy'
)

sns.despine(
    left=True,
    bottom=True
)

plt.tight_layout()
plt.show()

# PREGUNTA 5: Crea un Boxplot de cases_per_million dividido por continent. Esto te permitirá ver outliers (países con casos inusualmente altos dentro de su región).

# GRÁFICO

sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 8))

ax = sns.boxplot(
    x=df['continent'].str.upper(), 
    y='cases_per_million', 
    data=df, 
    palette="magma",
    hue='continent',      
    linewidth=1.5,
    fliersize=5,           
    legend=False          
)

def miles_formatter(x, pos):
    return f'{int(x/1000)}k' if x != 0 else '0'

ax.yaxis.set_major_formatter(FuncFormatter(miles_formatter))

plt.title(
    'DISTRIBUCIÓN DE CASOS DE COVID-19 POR MILLÓN DE HABITANTES SEGÚN CONTINENTE',
    fontsize=22,
    fontweight='heavy',
    pad=25
)

plt.xlabel(
    'CONTINENTE',
    fontsize=18,
    labelpad=15,
    fontweight='heavy'
)

plt.xticks(
    fontsize=12,
    fontweight='bold'
)

plt.yticks(
    fontsize=12,
    fontweight='bold'
)

plt.ylabel(
    'CASOS POR MILLÓN',
    fontsize=18,
    labelpad=15,
    fontweight='heavy'
)

ax.set_ylim(bottom=0)

sns.despine(
    left=True,
    bottom=True
)

plt.tight_layout()
plt.show()

# PREGUNTA 6: Calcula la matriz de correlación entre todas las variables numéricas. Visualízala con un Heatmap de Seaborn. Pregunta a resolver: ¿Qué variable tiene mayor correlación con total_deaths: total_cases, population o active_cases?

# FILTRADO

columnas_numericas = ['population', 'total_cases', 'active_cases', 'total_deaths', 'total_tests', 'mortality_rate', 'positivity_rate', 'cases_per_million', 'deaths_per_million', 'tests_per_million']

# MATRIZ DE CORRELACIÓN

corr_matrix = df[columnas_numericas].corr()

# GRÁFICO

sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 10))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

ax = sns.heatmap(
    corr_matrix,
    mask=mask,               
    cmap='magma',            
    annot=True, 
    fmt=".3f",               
    linewidths=0.5, 
    cbar_kws={'ticks': [-1, -0.5, 0, 0.5, 1], 'label': 'CORRELACIÓN'},
    vmin=-1, vmax=1
)

ax.set_title(
    'MATRIZ DE CORRELACIÓN ENTRE VARIABLES NUMÉRICAS',
    fontsize=22,
    fontweight='heavy',
    pad=30
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    fontsize=12,
    fontweight='bold',
    rotation=45,
    ha='right'               
)

ax.set_yticklabels(
    ax.get_yticklabels(),
    fontsize=12,
    fontweight='bold'
)

sns.despine(
    left=True,
    bottom=True
)

plt.tight_layout()
plt.show()

# PREGUNTA 7: Detección de Anomalías: Filtra los países que tengan una mortality_rate superior al percentil 95 (el top 5% más letal). ¿Qué tienen en común esos países?

print_purple('Pregunta 7: Detección De Anomalías: Filtra Los Países Que Tengan Una Mortality_rate Superior Al Percentil 95 (El Top 5% Más Letal). ¿Qué Tienen En Común Esos Países?\n')

percentil_95 = df['mortality_rate'].quantile(0.95)
anomalos = df[df['mortality_rate'] > percentil_95]

print('Detección De Anomalías:\n')
print(anomalos)

# PREGUNTA 8: Comparativa de Testing: Compara países con alta tasa de testeo (tests_per_million) vs. baja tasa de testeo. ¿Afecta esto a la tasa de mortalidad reportada?

print_purple('\nPregunta 8: Comparativa De Testing: Compara Países Con Alta Tasa De Testeo (tests_per_million) Vs. Baja Tasa De Testeo. ¿Afecta Esto A La Tasa De Mortalidad Reportada?\n')

percentil_75_tests = df['tests_per_million'].quantile(0.75)
percentil_25_tests = df['tests_per_million'].quantile(0.25)

print(f"Promedio de Mortalidad en Países con Alta Tasa de Testeo: {df[df['tests_per_million'] > percentil_75_tests]['mortality_rate'].mean():.2f}")
print(f"Promedio de Mortalidad en Países con Baja Tasa de Testeo: {df[df['tests_per_million'] < percentil_25_tests]['mortality_rate'].mean():.2f}")

# PREGUNTA 9: Comparativa de Mortalidad: Compara países con alta mortalidad (mortality_rate) vs. baja mortalidad. ¿Afecta esto a la tasa de testeo reportada?

print_purple('\nPregunta 9: Comparativa De Mortalidad: Compara Países Con Alta Mortalidad (mortality_rate) Vs. Baja Mortalidad. ¿Afecta Esto A La Tasa De Testeo Reportada?\n')

percentil_75_mortality = df['mortality_rate'].quantile(0.75)
percentil_25_mortality = df['mortality_rate'].quantile(0.25)

print(f"Promedio de Testeo en Países con Alta Mortalidad: {df[df['mortality_rate'] > percentil_75_mortality]['tests_per_million'].mean():.2f}")
print(f"Promedio de Testeo en Países con Baja Mortalidad: {df[df['mortality_rate'] < percentil_25_mortality]['tests_per_million'].mean():.2f}")
