import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import io 
from scipy import stats
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score , davies_bouldin_score



d14 = pd.read_csv('datasets/data2014.csv',sep=';')
d15 = pd.read_csv('datasets/data2015.csv',sep=';')
d16 = pd.read_csv('datasets/data2016.csv',sep=';')
d17 = pd.read_csv('datasets/data2017.csv',sep=';')
#snsi3 = pd.read_excel('SNSI3.xlsx')


d14.drop(columns=['Unnamed: 0','index'] , inplace=True)
d15.drop(columns=['Unnamed: 0','index'] , inplace=True)
d16.drop(columns=['Unnamed: 0','index'] , inplace=True)
d17.drop(columns=['Unnamed: 0','index'] , inplace=True)

df = pd.read_csv(filepath_or_buffer='datasets/Desagregado-20191010161139.csv' ,index_col=False, sep=';' , encoding='utf_16_le', thousands='.' ,decimal=',' , skipfooter=1)
df.columns
df.rename(columns={'Município' : 'municipio' 
                  ,'Ano de Referência':'ano' 
                  ,'Código do Município' : 'Cod_ibge' 
                  ,'POP_TOT - População total do município do ano de referência (Fonte: IBGE):' : 'POP_TOT' 
                  ,'POP_URB - População urbana do município do ano de referência (Fonte: IBGE):' :'POP_URB' },
                  inplace=True)
ind_dict = {
    'AG006 - Volume de água produzido' : 'AG006' ,
    'AG011 - Volume de água faturado' : 'AG011',
    'AG018 - Volume de água tratada importado' : 'AG018',
    'AG021 - Quantidade de ligações totais de água' : 'AG021',
    'ES009 - Quantidade de ligações totais de esgotos' : 'ES009',
    'FN006 - Arrecadação total' : 'FN006',
    'FN033 - Investimentos totais realizados pelo prestador de serviços' : 'FN033',
    'FN048 - Investimentos totais realizados pelo(s) município(s)' : 'FN048' ,
    'FN058 - Investimentos totais realizados pelo estado' : 'FN058',
    'IN023 - Índice de atendimento urbano de água' : 'IN023',
    'IN024 - Índice de atendimento urbano de esgoto referido aos municípios atendidos com água' : 'IN024',
    'IN046 - Índice de esgoto tratado referido à água consumida' : 'IN046',
    'IN049 - Índice de perdas na distribuição' : 'IN049',
    'IN055 - Índice de atendimento total de água' : 'IN055',
    'IN056 - Índice de atendimento total de esgoto referido aos municípios atendidos com água' : 'IN056',
    'IN075 - Incidência das análises de cloro residual fora do padrão': 'IN075',
    'IN076 - Incidência das análises de turbidez fora do padrão' : 'IN076',
    'IN079 - Índice de conformidade da quantidade de amostras - cloro residual' : 'IN079',
    'IN080 - Índice de conformidade da quantidade de amostras - turbidez':'IN080',
    'IN084 - Incidência das análises de coliformes totais fora do padrão': 'IN084',
    'IN085 - Índice de conformidade da quantidade de amostras - coliformes totais' : 'IN085'
}

df = df.rename(columns = ind_dict)
map_df = gpd.read_file('datasets/map/LimiteMunicipalPolygon.shp')
map_df = map_df.rename(columns={'Nome' : 'municipio'})


def plot_numerical_map(ano, column):
    
    if (ano == 2014):
      data = d14
    elif (ano == 2015):
      data = d15
    elif (ano == 2016):
      data = d16
    elif (ano == 2017):
      data = d17

    data = data.loc[:,['municipio',column]]
    dmunl = data['municipio'].tolist()
    dmunln = data['municipio'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    
    dmun = dict(zip(dmunln,dmunl))
    map_tmp = map_df.replace(dmun)  
    merged = map_tmp.set_index('municipio').join(data.set_index('municipio'), how='inner')
    
    vmin, vmax = data[column].quantile(0.1) , data[column].quantile(0.9)
    fig, ax = plt.subplots(1,figsize=(10,6))
    
    merged.plot(column=column , cmap='coolwarm' , vmin=vmin,vmax=vmax,linewidth=0.8,ax=ax,edgecolor='0.8', legend=True)
    
    ax.set_title('{} : Questionário de {}'.format(column,ano), fontdict={'fontsize': '30', 'fontweight' : '3'})
    ax.axis('off')
    plt.tight_layout()

    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image


def plot_qualitative_map(ano,column):
    if (ano == 2014):
      data = d14
    elif (ano == 2015):
      data = d15
    elif (ano == 2016):
      data = d16
    elif (ano == 2017):
      data = d17
    data = data.loc[:,['municipio',column]]
    dmunl = data['municipio'].tolist()
    dmunln = data['municipio'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    
    dmun = dict(zip(dmunln,dmunl))
    map_tmp = map_df.replace(dmun)
    
    merged = map_tmp.set_index('municipio').join(data.set_index('municipio'), how='inner')
    fig, ax = plt.subplots(figsize=(15,15))
    merged.plot(column=column,cmap='coolwarm' , categorical=True,linewidth=0.8,ax=ax,edgecolor='0.8', legend=True)
    ax.set_title('{} : Questionário de {}'.format(column,ano), fontdict={'fontsize': '40', 'fontweight' : '3'})
    ax.axis('off')
    
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image


desag_l = ['AG006', 'AG011', 'AG018',
       'AG021', 'ES009', 'FN006', 'FN033', 'FN048', 'FN058', 'IN023', 'IN024',
       'IN046', 'IN049', 'IN055', 'IN056', 'IN075', 'IN076', 'IN079', 'IN080',
       'IN084', 'IN085']

tce_l = ['SNSI4', 'SNSI1', 'SNSI2', 'SNSI5', 
       'TCE01', 'TCE02', 'TCE05', 'TCE06', 'TCE07', 'TCE08', 'TCE09', 'TCE11',
       'TCE12', 'TCE13', 'TCE14', 'TCE15', 'TCE16', 'TCE17', 'TCE18', 'TCE19',
       'TCE20', 'TCE21', 'TCE22', 'TCE23', 'TCE24', 'TCE25', 'TCE26', 'TCE28',
       'TCE29']


def get_history(mun, col):
    mun_tmp = df[['municipio','ano',col]].groupby(['municipio','ano']).sum()
    mun_tmp = mun_tmp.loc[(mun)].reset_index()
    title = '{}: Série histórica do indicador {}'.format(mun,col)
    fig, ax = plt.subplots()
    ax.plot(mun_tmp['ano'], mun_tmp[col], 'o-r')
    ax.set(xlabel='Anos', ylabel=col ,title=title)
    plt.ylim((0,110))
    for x,y in zip(mun_tmp['ano'], mun_tmp[col]):
      ax.annotate(str(y) , xy=(x,y+1))
    ax.grid()
    
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image

def get_mean_history(col):
    df_tmp = df[['ano',col]].groupby(['ano']).mean()
    df_tmp = df_tmp.reset_index()
    title  = 'Série histórica das médias do indicador {}'.format(col)
    fig, ax = plt.subplots()
    ax.plot(df_tmp['ano'], df_tmp[col] , 'o-r')
    ax.set(xlabel='Anos',ylabel=col,title=title)
    plt.ylim((0,110))
    for x,y in zip(df_tmp['ano'], df_tmp[col]):
      ax.annotate(str(round(y,2)) , xy=(x,y+1))
    ax.grid()

    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image


def get_data(mun, col, ano):
  if col in tce_l:
    if ano == 2014:
      data = d14.loc[(d14.municipio == mun),col].iloc[0]
    elif ano == 2015:
      data = d15.loc[(d15.municipio == mun),col].iloc[0]
    elif ano == 2016:
      data = d16.loc[(d16.municipio == mun),col].iloc[0]
    elif ano == 2017:
      data = d17.loc[(d17.municipio == mun),col].iloc[0]
  elif 2010 <= ano and ano <= 2017 and col in desag_l:
    data = df.loc[((df.municipio == mun) & (df.ano == ano)),col].iloc[0]
  else:
    data = 'Indicador não existe para esse município nesse ano'
  
  if isinstance(data , np.integer):
    return int(data)
  elif isinstance(data, np.floating):
    return float(data)
  else:
    return data

def get_mean(col, ano):
  if col in tce_l:
    if ano == 2014:
      return float(d14[col].mean())
    elif ano == 2015:
      return float(d15[col].mean())
    elif ano == 2016:
      return float(d16[col].mean())
    elif ano == 2017:
      return float(d17[col].mean())
  else:
    return float(df.loc[(df.ano == ano),col].mean())

def cluster(ano, vals , n):
    if (ano == 2014):
        data = d14
    elif (ano == 2015):
        data = d15
    elif (ano == 2016):
        data = d16
    elif (ano == 2017):
        data = d17
    else:
        return "O ano não possui dados disponíveis"
    
    feats = vals
    cols = feats + ['municipio', 'ano']
    
    
    df_c = pd.DataFrame(data[cols])
    df_c = df_c.dropna()
    
    df_c = df_c[(np.abs(stats.zscore(df_c[feats])) < 3).all(axis=1)]
    scaler = preprocessing.RobustScaler().fit(df_c[feats])
    train = scaler.transform(df_c[feats])

    kmeans = KMeans(n_clusters=n , random_state=0).fit(train)
    labels = kmeans.labels_
    df_c['labels'] = labels
    return (df_c , silhouette_score(train, labels, metric='euclidean'),davies_bouldin_score(train,labels))

def plot_cluster_map(ano,vals,n):
    data = cluster(ano,vals,n)[0]

    data = data.loc[:,['municipio','labels']]
    dmunl = data['municipio'].tolist()
    dmunln = data['municipio'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    
    dmun = dict(zip(dmunln,dmunl))
    map_tmp = map_df.replace(dmun)
    
    merged = map_tmp.set_index('municipio').join(data.set_index('municipio'), how='inner')
    fig, ax = plt.subplots(figsize=(15,15))
    merged.plot(column='labels',cmap='coolwarm' , categorical=True,linewidth=0.8,ax=ax,edgecolor='0.8', legend=True)
    ax.set_title('Clusterização por {} em {}'.format(vals,ano), fontdict={'fontsize': '40', 'fontweight' : '3'})
    ax.axis('off')

    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image

def plot_cluster(ano, vals, n):
  data = cluster(ano,vals,n)[0]
  fig = sns.pairplot(data,hue='labels',diag_kind='kde',vars=vals)

  bytes_image = io.BytesIO()
  fig.savefig(bytes_image, format='png')
  bytes_image.seek(0)
  return bytes_image
