try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import  pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import decomposition
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

df = pd.read_csv('Changing_data.csv')
data = pd.DataFrame()
'''Замена символов на коды в ценовом диапазоне'''
df.loc[df['Price Range'] == '$', 'Price ID'] = 1
df.loc[df['Price Range'] == '$$ - $$$', 'Price ID'] = 2
df.loc[df['Price Range'] == '$$$$', 'Price ID'] = 3
'''******************************************'''
data = df['Price ID']

df['Ranking norm'] = df.groupby('City')['Ranking'].transform(lambda x:
        1+(5-1)*(x - x.min())/(x.max()-x.min()))
df['Ranking norm'] = 6 - df['Ranking norm']

pattern = re.compile('\'\w+')
df['Num Cuisine'] = df['Cuisine Style'].apply(lambda x: 1 if pd.isna(x) 
        else len(pattern.findall(x)))
n = df['Cuisine Style'].tolist()
id_ta = df['ID_TA'].tolist()
city = df['City'].tolist()
m = []
for i in range(len(n)):
    try:
        l = n[i][1:-1].split(', ')
        for j in l:
            m.append([id_ta[i],city[i],j[1:-1]])
    except: pass
        
Cuisine = pd.DataFrame(m, columns = ['ID_TA', 'City', 'Cuisine'])
pivot = Cuisine.pivot_table(values=['ID_TA'],
                       index=['City'],
                       columns=['Cuisine'],
                       aggfunc='count',
                       fill_value=0)
col = pivot.index
pivot = Cuisine.pivot_table(values=['ID_TA'],
                       index=['Cuisine'],
                       columns=['City'],
                       aggfunc='count',
                       fill_value=0)
pivot.columns = col

for i in range(len(m)):
    try:
        m[i].append(pivot.loc[m[i][2] , m[i][1]])
    except: pass

Cuisine = pd.DataFrame(m, columns = ['ID_TA', 'City', 'Cuisine', 'Num cuisines in City'])
Cuisine_gr = Cuisine.groupby(['ID_TA'])['Num cuisines in City'].sum().sort_values(ascending=False)
df = pd.merge(df, Cuisine_gr, on='ID_TA', how='left')

df['Rank1 norm'] = df.groupby('City')['Pars Rank1'].transform(lambda x:
        1+(5-1)*(x - x.min())/(x.max()-x.min()))
df['Rank1 norm'] = 6 - df['Rank1 norm']

df['Rank2 norm'] = df.groupby('City')['Pars Rank2'].transform(lambda x:
        1+(5-1)*(x - x.min())/(x.max()-x.min()))
df['Rank2 norm'] = 6 - df['Rank2 norm']



list_data = ['Rating','Restaurant_id','City','Country','Price ID','Ranking norm','Num Cuisine','Num cuisines in City','Rank1 norm',
           'Rank2 norm','Number of Reviews','Pars Num of Reviews','Pars Num Foto',
           'Pars Excellent','Pars Good','Pars Average','Pars Poor','Pars Terible']
data = df.loc[:,list_data]

col = list(data.columns)
for colname in col:
    data[colname+'_isNAN'] = pd.isna(data[colname]).astype('uint8')
    
data['Rank1 norm'] = data['Rank1 norm'].fillna(data['Ranking norm'])
data['Rank2 norm'] = data.groupby('City')['Rank2 norm'].transform(lambda x: x.fillna(x.mean()))
data['Price ID'] = data['Price ID'].fillna(2)
data['Num cuisines in City'] = data.groupby('City')['Num cuisines in City'].transform(lambda x: x.fillna(x.median()))
data['Number of Reviews'] = data['Number of Reviews'].fillna(0)
data['Pars Num of Reviews'] = data['Pars Num of Reviews'].fillna(0)
data['Pars Num Foto'] = data['Pars Num Foto'].fillna(0)
data['Pars Excellent'] = data['Pars Excellent'].fillna(0)
data['Pars Good'] = data['Pars Good'].fillna(0)
data['Pars Average'] = data['Pars Average'].fillna(0)
data['Pars Poor'] = data['Pars Poor'].fillna(0)
data['Pars Terible'] = data['Pars Terible'].fillna(0)


data['Delta Rank'] = data['Rank1 norm'] - data['Ranking norm']


data['Meanmark'] = data[['Pars Excellent','Pars Good','Pars Average','Pars Poor',
    'Pars Terible']].apply(lambda x: (5 * x[0] + 4 * x[1] + 3 * x[2] + 2 * x[3] +
    x[4]) / x.sum() if x.sum() > 0 else 0, axis = 1)

data.info()
decr = data.describe()

#df = df.drop(list_data, axis = 1)


col = list(df.columns)
ind1 = col.index('Rev1 mark')

#CurrentTime = time.time()
c = 30
m = []
for i in range(len(df)): 
     m.append(df.iloc[i,ind1:ind1+c].tolist())
     
col = col[ind1:ind1+c]

nlist = [0 for x in range(c)]     
for i in range(len(m)):
    if pd.isnull(m[i][1]):
        m[i] = nlist
        continue
    for j in range(c):
         if pd.isnull(m[i][j]) and j >= 3:
             m[i][j] = m[i][j-3]

df_date = pd.DataFrame(m, columns = col)
today = pd.to_datetime('today')
for i in range(len(col)):
    if (i - 1) % 3 == 0:
        df_date.iloc[:,i] = df_date.iloc[:,i].apply(lambda x: (today - pd.to_datetime(x)).days)

'''Функция подсчета рейтинга по отзывам с эффектом уменьшения веса по мере устаревания
Функцию устаревания по хорошему нужно подбирать оптимизацией. Пока взял по аналогии с каглом'''
def rating_by_rev(x):
    a = 0
    b = 0
    for i in range(0,len(x)-1,3):
        a += x[i]*np.exp(-x[i + 1]/500)
        b += np.exp(-x[i + 1]/500)
    try: return a / b
    except: return 0
        
df_date['rating_by_rev'] = df_date.apply(rating_by_rev, axis = 1)

data = pd.concat([data, df_date], axis = 1, sort = False)        
#print(time.time() - CurrentTime)

'''Создание признаков dummy variables'''
data = pd.concat([data, pd.get_dummies(df['City'])], axis=1, sort=False)
data = pd.concat([data, pd.get_dummies(df['Country'])], axis=1, sort=False)

data_train = data[data['Rating'] > 0]
data_test = data[data['Rating'] == 0]

iskl = ['Rating','Restaurant_id','City','Country']
#data_train = data_train.drop(iskl, axis = 1)




X = data_train.drop(iskl, axis = 1)
y = data_train['Rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
regr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
fi = pd.DataFrame({'feature': list(X.columns),
                   'importance': regr.feature_importances_}).\
                    sort_values('importance', ascending = False)

data_for_kaggle = data_test['Restaurant_id']
data_test = data_test.drop(iskl, axis = 1)

pred_test = regr.predict(data_test)

#best_pars = ['Meanmark','Ranking norm','Number of Reviews','Delta Rank','rating_by_rev',
#             'Rev1 date','Rank1 norm']
#X_ = X[best_pars]
#X_for_pca = X.drop(best_pars, axis = 1)
#y_pred = []
#for i in range(1,5):
#    pca = decomposition.PCA(n_components=i)
#    pca.fit(X_for_pca)
#    X_pca = pca.transform(X_for_pca)
#    X_all = np.concatenate((X_, X_pca), axis=1, out=None)
#    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.25)
#    regr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
#    regr.fit(X_train, y_train)
#    y_pred.append(regr.predict(X_test))
#    
#    print('MAE:', metrics.mean_absolute_error(y_test, y_pred[i]))
#
#print('MAE:', metrics.mean_absolute_error(y_test, np.mean(y_pred)))

                    
                    
                    
                    
                    
                    
                    
                    
                    