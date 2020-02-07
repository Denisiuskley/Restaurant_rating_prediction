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

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

df = pd.read_csv('main_task.xls')

'''Замена символов на коды в ценовом диапазоне'''
df.loc[df['Price Range'] == '$', 'Price ID'] = 1
df.loc[df['Price Range'] == '$$ - $$$', 'Price ID'] = 2
df.loc[df['Price Range'] == '$$$$', 'Price ID'] = 3
'''******************************************'''

'''Нормировка рейтинга в пределах каждого города'''
df['Ranking norm'] = df.groupby('City')['Ranking'].transform(lambda x:
        1+(5-1)*(x - x.min())/(x.max()-x.min()))
df['Ranking norm'] = 6 - df['Ranking norm']

'''Создание метрики количества кухонь'''
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

'''Обработка отзывов. Выберем все слова в один массив, а даты в другой'''
pattern = re.compile('\w+')
m = []
k = []
for index, row in df.iterrows(): 
    if type(row['Reviews']) == str:
        l = row['Reviews'][1:-1].split('], [')
        l1 = l[0].split(', ')
        l2 = l[1].split(', ')
        for i in l1:
            if i != '':
                i2 = i.split()
                for j in i2:
                    n = pattern.findall(j)
                    if len(n) > 0 and len(n[0]) > 2:
                        k.append([row['ID_TA'],n[0].lower(),row['Ranking norm']])
        for i in l2:
            if i != '':
                m.append([row['ID_TA'],i[1:-1]])

'''Выбираем наиболее популярные слова и создаем метрики положительного и отрицательного окраса.'''
Words = pd.DataFrame(k, columns = ['ID_TA', 'Word','Ranking norm'])
Words_value = Words['Word'].value_counts().reset_index()
Words_rating = Words.groupby(['Word'])['Ranking norm'].sum().reset_index()
Words_value.columns = ['Word','Num']
Words_value = pd.merge(Words_value, Words_rating, on='Word', how='left')


good_words = ['best','excellent','delicious','amazing','fantastic',
              'wonderful','perfect','awesome','super','superb','fabulous',
              'xcellent','charming','delightful','incredible','great'
              'wow','ideal','excelent','perfectly','jewel','great',
              'good','nice','tasty','beautiful','friendly','lovely','quick',
              'fresh','decent','cosy','pleasant','cozy','fast',
              'reasonable','taste','fun','well','yummy','interesting','fine',
              'healthy','loved','brilliant','relaxed','cute','enjoyable',
              'affordable','relaxing','welcoming','convenient','enjoy',
              'comfortable','reasonably','romantic','cheerful','inexpensive',
              'unusual','lucky','pleasantly','relax','decent',
              'pretty','happy','welcome','phenomenal','magic','discovery',
              'finest','exquisite','memorable','highly','beautifully','creative',
              'gem','vegan','michelin','discover','lover','terrific','perfection',
              'unforgettable','innovative','friendliest','outstanding','wholesome',
              'refreshing','dream','art','hidden','empanadas','private','must',
              'personal','oasis','exceptional','extraordinary','home','homemade',
              'organic','yummy']

bad_words = ['bad','small','worth','terrible','but','poor','worst',
             'disappointing','overpriced','rude','horrible','awful',
             'pricey','disappointed','unfriendly','disappointment','noisy','die',
             'never','not','wrong','nothing',' inedible',
             'dodgy','avoid','dirty','away','unpleasant','garbage','hideous',
             'warning','disgusting','ripoff','rip','trap','ripped','rubish',
             'disgraceful','hate','shame','bother','dreadful','appalling',
             'wouldn','waste','rubbish','awful','worse','hut','careful',
             'dissapointed','tasteless','underwhelming','dirty','dont','pretentious',
             'arrogant','disappointing','care','salty','didn','average','boring',
             'nothing','cold','over','don','slow','sadly','expensive','crazy',
             'not','less','despite','could','empty','wrong','pricey']

Words['ID Word'] = Words['Word'].apply(lambda x: 5 if x in good_words else 
     (1 if x in bad_words else np.nan))

Words_id = Words.groupby(['ID_TA'])['ID Word'].mean().reset_index()
df = pd.merge(df, Words_id, on='ID_TA', how='left')
'''Анализ показывает, что множества линейно неразделимы.'''

'''Собираем признаки по датам. Формируем метрики разницы дат отзывов
 и древность последнего отзыва'''
Dates = pd.DataFrame(m, columns = ['ID_TA', 'Date'])
Dates['Date'] = pd.to_datetime(Dates['Date'])

Date_first = Dates.groupby(['ID_TA'])['Date'].min().reset_index()
Date_first.columns = ['ID_TA','Date first review']
df = pd.merge(df, Date_first, on='ID_TA', how='left')

Date_second = Dates.groupby(['ID_TA'])['Date'].max().reset_index()
Date_second.columns = ['ID_TA','Date second review']
df = pd.merge(df, Date_second, on='ID_TA', how='left')

df['Period'] = df['Date second review'] - df['Date first review']
a = pd.to_datetime('today')
df['Sec to Now'] = df['Date second review'].apply(lambda x: (a - x).days)
df['Period'] = df['Period'].apply(lambda x: x.days)

'''Вычисление параметра разницы между ID ресторана и местом в городе'''
df['dif Rank rID'] = df['Restaurant_id'].apply(lambda x: int(x[3:])) - df['Ranking']


'''Исключение пропусков'''
df = df.sort_values('Ranking norm')
df['Num cuisines in City_isNAN'] = pd.isna(df['Num cuisines in City']).astype('uint8')
df['Num cuisines in City'] = df.groupby('City')['Num cuisines in City'].transform(lambda x: x.fillna(x.median()))
df['ID Word_isNAN'] = pd.isna(df['ID Word']).astype('uint8')
df['ID Word'] = df.groupby('City')['ID Word'].transform(lambda x: x.fillna(x.mean()))
df['Period_isNAN'] = pd.isna(df['Period']).astype('uint8')
df['Period'] = df['Period'].fillna(method = 'backfill')
df['Sec to Now_isNAN'] = pd.isna(df['Sec to Now']).astype('uint8')
df['Sec to Now'] = df['Sec to Now'].fillna(method = 'backfill')
df['Number of Reviews_isNAN'] = pd.isna(df['Number of Reviews']).astype('uint8')
df.loc[(pd.isna(df['Number of Reviews'])) & (df['Period'] == 0), 'Number of Reviews'] = 1
df.loc[pd.isna(df['Number of Reviews']), 'Number of Reviews'] = 0
df['Price ID_isNAN'] = pd.isna(df['Price ID']).astype('uint8')
df['Price ID'] = df['Price ID'].fillna(2)

'''Дополнительные метрики, хорошо коррелирующие со средними по городам рейтингами'''
df['ID Word mean City'] = df.groupby('City')['ID Word'].transform(lambda x: x.mean())
df['Price ID mean City'] = df.groupby('City')['Price ID'].transform(lambda x: x.mean())
df['Word/Price'] = df['ID Word'] / df['Price ID']
df['Word/Price mean City'] = df['ID Word mean City'] / df['Price ID mean City']

df['ID Word mean City rank'] = df['ID Word mean City'] * df['Ranking norm']
df['Price ID mean City rank'] = df['Price ID mean City'] * df['Ranking norm']
df['Word/Price mean City rank'] = df['Word/Price mean City'] * df['Ranking norm']

'''Создание признаков dummy variables'''
'''Метрики городов (немного уменьшает ошибку)'''
df = pd.concat([df, pd.get_dummies(df['City'])], axis=1, sort=False)

iskl = ['Restaurant_id', 'Rating','City','Cuisine Style','Price Range',
         'Reviews','URL_TA','ID_TA','Date first review','Date second review','Ranking']

X = df.drop(iskl, axis = 1)
y = df['Rating']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
regr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

fi = pd.DataFrame({'feature': list(X.columns),
                   'importance': regr.feature_importances_}).\
                    sort_values('importance', ascending = False)





