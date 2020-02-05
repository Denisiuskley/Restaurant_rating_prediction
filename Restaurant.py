try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

'''Restaurant_id — идентификационный номер ресторана;
City — город, в котором находится ресторан;
Cuisine Style — стиль или стили, к которым можно отнести блюда, предлагаемые в ресторане;
Ranking — место, которое занимает данный ресторан среди всех ресторанов своего города;
Rating — рейтинг ресторана по данным TripAdvisor (именно это значение должна будет предсказывать модель);
Price Range — диапазон цен в ресторане;
Number of Reviews — количество отзывов о ресторане;
Reviews — данные о двух отзывах, которые отображаются на сайте ресторана;
URL_TA — URL страницы ресторана на TripAdvosor;
ID_TA — идентификатор ресторана в базе данных TripAdvisor'''

import  pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

df = pd.read_csv('main_task.xls')
Capitals = pd.read_csv('Capitals.csv', sep = ';')
df = pd.merge(df, Capitals, on='City', how='left')
df = df.drop(['Country'], axis = 1)
df['Capital'] = df['Capital'].fillna(0)
s = df['Capital'].value_counts()
df = df.sort_values('City')
#raise Exception()
City_count = df.City.value_counts().reset_index()
a2 = df['Cuisine Style'].value_counts(normalize=True).reset_index()
a3 = df['Price Range'].value_counts().reset_index()


'''Информация по пропущенным значениям'''
#df.info()
#
#for col in df.columns:
#    s = df[col].isna().value_counts()
#    print(col, s)
'''******************************************'''

#plt.figure()
#sns.set()
#sns.countplot(x = 'City', data = df)
#plt.show()

'''Замена символов на коды в ценовом диапазоне'''
def pf(p):
    m = []
    for i in p:
        if i == '$':
            m.append(1)
        elif i == '$$ - $$$':
            m.append(2)
        elif i == '$$$$':
            m.append(3)
        else:
            m.append(np.nan)
    return m
df['Price ID'] = pf(df['Price Range'])
'''******************************************'''
#a4 = df.groupby(['City'])['Price ID'].agg(pd.Series.mode).sort_values(ascending=False).reset_index()

#plt.figure()
#sns.distplot(df['Price ID'].fillna(0))
#plt.show()


#plt.figure()
#sns.distplot(df['Price ID'].fillna(0))
#plt.show()

'''Графики парных корреляций'''
#plt.figure()
#correlation = df.corr()
#sns.heatmap(correlation, annot = True, cmap = 'coolwarm')
#dfP = df.loc[df['City'] == 'Paris']
#plt.figure()
#correlation = dfP.corr()
#sns.heatmap(correlation, annot = True, cmap = 'coolwarm')

#dfP = df.loc[df['City'] == 'Paris']
#plt.figure()
#correlation = dfP.corr()
#sns.heatmap(correlation, annot = True, cmap = 'coolwarm')

'''Убедившись, что в отдельном городе связи параметров выше, создаем 
идентификатор городов для улучшения классификации'''
City_count.columns = ['City', 'Count rest in city']
City_count['City ID'] = [x for x in range(1,City_count.shape[0]+1)]
df = pd.merge(df, City_count, on='City')

'''Нормировка рейтинга в пределах каждого города. Существенно увеличивает корреляцию'''
df['Ranking norm'] = df.groupby('City')['Ranking'].transform(lambda x:
        1+(5-1)*(x - x.min())/(x.max()-x.min()))

df['Ranking norm'] = 6 - df['Ranking norm']
    
#plt.figure(1)
#plt.subplot(5,6,1)
#df[df['City'] == City_count['City'][0]].plot(
#        x = 'Ranking',
#        y = 'Rating',
#        kind = 'scatter',
#        grid = True,
#        title = 'Ranking Vs Rating',ax=plt.gca())
#    
#for i in range(1,City_count.shape[0]):
#    plt.subplot(6,6,i+1)
#    df[df['City'] == City_count['City'][i]].plot(
#            x = 'Ranking',
#            y = 'Rating',
#            kind = 'scatter',
#            grid = True,
#            title = 'Ranking Vs Rating',
#            ax=plt.gca())

#raise Exception()

id_rest = df['Restaurant_id'].value_counts().reset_index()
id_rest.columns = ['Restaurant_id', 'Rest count']
'''Оказывается, некоторые рестораны являются сетевыми, что тоже может влиять
на рейтинг, соответственно, в метрики нужно добавить количество ресторанов 
данной сети'''
df = pd.merge(df, id_rest, on='Restaurant_id')

#raise Exception()

'''Проверка на уникальность индексов ТА. Имеются повторяющиеся значения'''
id_ta = df.groupby(['ID_TA'])['City'].count().sort_values(ascending=False)
id_ta = df.loc[df['ID_TA'] == 'd7337366']

'''Составление списка кухонь'''
m = []
k = []
for index, row in df.iterrows(): 
    if type(row['Cuisine Style']) == str:
        l = row['Cuisine Style'][1:-1].split(', ')
        k.append([row['ID_TA'],len(l)])
        for i in l:
            m.append([row['ID_TA'],i[1:-1]])


Cuisine = pd.DataFrame(m, columns = ['ID_TA', 'Cuisine'])
Cuisine_val = Cuisine['Cuisine'].value_counts().reset_index()
Cuisine_val.columns = ['Cuisine', 'freq']

Cuisine_kol = pd.DataFrame(k, columns = ['ID_TA', 'Num Cuisine'])

'''Создание метрики, оценивающей популярность предлагаемой кухни.
На первом этапе попробуем среднее из популярностей предлагаемых вариантов кухонь'''
Cuisine = pd.merge(Cuisine, Cuisine_val, on='Cuisine')

Cuisine.columns = ['ID_TA','Cuisine','freq sum']
Cuisine_gr = Cuisine.groupby(['ID_TA'])['freq sum'].sum().sort_values(ascending=False)
df = pd.merge(df, Cuisine_gr, on='ID_TA', how='left')



'''Создание метрики количества кухонь'''
df = pd.merge(df, Cuisine_kol, on='ID_TA', how='left')
df['Num Cuisine'] = df['Num Cuisine'].fillna(1)
m = df['Num Cuisine'].mean()

#plt.figure()
#correlation = df.corr()
#sns.heatmap(correlation, annot = True, cmap = 'coolwarm')


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
Words_val = Words['Word'].value_counts().reset_index()
Words_rating = Words.groupby(['Word'])['Ranking norm'].mean().reset_index()
Words_val.columns = ['Word','Num']
Words_val = pd.merge(Words_val, Words_rating, on='Word', how='left')
s = Words_val[Words_val['Num'] > 10]
#raise Exception()

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
all_words = good_words + bad_words
Words['ID Word'] = Words['Word'].apply(lambda x: 5 if x in good_words else 
     (1 if x in bad_words else np.nan))

Words_id = Words.groupby(['ID_TA'])['ID Word'].mean().reset_index()
df = pd.merge(df, Words_id, on='ID_TA', how='left')

#plt.figure()
#sns.set_style('whitegrid')
#sns.distplot(df[df['ID Word'] == 1]['Rating'], kde = True, norm_hist = True)
#sns.distplot(df[(df['ID Word'] > 1) & (df['ID Word'] < 5)]['Rating'], kde = True, norm_hist = True)
#sns.distplot(df[df['ID Word'] == 5]['Rating'], kde = True, norm_hist = True)
#plt.legend(['bad', 'good', 'best'])

'''Анализ показывает, что множества линейно неразделимы.'''
 

'''Собираем признаки по датам. Формируем метрики разницы дат отзывов
 и древность последнего отзыва'''
Dates = pd.DataFrame(m, columns = ['ID_TA', 'Date'])
Dates['Date'] = pd.to_datetime(Dates['Date'])

Date_first = Dates.groupby(['ID_TA'])['Date'].min().reset_index()
Date_first.columns = ['ID_TA','Date first rew']
df = pd.merge(df, Date_first, on='ID_TA', how='left')

Date_second = Dates.groupby(['ID_TA'])['Date'].max().reset_index()
Date_second.columns = ['ID_TA','Date sec rew']
df = pd.merge(df, Date_second, on='ID_TA', how='left')

df['Period'] = df['Date sec rew'] - df['Date first rew']

a = pd.to_datetime('today')
df['Sec to Now'] = df['Date sec rew'].apply(lambda x: (a - x).days)

df['Period'] = df['Period'].apply(lambda x: x.days)
#raise Exception()
#plt.figure()
#sns.barplot("City", y="Period", data=df, estimator = np.median)
#plt.figure()
#sns.barplot("City", y="First to Now", data=df, estimator = np.mean)

'''Вычисление параметра разницы между ID ресторана и местом в городе'''
df['Rest_id int'] = df['Restaurant_id'].apply(lambda x: int(x[3:]))
df['dif Rank rID'] = df['Rest_id int'] - df['Ranking']
df = df.drop(['Rest_id int'], axis = 1)

#df['len of rew'] = df['Reviews'].apply(lambda x: len(x))

#plt.figure()
#correlation = df.corr()
#sns.heatmap(correlation, annot = True, cmap = 'coolwarm')
#
#plt.figure()
#sns.set_style('whitegrid')
#sns.distplot(df[df['Number of Reviews'] >= 1]['Number of Reviews'], kde = True, norm_hist = True)
#raise Exception()
#s = df[['City','Period','Reviews','Date first rew','Date sec rew']]
#s = df.groupby('Num Cuisine')['Rating'].mean()
#s1 = df['Num Cuisine'].value_counts()

#df['Num Cuisine'] = df['Num Cuisine'].apply(lambda x: 6 if x > 6 else x)

'''Исключение пропусков'''
df['ID Word'] = df.groupby('City')['ID Word'].transform(lambda x: x.fillna(x.mean()))
df['Period'] = df.groupby('City')['Period'].transform(lambda x: x.fillna(x.mean()))
df['Sec to Now'] = df.groupby('City')['Sec to Now'].transform(lambda x: x.fillna(x.median()))

def f(x):
    if pd.isna(x[0]) and x[1] == 0:
        return 1
    elif pd.isna(x[0]):
        return 0
    else:
        return x[0]

df['Number of Reviews'] = df[['Number of Reviews','Period']].apply(lambda x: f(x), axis=1)
df['Number of Reviews otn'] = df.groupby('City')['Number of Reviews'].transform(
        lambda x: x / x.max())
df['Number of Reviews otn'] = df['Number of Reviews otn'].fillna(0)
#df['Number of Reviews'] = df.groupby('City')['Number of Reviews'].transform(lambda x: x.fillna(int(x.mean())))

df['freq sum'] = df.groupby('City')['freq sum'].transform(lambda x: x.fillna(int(x.median())))
df['Price ID'] = df['Price ID'].fillna(2)

#pattern = re.compile('\d+')
#df['URL'] = df['URL_TA'].apply(lambda x: int(pattern.findall(x)[0]))
#s = df[['City','URL_TA','URL']]

#plt.figure()
#correlation = df.corr()
#sns.heatmap(correlation, annot = True, cmap = 'coolwarm')
#plt.figure()
#sns.set_style('whitegrid')
#sns.distplot(df['ID Word'], kde = True, norm_hist = True)


'''Создание признаков dummy variables'''
#'''Метрики кухонь (не уменьшает ошибку)'''
#for index, row in Cuisine_val.iterrows():
#    df[row['Cuisine']] = df['Cuisine Style'].apply(lambda x: 1 if type(x) == str
#      and row['Cuisine'] in x else 0)

'''Метрики городов (немного уменьшает ошибку)'''
for index, row in City_count.iterrows():
    df[row['City']] = df['City'].apply(lambda x: 1 if type(x) == str
      and row['City'] in x else 0)
df = df.drop(['City ID'], axis = 1)

#'''Метрики слов (не уменьшает ошибку)'''
#for i in all_words:
#    df[i] = df['Reviews'].apply(lambda x: 1 if type(x) == str
#      and i in x else 0)
#df = df.drop(['ID Word'], axis = 1)
#raise Exception()    
iskl = ['Restaurant_id', 'Rating','City','Cuisine Style','Price Range',
         'Reviews','URL_TA','ID_TA','Date first rew','Date sec rew','Count rest in city',
         'Rest count','Ranking','freq sum']
# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
X = df.drop(iskl, axis = 1)
y = df['Rating']

# Загружаем специальный инструмент для разбивки:
from sklearn.model_selection import train_test_split

# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования мы будем использовать 25% от исходного датасета.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели

# Создаём модель
regr = RandomForestRegressor(n_estimators=100, n_jobs=-1)

# Обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = regr.predict(X_test)

# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

fi = pd.DataFrame({'feature': list(X.columns),
                   'importance': regr.feature_importances_}).\
                    sort_values('importance', ascending = False)





