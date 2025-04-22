# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:01:34 2025

@author: Admin
"""

import pandas as pd
import pymysql

from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()
import MySQLdb


from matplotlib import font_manager, rc
import platform

if platform.system() == 'Windows':
    path  = 'c:/Windows/Fonts/malgun.ttf'
    font_name=font_manager.FontProperties(fname = path).get_name()
    rc('font',family=font_name)
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
else:
    print('Check your OS system')
    

host = 'Localhost'
user = 'root'
password ='doitmysql'
db ='test'
charset='utf8'


df_1 = pd.read_csv('./data/2024년 1월 국내노선 여객 이용률.csv')
df_2 = pd.read_csv('./data/2024년 2월 국내노선 여객 이용률.csv')
df_3 = pd.read_csv('./data/2024년 3월 국내노선 여객 이용률.csv')
df_4 = pd.read_csv('./data/2024년 4월 국내노선 여객 이용률.csv')
df_5 = pd.read_csv('./data/2024년 5월 국내노선 여객 이용률.csv')
df_6 = pd.read_csv('./data/2024년 6월 국내노선 여객 이용률.csv')
df_7 = pd.read_csv('./data/2024년 7월 국내노선 여객 이용률.csv')
df_8 = pd.read_csv('./data/2024년 8월 국내노선 여객 이용률.csv')

engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{db}")
conn = engine.connect()




# df를 mysql로

# conn.close()

df_8.isna().sum()

df_3 = df_3.dropna() # 한개의 컬럼이 항공사쪽이 결측치라 제거


'''
Out[27]: 
노선     0
항공사    1
좌석수    0
성인     0
유아     0
여객수    0
이용률    0
dtype: int64
'''


df_4 = df_4.fillna(0) # 유아쪽에 결측치가 있길래 값 0으로 대체

df_1['항공사'].unique()
'''
array(['대한항공', '제주항공', '아시아나', '이스타항공', 
       '티웨이항공', '에어부산', '진에어', '에어서울',
       '에어로케이'], dtype=object)
'''
df_2['항공사'].unique()
'''
array(['ESR', 'KAL', 'AAR', 'JJA', 'JNA', 'ABL', 'TWB',
       'ASV', 'EOK'],
      dtype=object)
'''

airline_dict = { 
    '대한항공': 'KAL',
    '제주항공': 'JJA',
    '아시아나': 'AAR',
    '이스타항공': 'ESR',
    '티웨이항공': 'TWB',
    '에어부산': 'ABL',
    '진에어': 'JNA',
    '에어서울': 'ASV',
    '에어로케이': 'EOK'
} # 1월 데이터만 항공사명이 한글이라 영어로 통일

# 데이터 통일
df_1['항공사'] = df_1['항공사'].replace(airline_dict)
df_1.rename(columns={'이용율': '이용률'}, inplace=True)
df_2.rename(columns={'이용율': '이용률'}, inplace=True)

# 월별 데이터를 하나로 합치기
df_1['월'] = '1월'
df_2['월'] = '2월'
df_3['월'] = '3월'
df_4['월'] = '4월'
df_5['월'] = '5월'
df_6['월'] = '6월'
df_7['월'] = '7월'
df_8['월'] = '8월'


# 모든 데이터프레임을 합침
df_all = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8], ignore_index=True)

df_all.isna().sum()

df_grouped = df_all.groupby(['항공사', '노선', '월'])[['이용률']].sum().reset_index()

df_all.to_sql(name='df_all',con=engine,if_exists='replace',index=False) # sql로 데이터
conn.close()

import seaborn as sns
import matplotlib.pyplot as plt

# 월별로 여객수와 좌석수 합산
df_monthly = df_all.groupby('월')[['여객수', '좌석수']].sum().reset_index()

# 이용률 계산
df_monthly['이용률'] = df_monthly['여객수'] / df_monthly['좌석수']



plt.figure(figsize=(12, 8))
sns.lineplot(data=df_monthly, x='월', y='이용률', markers=True)
plt.title('월별 전체 이용률 변화')
plt.xlabel('월')
plt.ylabel('이용률')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


df_airline_popularity = df_all.groupby('항공사')['여객수'].sum().reset_index()


df_airline_popularity_sorted = df_airline_popularity.sort_values(by='여객수', ascending=False)


most_popular_airline = df_airline_popularity_sorted.iloc[0]



plt.figure(figsize=(12, 10))
sns.barplot(data=df_airline_popularity_sorted, x='항공사', y='여객수')
plt.title('항공사별 여객수 (인기순)')
plt.xlabel('항공사')
plt.ylabel('여객수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


df_airline_popularity = df_all.groupby('항공사')[['여객수','좌석수']].sum().reset_index()

df_airline_popularity['이용률'] = df_airline_popularity['여객수'] / df_airline_popularity['좌석수']
df_airline_popularity_sorted = df_airline_popularity.sort_values(by='이용률', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(data=df_airline_popularity, x='항공사', y='이용률')
plt.title('항공사별 이용률')
plt.xlabel('항공사')
plt.ylabel('여객수/좌석수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df_airline_popularity['남은 좌석'] = df_airline_popularity['좌석수'] - df_airline_popularity['여객수']
df_airline_popularity['남은 좌석 비율'] = df_airline_popularity['남은 좌석'] / df_airline_popularity['좌석수']

plt.figure(figsize=(12, 10))
sns.barplot(data=df_airline_popularity, x='항공사', y='남은 좌석')
plt.title('항공사별 잔여 좌석 수')
plt.xlabel('항공사')
plt.ylabel('잔여 좌석수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
sns.barplot(data=df_airline_popularity, x='항공사', y='남은 좌석 비율')
plt.title('항공사별 잔여 좌석 비율')
plt.xlabel('항공사')
plt.ylabel('잔여 좌석 비율')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


cor = df_all.corr()
cor


df_all_all = df_all.groupby(['항공사', '노선', '월'])[['여객수', '유아', '성인', '좌석수']].sum().reset_index()
month = {
    '1월':1,
    '2월':2,
    '3월':3,
    '4월':4,
    '5월':5,
    '6월':6,
    '7월':7,
    '8월':8}
df_all_all['월'] = df_all_all['월'].replace(month)
cor = df_all_all[['여객수','유아','성인','좌석수','월']].corr()
cor

df_all_new = df_all.groupby('월')[['여객수', '유아', '성인', '좌석수']].sum().reset_index()


plt.figure(figsize=(12, 6))


plt.plot(df_all_new['월'], df_all_new['여객수'], marker='o', linestyle='-', label='여객수')

plt.plot(df_all_new['월'], df_all_new['좌석수'], marker='s', linestyle='--', label='좌석수')


plt.title('월별 여객수 및 좌석수 변화')
plt.xlabel('월')
plt.ylabel('수량')
plt.xticks(df_all_new['월'])
plt.legend() 
plt.grid(True) 

plt.show()

df_all_new['남는 좌석'] = df_all_new['좌석수']-df_all_new['여객수']







