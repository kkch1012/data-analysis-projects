# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:27:43 2025

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib import font_manager,rc
import platform

if platform.system() == 'Windows':
    path  = 'c:/Windows/Fonts/malgun.ttf'
    font_name=font_manager.FontProperties(fname = path).get_name()
    rc('font',family=font_name)
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
else:
    print('Check your OS system')
    
df_content = pd.read_excel('./data_ana/happy_data/context.xlsx')
df_economy = pd.read_excel('./data_ana/happy_data/economy.xlsx')
df_education = pd.read_excel('./data_ana/happy_data/education.xlsx')
df_environment = pd.read_excel('./data_ana/happy_data/environment.xlsx')
df_health = pd.read_excel('./data_ana/happy_data/health.xlsx')
df_recreation = pd.read_excel('./data_ana/happy_data/recreation.xlsx')
df_relation = pd.read_excel('./data_ana/happy_data/relation.xlsx')
df_safe = pd.read_excel('./data_ana/happy_data/safe.xlsx')

df_list = [df_content, df_economy, df_education, df_environment, 
           df_health, df_recreation, df_relation, df_safe]

cols_to_drop = ['시도', '구군', 'No']

def region(data_x,data_y):
    thing = [ f'{x} {y}'for x, y in zip(data_x,data_y)]
    return thing

for df in df_list:
    df['지역'] = region(df['시도'], df['구군'])
    
for df in df_list:
    df.drop(columns=cols_to_drop, inplace=True)

df_content.columns    
df_content = df_content[['지역', '삶의 만족도']]

df_merged = pd.merge(df_content, df_economy,on='지역')
df_list2 = [df_education, df_environment, 
           df_health, df_recreation, df_relation, df_safe]

df_list3 = [df_economy, df_education, df_environment, 
           df_health, df_recreation, df_relation, df_safe]

for df in df_list3:
    df.drop(columns='평균',inplace=True)
for df in df_list2:
    df_merged = pd.merge(df_merged,df,on='지역')
    
df_merged.info()
df_merged.isna().sum()        
df_merged = df_merged.fillna(method="pad")    

df_merged.columns

df_merged.dtypes

df_merged.set_index('지역', inplace=True)
corr = df_merged.corr()
    
    
plt.figure(figsize=(15,10))
sns.heatmap(data=corr,annot=True,fmt='.2f',cmap='hot')
plt.show()  

# 상관행렬에서 대각선 값(자기 자신과의 상관관계)을 NaN으로 설정
np.fill_diagonal(corr_abs.values, np.nan)

# 상관계수를 절대값 기준으로 정렬하고, 상위 10개를 추출하되 중복된 쌍을 제거
corr_top10 = corr_abs.stack().sort_values(ascending=False).drop_duplicates().head(10)

# 결과 출력
print(corr_top10)


