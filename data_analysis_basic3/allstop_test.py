# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:02:38 2025

@author: Admin
"""

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Malgun Gothic'


csv_file = './data/allStoreModified.csv'

myframe = pd.read_csv(csv_file, index_col=0, encoding='utf-8')

print(myframe['brand'].unique())
# ['cheogajip' 'goobne' 'nene' 'pelicana'] 브랜드 값엔 결측치 X

brand_dict = {'cheogajip': ' 처가집', 'goobne':'굽네',
              'kyochon':'교촌','pelicana':'페리카나','nene':'네네'}

mygrouping = myframe.groupby(['brand'])['brand']

charData = mygrouping.count()
charData
# brand
# cheogajip    1204
# goobne       1066
# nene         1125
# pelicana     1098
# Name: brand, dtype: int64

newindex = [brand_dict[idx] for idx in charData.index]

charData.index = newindex
charData
'''
Out[148]: 
 처가집    1204
굽네      1066
네네      1125
페리카나    1098
Name: brand, dtype: int64
'''

mycolor=['r','g','b','m']
plt.figure()
charData.plot(kind='pie',
              legend=False,
              autopct='%1.2f%%',
              colors=mycolor)



filename = 'makeChicken.png'
plt.savefig(filename, dpi=400, bbox_inches='tight')
print(filename + ' 파일이 저장되었습니다')


plt.show()



















































