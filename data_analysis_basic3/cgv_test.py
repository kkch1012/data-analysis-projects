# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:24:02 2025

@author: Admin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'


csv_file = './data/theater.csv'

myframe = pd.read_csv(csv_file, index_col=0, encoding='utf-8')
myframe
print(myframe['daehan'].unique())

mygrouping = myframe.groupby(['daehan'])

group_columns = ['daehan', 'cgv', 'megabox']

result = mygrouping['15'].agg(['sum', 'mean', 'count'])
result

colors = ['blue', 'orange', 'green']  
metrics = ['sum', 'mean', 'count']  

plt.figure()
result.plot(kind='bar',
            legend=False,
            color=colors)

plt.title('극장별 관객 수')
plt.xlabel('극장')
plt.ylabel('수치')
plt.show()

plt.figure()
result['sum'].plot(kind='pie',
              legend=False,
              autopct='%1.2f%%',
              colors=colors)
plt.show()