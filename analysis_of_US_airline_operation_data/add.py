# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:34:51 2025

@author: Admin
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df= pd.read_csv('./data_2/2007.csv')
data = [
        'ActualElapsedTime','CRSElapsedTime','WeatherDelay'
        ]

df_last = df[data]
print(len(df[df['WeatherDelay'] == 0]))
df_real = df_last.dropna()
df_real['Time'] = df_real['CRSElapsedTime']-df_real['ActualElapsedTime']

# Calculate the correlation matrix
corr_matrix = df_real[['Time', 'WeatherDelay']].corr()

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap between Time and WeatherDelay')
plt.show()