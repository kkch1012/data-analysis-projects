# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:01:59 2025

@author: Admin
"""

'''
히스토그램 : hist()
자료의 분포 상태를 직사각형 모양의 막대 그래프
=> 데이터 빈도에 따라 막대 그래프 높이가 결정됨

'''
import matplotlib.pyplot as plt

hist_data = [1,1,2,3,4,5,6,6,7,8,10]

plt.hist(hist_data)
plt.show()
    

import csv

f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)    
    
    
result = []

for row in data:
     # 최고 기온 값이 존재한다면
    if row[-1] != '':
        result.append(float(row[-1]))

plt.hist(result,bins=100,color='r')
# bins = : 구간은 100개
plt.show()

    
    
    
import csv

f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)  
aug = []

for row in data:
    month = row[0].split('-')[1]
    
    if row[-1] != '':
        if month == '08':
            aug.append(float(row[-1]))
    
plt.hist(aug,bins=100,color='r')
plt.show()
    
    
import csv

f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)  
aug = []
jan = []

for row in data:
    month = row[0].split('-')[1]
    
    if row[-1] != '':
        if month == '08':
            aug.append(float(row[-1]))
        if month == '01':
            jan.append(float(row[-1]))
    
plt.hist(aug,bins=100,color='r',label='AUG')
plt.hist(jan,bins=100,color='g',label="JAN")
plt.show()
    


'''
최댓값, 최솟값, 상위 1/4 , 2/4 ,3/4에 위치한 값을 보여주는 그래프
'''
import matplotlib.pyplot as plt
import random

result= []
for i in range(13):
    result.append(random.randint(1,1000))

print(sorted(result))

plt.boxplot(result)
plt.show()



import csv

f = open('./data/seoul.csv')
data = csv.reader(f)
next(data)  
result = []

for row in data:
    if row[-1] != '':
        result.append(float(row[-1]))

import matplotlib.pyplot as plt
plt.boxplot(result)
plt.show()






import csv

f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)  
aug = []
jan = []

for row in data:
    month = row[0].split('-')[1]
    
    if row[-1] != '':
        if month == '08':
            aug.append(float(row[-1]))
        if month == '01':
            jan.append(float(row[-1]))


import matplotlib.pyplot as plt
plt.boxplot(aug)
plt.boxplot(jan)
plt.show()

'''
이상치(outlier)
다른 수치에 비해 너무 크거나 작은 값을 자동으로 나타낸 것
'''



'''
1. 데이터를 월별로 분류해 지정
2. 월별 데이터를 상자 그림
'''

import csv

f = open('./data/seoul.csv')
data = csv.reader(f)
next(data)  

month = [ [],[],[],[],[],[],[],[],[],[],[],[]]

for row in data:
    if row[-1] != '':
        month[int(row[0].split('-')[1])-1].append(float(row[-1]))

'''
날짜 row[0] 은 2017-01-22 
int(row[0].split('-')[1] => 1)
위에서 -1 = 0 => 
'''

import matplotlib.pyplot as plt
plt.boxplot(month)
plt.show()








    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    