# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:55:48 2025

@author: Admin
"""

import csv
# 데이터를 읽어온다
# 모든 역에 대해 시간대별 승차 인원과 하차 인원을 누적
# 시간대별 승차 인원과 하차 인원을 시각화
f = open('./data/subwaytime.csv')
data = csv.reader(f)
next(data)
next(data)
name = input(" 원하시는 역명은 무엇인가요: ")


size = []

up = [[] for _ in range(24)]
down = [[] for _ in range(24)]

# 승차 하차 추이


for row in data:
    if name in row[3]:
        for i in range(4,44,2):
            hour = int(header[i].split(':')[0])
            up[hour-4].append(row[hour])
            down[hour-4].append(row[hour+1])
            
            
            
            
'''
정답
'''
            
import csv
# 데이터를 읽어온다
# 모든 역에 대해 시간대별 승차 인원과 하차 인원을 누적
# 시간대별 승차 인원과 하차 인원을 시각화
f = open('./data/subwaytime.csv')
data = csv.reader(f)
next(data)
next(data)
name = input(" 원하시는 역명은 무엇인가요: ")            
            
            
s_in = [0] * 24 

s_out = [0] * 24     

#각 행의 4번 인덱스부터 마지막까지의 데이터는 정수로 변환

'''
for 반복문 대신 map() 함수를 사용해서 데이터를 한꺼번에 정수형을 변환
'''      
        
for row in data:
    if name in row[3]:
        row[4:] = map(int, row[4:])
    
        for i in range(24):
            s_in[i] += row[4 + i*2]
            s_out[i] += row[5 + i*2]


import matplotlib.pyplot as plt

plt.figure(dpi=300)
plt.rc('font',family='Malgun Gothic')

plt.title('지하철 시간대별 승하차 인원 추이')

plt.plot(s_in, label='승차')
plt.plot(s_out, label='하차')
plt.legend()
plt.grid()


plt.xticks(range(24), range(4,28))

plt.show()
























            