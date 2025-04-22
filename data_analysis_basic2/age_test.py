# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:04:28 2025

@author: Admin
"""

import csv

f = open('./data/age.csv')
data = csv.reader(f)

'''
우리 동네의 인구 구조를 시각화

1.인구 데이터 파일을 읽어온다.
2. 전체 데이터에서 한 줄씩 반복해서 읽어온다.
3. 우리 동네에 대한 데이터인지 확인
4. 우리 동네일 경우 0세부터 100세 이상까지의 인구수를 순서대로 저장
5. 저장된 연령별 인구수 데이터를 시각화
'''
result = []
name = input("원하는 지역의 이름은?: ")
for row in data:
    if name in row[0]:
        for i in row[3:]:
            result.append(int(i))
            
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rc('font',family='Malgun Gothic')
plt.title(name + '지역의 인구 구조')
plt.plot(result)
plt.show()



plt.bar(range(101), result)
plt.show()

plt.barh(range(101),result)
plt.show()

# 남녀 성별 인구 분포

import csv
f = open('./data/gender.csv')
data = csv.reader(f)

m = []
f = []
     
     
for row in data:
    if '신도림' in row[0]:
        for i in row[3:104]:
            m.append(-int(i))
        
        for i in row[106:]:
            f.append(int(i))
     
     
print(len(m),len(f)) #101 101
     
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5),dpi = 300)
plt.rc('font',family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

plt.title('신도림 지역의 남녀 성별 인구 분포')
plt.barh(range(101),m, label = '남성')
plt.barh(range(101),f, label = '여성')
plt.legend()
plt.show()
     
# 함수처리 GUI 만들어보기
     
     
     
     
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5),dpi = 300)
plt.rc('font',family='Malgun Gothic')

size = [2441, 2312, 1032, 1233]
label = ['A형','B형','AB형','O형']
color = ['darkmagenta','deeppink','hotpink','pink']

plt.axis('equal')
plt.pie(size, labels= label, autopct = "%.1f%",
        colors = color,
        explode = (0, 0.5, 0, 0))
# '%.1f% ' : 소수점 이하 1자리


plt.legend()
plt.show()
     
     
     
     
import csv
f = open('./data/gender.csv')
data = csv.reader(f)

name = input('찾고 싶은 지역의 이름을 알려주세요: ')

size = [] # 남자 / 여자

for row in data:
    if name in row[0]:
        m = 0
        f = 0
        
        
        for i in range(101) :
            m += int(row[i+3])
            f += int(row[i+106])
            
        break
size.append(m)
size.append(f)


    
import matplotlib.pyplot as plt
plt.rc('font',family='Malgun Gothic')
color = ['crimson','darkcyan']
plt.axis('equal')

plt.pie(size, labels = ['남','여'],
        autopct='%.1f%%',
        colors = color,
        startangle = 90)
plt.title(name+' 지역의 남녀 성별 비율')     
plt.show()
     
    
    
import csv
f = open('./data/gender.csv')
data = csv.reader(f)

m = []
f = []
name = input('찾고 싶은 지역의 이름을 알려주세요: ')

     
for row in data:
    if name in row[0]:
        for i in range(3,104):
            m.append(int(row[i]))
            f.append(int(row[i+103]))
        break
    
import matplotlib.pyplot as plt
plt.plot(m, label='Male')
plt.plot(f, label='Female')
plt.legend()
plt.show()    
     



import matplotlib.pyplot as plt
plt.style.use('ggplot')

x = [1,2,3,4]
y = [10,20,30,40]

size = [100,25,200,180]
color = ['red','blue','green','gold']

plt.scatter(x,y,s=size, c = color)
plt.show()     
     
     
     
     
import csv
f = open('./data/gender.csv')
data = csv.reader(f)

m = []
f = []  
     
     
name = input('궁금한 지역(도)을 입력해주세요 : ')
for row in data:
    if name in row[0]:
        for i in range(3,104):
            m.append(int(row[i]))
            f.append(int(row[i+103]))
        break

plt.style.use('ggplot')  
plt.scatter(m, f)
plt.show()
     
     
plt.scatter(m, f, c = range(101), alpha = 0.5, cmap='jet')


plt.colorbar()
plot.plot(range(max(m)),range(max(m)),'g')
plt.show()
     
     
# ----------------------------------------------------
import csv

f = open('./data/gender.csv')
data = csv.reader(f)

m = []
f = []  
size = []
     
name = input('궁금한 지역(도)을 입력해주세요 : ')

import math

for row in data:
    if name in row[0]:
        for i in range(3,104):
            m.append(int(row[i]))
            f.append(int(row[i+103]))
            size.append(math.sqrt(int(row[i]) + int(row[i+103])))
        break
import matplotlib.pyplot as plt

plt.style.use('ggplot') 
plt.rc('font',family='Malgun Gothic')
plt.figure(figsize=(10,5),dpi=300)

plt.title(name + " 지역의 인구 그래프")
plt.scatter(m, f, s = size, alpha = 0.5, cmap='jet')
plt.colorbar()
plt.plot(range(max(m)),range(max(m)),'g')
plt.show()



































     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     