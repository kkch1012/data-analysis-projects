# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:19:57 2025

@author: Admin
"""


'''
csv.writer() : CSV 파일에 데이터를 저장하는 함수
1. csv 모듈
2. csv 파일을 open()함수로 열고 저장 (f변수)
3. f 변수를 reader()에 전달 => csv reader 객체 
'''


import csv
f = open('./data/seoul.csv','r',encoding='cp949')
# 'r' 읽기 전용
# encoing = 'cp949' 한글 패치

data = csv.reader(f, delimiter = ',')
# delimiter =','
# csv 파일: , / tap
# CSV 파일 데이터를 콤마를 기준으로 분리해서 저장하라는 의미

print(data)
# <_csv.reader object at 0x000002AF995476A0>
f.close()



f = open('./data/seoul.csv','r',encoding='cp949')

data = csv.reader(f, delimiter = ',')
seoul_list = []
for row in data:
    seoul_list.append(row)
    # print(row)
f.close()





f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)
'''
next() : 실행할 때마다 한 행씩 반환하면서 
        데이터를 패스..
'''
f.close()






f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)
seoul_list_noheader=[]
for row in data:
    seoul_list_noheader.append(row)

f.close()













f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)
for row in data:
    row[-1] = float(row[-1]) # 최고 기온을 실수로 변형
    print(row)
f.close()
# 이 결과 마지막에 에러가 걸림
# Cell In[37], line 5
#     row[-1] = float(row[-1]) # 최고 기온을 실수로 변형

# ValueError: could not convert string to float: ''

f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)
for row in data:
    if row[-1] == '':
        row[-1] = -999
    row[-1] = float(row[-1])
    
    # 최고 기온 : max_temp = -999
    if max_temp < row[-1]:
        max_date = row[0]
        max_temp = row[-1]
f.close()


print('기상 관측 이래 서울의 최고 기온이 가장 높았던 날은',max_data+'로, ',max_temp,'도 있습니다.')


f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)

result=[]

for row in data:
    if row[-1] != '':
        result.append(float(row[-1]))
print(len(result))

import matplotlib.pyplot as plt

# plt.figure(figsize=(10,6))
# plt.plot(result,'r')
# plt.show()

plt.figure(figsize=(10,2), dpi=300)
plt.plot(result,'r')
plt.show()

# 날짜데이터 추출
'''
'1908-02-03'
split('-')
['1908','02','03']
'''
result = []

for row in data:
     # 최고 기온 값이 존재한다면
    if row[-1] != '':
        
        # 8월에 해당하는 값이라면
        if row[0].split('-')[1] == '08':
            result.append(float(row[-1]))
plt.figure(dpi=300)
plt.plot(result,'hotpink')
plt.show()



import csv



f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)

high = []
low = []
for row in data:
    if row[-1] != '' and row[-2] != '':
        
        date_split = row[0].split('-')
        if date_split[1] == '02' and date_split[2] == '14':
            high.append(float(row[-1]))
            low.append(float(row[-2]))
            
import matplotlib.pyplot as plt
plt.figure(dpi = 300)            
plt.plot(high, 'hotpink')
plt.plot(low,'skyblue')
plt.show()


# 1983년 이후 데이터만 추출해서 매년 생일의 최고/최저 기온 데이터를 시각화

import csv



f = open('./data/seoul.csv')
data = csv.reader(f)
header = next(data)

high = []
low = []
for row in data:
    if row[-1] != '' and row[-2] != '':
        
        date_split = row[0].split('-')
        
        if 1983 <= int(date_split[0]):
            if date_split[1] == '02' and date_split[2] == '14':
                # 5. 최고 기온을 high 리스트에 저장
                high.append(float(row[-1]))
                # 6  최저 기온을 low 리스트에 저장
                low.append(float(row[-2]))
                
import matplotlib.pyplot as plt

#맑은 고딕을 기본 글꼴로 설정
plt.rc('font', family ='Malgun Gothic')
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

plt.figure(dpi = 300)            
plt.title("내 생일의 기온 변화 그래프")
plt.plot(high, 'hotpink',label='high') # label = : 범례를 위한 라벨
plt.plot(low,'skyblue',label='low')

plt.legend() # 범례 표시
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    