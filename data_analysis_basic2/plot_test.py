# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:07:16 2025

@author: Admin
"""

import matplotlib.pyplot as plt

plt.plot([10, 20, 30, 40])
plt.show()

plt.title('plotting')
plt.plot([1,2,3,4],[12,43,25,15])
plt.show()

plt.title("Legend")
plt.plot([10, 20, 30, 40], label='asc')
plt.plot([12,43,25,15],label='desc')
plt.legend()
plt.show()

# 색상: color="알려진 색상
plt.title("color")
plt.plot([10,20,30,40],color='skyblue',label='skyblue')
plt.plot([40,30,20,10],color='pink',label='pink')
plt.legend()
plt.show()

# 선 형태: ls = "" / linestyle = 
plt.title("linestyle")
plt.plot([10,20,30,40], color='r', linestyle ='--', label='dashed')
plt.plot([40,30,20,10],color='g',ls=':',label='dotted')
plt.legend()
plt.show()


# 선 대신 , 모양으로
plt.plot([10,20,30,40],'r-',label='circle')
plt.plot([40,30,20,10],'g^',label='triangle up')
plt.legend()
plt.show()

#그래프 기본 구조 생성 : matplotlib.pyplot.figure()
plt.figure()


# 그래프 기본 구조에 그래프를 그려주기

# 그래프 출력 : matplotlib.pyplot.show()
plt.show()

### Numpy를 이용하여 Dummy 데이터 생성 후, sin() 을 이용하여 시각화
import numpy as np

# Numpy의 범위 ; Numpy.arange(시작, 끝, 사이간격)
t = np.arange(0,10, 0.01)
y = np.sin(t)

## 시각화
# figsize=(10,6) 10:6 비율로 준비
plt.figure(figsize=(10,6))
plt.plot(t,y,lw=3,label='sin') # lw : 선 두께
plt.plot(t, np.cos(t),'r',label='cos') # 'r': 선 모양
plt.grid() # grid 추가
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Sample Graph')
plt.xlim(0,np.pi) # x tick 값 변경
plt.ylim(-1.2, 1.2) # y tick 값 변경
plt.show()


# 다양한 모양

t = np.arange(0, 5, 0.5)

plt.figure(figsize=(10,6))
plt.plot(t, t,'r--')
plt.plot(t, t**2,'bs')
plt.plot(t, t**3,'g^')
plt.show()


# 색상, 선 스타일 변경 두번째 방법

t=[0,1,2,4,5,8,9]
y=[1,4,5,8,9,5,3]

plt.figure(figsize=(10,6))
plt.plot(t, y,color='green',linestyle='dashed',
         marker='o',
         markerfacecolor='blue',
         markersize=20)
plt.show()


#### 산점도 그래프


t = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])

plt.figure(figsize=(10,6))
plt.scatter(t,y , marker='>')
plt.show()

colormap= t
plt.figure(figsize=(10,6))
plt.scatter(t,y , marker='>',
            s = 50,
            c = colormap)
plt.colorbar()
plt.show()






























