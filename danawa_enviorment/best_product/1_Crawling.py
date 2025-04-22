# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 04:33:53 2025

@author: tuesv
"""
### 다나와 검색 페이지 접속 ###

# selenium으로 다나와 검색 결과 URL에 접속
from selenium import webdriver
driver = webdriver.Chrome()
url = "http://search.danawa.com/dsearch.php?query=무선청소기&tab=main"
driver.get(url)


### 다나와 검색 웹 페이지에서 상품 정보 가져오기 ###

# 웹 페이지의 HTML 정보 가져오기
from bs4 import BeautifulSoup
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# 페이지에 대한 무선청소기 정보 가져오기


# 상품명 정보 가져오기
import pandas as pd
data = pd.read_excel('./best_product/files/danawa_crawling_result.xlsx')
data.info()
# 상품명, 회사명으로 분리


data['상품명'][:10]
'''
Out[8]: 
0                                샤오미 드리미 V10
1                              원더스리빙 다이나킹 Z9
2                          LG전자 코드제로 A9 A978
3    샤오미 SHUNZAO 차량용 무선청소기 2세대 Z1 PRO (해외구매)
4                            델로라 V11 파워 300W
5                                 샤오미 드리미 V9
6                          카렉스 파워스톰 미니 무선청소기
7                        삼성전자 제트 VS20R9078S2
8                       다이슨 V11 220 에어와트 CF+
9                            일렉트로룩스 ZB3302AK
Name: 상품명, dtype: object
'''
# title = 'LG전자 코드제로 A9 A978'
# info = title.split(' ', 1)
# info

company_list = []
product_list = []

for title in data['상품명']:
    title_info = title.split(' ', 1)
    company_name = title_info[0]
    product_name = title_info[1]
    
    company_list.append(company_name)
    product_list.append(product_name)
# 스펙 목록 정보 가져오기
# 데이터를 분석해 필요한 요소만 추출
## 카테고리, 사용시간, 흡입력
# ------------------------테스트 코드-----------------------------
data['스펙 목록'][0]
'''
'핸디/스틱청소기 / 핸디+스틱형 / 무선형 / 전압: 25.2V / 헤파필터 
/ H12급 / 5단계여과 / 흡입력: 140AW / 흡입력: 22000Pa
 / 먼지통용량: 0.5L / 충전시간: 3시간30분 / 사용시간: 1시간
 / 용량: 2500mAh / 브러쉬: 바닥, 솔형, 틈새, 침구, 연장관 / 거치대 
 / 무게: 1.5kg / 색상:화이트 / 소비전력: 450W'
'''
data['스펙 목록'][0].split(' / ')[11]
# 카테고리 = 0 사용시간 = 11 흡입력 = 7,8 but, 회사마다 올린 형식이 달라서
# 인덱스 번호가 다르다 순서가 뒤섞여있다
## '스펙 목록' 에 대한 패턴 분석
'''
카테고리 : 첫 번째
사용시간 : 00분 / 00시간 <== 사용시간
흡입력   : 00AW/ 00PA <== 흡입력
'''

spec_list = data['스펙 목록'][0].split(' / ')

category = spec_list[0]

# 흡입력 / 사용시간
use_time_spec = ''
suction_spec = ''
for spec in spec_list:
    if '사용시간' in spec:
        use_time_spec = spec # 사용시간: 1시간
    elif '흡입력' in spec:
        suction_spec = spec # 흡입력: 22000Pa

use_time_value = use_time_spec.split(' ')[1].strip() #22000pa
suction_value = suction_spec.split(' ')[1].strip() # 1시간
#------------------------------------------------------------
category_list = []
use_time_list = []
suction_list = []

for spec_data in data['스펙 목록']:
    # ' / '기준으로 스펙 분리하기
    spec_list = spec_data.split(' / ')
    category = spec_list[0]
    category_list.append(category)
    
    use_time_value = None # 입력 안한 제품도 있기때문에 초기화
    suction_value = None # 문자열 변수에 None과 ''은 다르다 
    # ''은 값이 있는상태이므로 NaN 처리가 불가능
    
    
    for spec in spec_list:
        if '사용시간' in spec:
            use_time_value = spec.split(' ')[1].strip()
        if '흡입력' in spec:
            suction_value = spec.split(' ')[1].strip()
            
    use_time_list.append(use_time_value)
    suction_list.append(suction_value)    
# 시간 통일, 단위 통일
'''
시간 단어가 있으면
1. "시간" 앞의 숫자를 추출한 뒤 , 60곱하기 => 분
2. "시간" 뒤에 "분" 글자 앞의 숫자를 추출하여 앞의 시간에 더하기

"시간"이라는 단어가 없을 경우 분 글자 앞의 숫자를 추출하여 시간에 더하기

예외 처리:
    
'''
# test 코드
times = ["40분", "4분", "1시간","3시간30분","4시간"]

def convert_time_minute(time):
    try:
        if '시간' in time:
            hour = time.split('시간')[0]
            if '분' in time:
                minute = time.split('시간')[-1].split('분')[0]
            else:
                minute = 0
        else:
            hour = 0
            minute = time.split('분')[0]
        return int(hour)*60+int(minute)
    except:
        return None
# ---------------------------------
# 모델별 사용시간을 분 단위로 통일
new_use_time_list =[]

for elem in use_time_list:
    value = convert_time_minute(elem)
    new_use_time_list.append(value)
# 무선 청소기 흡입력 단위 통일
'''
AW : 진공청소기의 전력량
W : 모터의 소비 전력단위
PA: 흡입력 단위

(1W == 1AW == 100PA)
'''    

def get_suction(value):
    try:
        value = value.upper()
        if "AW" in value or "W" in value:
            result = value.replace("A","").replace("W","")
            result = int(result.replace(",",""))
        elif "PA" in value:
            result = value.replace("PA","")
            result = int(result.replace(",",""))/100
        else:
            result = None
        return result
    except:
        return None
new_suction_list = []

for elem in suction_list:
    value = get_suction(elem)    
    new_suction_list.append(value)
    
pd_data = pd.DataFrame()

pd_data['카테고리'] = category_list
pd_data['회사명'] = company_list
pd_data['제품'] = product_list
pd_data['가격'] = data['가격']
pd_data['사용시간'] = new_use_time_list
pd_data['흡입력'] = new_suction_list
    
# 카테고리 분류 기준 및 데이터 개수 점검
pd_data.info()    
# 핸디/스틱 청소기만 선택
pd_data_final = pd_data[pd_data['카테고리'].isin(['핸디/스틱청소기'])] 
    
# 가성비
pd_data_final['가격'].unique()

pd_data_final.to_excel('./best_product/files/danawa_data_final.xlsx',index=False)




























