# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 04:22:02 2025

@author: tuesv

2-1 address.py
seoul_starbucks_list.xlsx - 스타벅스 리스트
sgg_pop.xlsx - 지역별 인구
sgg_blz.xlsx - 지역별 사업체수
"""

import pandas as pd

seoul_starbucks = pd.read_excel('./starbucks_location/files/seoul_starbucks_list.xlsx',header=0)

# 스타벅스 주소 정봅에서 시군구명 추출
sgg_names = []

# 서울특별시 강남구 언주로 425 (역삼동) 이런식이므로 split으로 쪼갠다
for address in seoul_starbucks['주소']:
    sgg =address.split()[1]
    sgg_names.append(sgg)

seoul_starbucks['시군구명'] = sgg_names

# 엑셀로 저장
seoul_starbucks.to_excel('./starbucks_location/files/seoul_starbucks_list.xlsx', index=False)




































































