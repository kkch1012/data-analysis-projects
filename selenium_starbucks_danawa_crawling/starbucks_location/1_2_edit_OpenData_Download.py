# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 04:22:02 2025

@author: tuesv


1_2 
    ./files/report.txt  ./files/report2.txt
    
    
    서울 열린데이터광장 공공데이터 정리
"""
import pandas as pd
sgg_pop_df = pd.read_csv('./starbucks_location/files/report.txt',sep='\t',header=2)

columns = {
    '기간': 'GIGAN',
    '자치구': 'JACHIGU',
    '계': 'GYE_1',
    '계.1': 'GYE_2',
    '계.2': 'GYE_3',
    '남자': 'NAMJA_1',
    '남자.1': 'NAMJA_2',
    '남자.2': 'NAMJA_3',
    '여자': 'YEOJA_1',
    '여자.1': 'YEOJA_2',
    '여자.2': 'YEOJA_3',
    '세대': 'SEDAE',
    '세대당인구': 'SEDAEDANGINGU',
    '65세이상고령자': 'N_65SEISANGGORYEONGJA'
}
sgg_pop_df.rename(columns=columns,inplace = True)

sgg_pop_df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 26 entries, 0 to 25
Data columns (total 14 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   GIGAN                  26 non-null     object 
 1   JACHIGU                26 non-null     object 
 2   SEDAE                  26 non-null     object 
 3   GYE_1                  26 non-null     object 
 4   NAMJA_1                26 non-null     object 
 5   YEOJA_1                26 non-null     object 
 6   GYE_2                  26 non-null     object 
 7   NAMJA_2                26 non-null     object 
 8   YEOJA_2                26 non-null     object 
 9   GYE_3                  26 non-null     object 
 10  NAMJA_3                26 non-null     object 
 11  YEOJA_3                26 non-null     object 
 12  SEDAEDANGINGU          26 non-null     float64
 13  N_65SEISANGGORYEONGJA  26 non-null     object 
dtypes: float64(1), object(13)
memory usage: 3.0+ KB
'''

# 필요없는 데이터 제거 : 첫 번째 데이터 합계
condition = sgg_pop_df['JACHIGU'] != '합계'
sgg_pop_df_selected = sgg_pop_df[condition]

# 분석에 필요한 컬럼 선택
columns = ['JACHIGU','GYE_1']
sgg_pop_df_final = sgg_pop_df_selected[columns]

sgg_pop_df_final.columns = ['시군구명','주민등록인구']

sgg_pop_df_final.to_excel('./starbucks_location/files/sgg_pop.xlsx',index=False)

# ---------------------여기까지 인구 데이터 저일
#  서울시 동별 사업체 현황 통계 데이터: report2.txt
sgg_biz_df = pd.read_csv('./starbucks_location/files/report2.txt',sep='\t',header=2)

# 시군구동별 사업체 현황 데이터 추출
condition = sgg_biz_df['동'] == '소계'
sgg_biz_df_selected = sgg_biz_df[condition]

# 필요없는 컬럼 제거: '자치구','계','사업체수'
columns =['자치구','계','사업체수']
sgg_biz_df_final = sgg_biz_df_selected[columns]
sgg_biz_df_final.columns =['시군구명','종사자수','사업체수']
# 데이터프레임의 인덱스 초기화
sgg_biz_df_final = sgg_biz_df_final.reset_index(drop=True)
# 엑셀로 저장
sgg_biz_df_final.to_excel('./starbucks_location/files/sgg_biz.xlsx',index=False)







































