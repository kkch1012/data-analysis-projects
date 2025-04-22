# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:21:21 2025

@author: Admin
"""

import pandas as pd
excel_named = ['./files/melon.xlsx',
               './files/bugs.xlsx',
               './files/genie.xlsx']



appended_data = pd.DataFrame()

for name in excel_named:
    pd_data = pd.read_excel(name)
    
    appended_data = pd.concat([appended_data, pd_data]
                              ,ignore_index=True)
# concat[1,2] -> 1,2를 합치는데
# ignore_index = True 인덱스번호는 무시하겠다 

appended_data.info()

appended_data.to_excel('./files/total.xlsx', index=False)

'''
appended_data = appended_data.append(pd_data) < == ERROR
데이터 프레임 append => concat으로 버전이 up되면서 append 사라짐
'''





















