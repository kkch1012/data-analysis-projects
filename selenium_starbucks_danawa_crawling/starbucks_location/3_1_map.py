# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 04:22:02 2025

@author: tuesv

"""
import pandas as pd
import folium
import json

seoul_starbucks = pd.read_excel('./starbucks_location/files/seoul_starbucks_list.xlsx')

starbucks_map = folium.Map(location=[37.573050, 126.979189],
                           zoom_start= 11)
# starbuck_map
for idx in seoul_starbucks.index:
    lat = seoul_starbucks.loc[idx, '위도']
    lng = seoul_starbucks.loc[idx, '경도']
    
    
    folium.CircleMarker(location=[lat,lng],
                        fill = True,
                        fill_color='green',
                        fill_opacity=1,
                        coolor='yellow',
                        weight=1,
                        radius=3).add_to(starbucks_map)
    
    
    
starbucks_map.save('starbucks_map.html')


starbucks_map2 = folium.Map(location=[37.573050, 126.979189],
                           zoom_start= 11)
# starbuck_map
for idx in seoul_starbucks.index:
    lat = seoul_starbucks.loc[idx, '위도']
    lng = seoul_starbucks.loc[idx, '경도']
    
    store_type = seoul_starbucks.loc[idx, '매장타입']
    
    # 매장 타입별 색상 선택을 위한 조건문
    fillColor =''
    if store_type == 'general':
        fillColor = 'green'
        size = 3
    elif store_type == 'reserve':
        fillcolor = 'violet'
        size =5
    elif store_type == 'generalDT':
        fillcolor ='red'
        size =5
    folium.CircleMarker(location=[lat,lng],
                        fill = True,
                        fill_color=fillColor,
                        fill_opacity=1,
                        color=fillColor,
                        weight=1,
                        radius=size).add_to(starbucks_map2)

starbucks_map2.save('starbucks_map2.html')









































































