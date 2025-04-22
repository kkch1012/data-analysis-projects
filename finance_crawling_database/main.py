# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 09:41:25 2025

@author: Admin
파이썬 크롤러 제작
    yfinance를 호출 데이터 수집 가공
    MYSQL 서버로 데이터 저장하는 역할
    
작업 순서
1. 파이선 패키지 업로드
2. MYSQL 접속 정보
3. 실제 주식 데이터를 가져 오는 함수 생성. : getStock() <= getCompany()에서 호출
    데이터 가공 및 Mysql에 저장
4. 크롤링 할 기업의 목록을 데이터베이스로 읽어 오는 함수 생성: getcompany
5. 파일을 실행할때 처음 실행되는 코드
if __name__ == '__main__'
"""

from datetime import datetime, timedelta

import pymysql
import yfinance as yf

### 2. MYSQL 접속 정보 ###
hostName = 'localhost'
userName = 'root'
password = 'doitmysql'
dbName = 'us_stock'

mysql_conn = pymysql.connect(host = hostName,
                             user = userName,
                             password = password,
                             db = dbName)

### 3. 실제 주식 데이터를 가져오는 함수 설정: getstock(종목코드,시작날짜,종료날짜) ###
def getStock(_symbol, _start_date, _end_date):
    mysql_cur = mysql_conn.cursor()
    
    mysql_cur.execute("delete from us_stock.stock where date >= %s and date <= %s and symbol = %s", (_start_date, _end_date, _symbol))
    # 같은 데이터가 있으면 삭제 
    mysql_conn.commit()
    
    try:
        stock_price = yf.download(_symbol, start=_start_date, end=_end_date)
        print(stock_price)
        
        for index,row in stock_price.iterrows():
            _date = index.strftime("%Y-%m-%d")
            _open = float(row["Open"])
            _high = float(row["High"])
            _low = float(row['Low'])
            _close = float(row['Close'])
            _adj_close = float(101309500)
            _volume = float(row['Volume'])
            
            mysql_cur.execute("insert into us_stock.stock (date, symbol, open, high, low, close, adj_close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", 
                              (_date, _symbol, _open, _high, _low, _close, _adj_close, _volume))
        mysql_conn.commit()
        
        
        mysql_cur.execute("update us_stock.nasdaq_company set open = %s, high = %s, low = %s, close = %s, adj_close = %s, volume = %s, last_crawel_date_stock = %s where symbol = %s", 
                          (_open, _high, _low, _close, _adj_close, _volume, _date, _symbol))
        mysql_conn.commit()
        
    except Exception as e:
        print("error for getStock() : " + str(e))
        mysql_conn.commit()
        mysql_conn.close()
        
        return {'error for getStock() ': str(e)}

### 4. 크롤링 할 기업의 목록을 데이터베이스로 읽어 오는 함수 생성 : getCompany() ###

def getCompany():
    mysql_cur = mysql_conn.cursor()
    
    today = datetime.today() + timedelta(days=1)
    
    try:
        mysql_cur.execute("select symbol, company_name, ipo_year, last_crawel_date_stock from us_stock.nasdaq_company where is_delete is null;")
        results = mysql_cur.fetchall()
        print(results)
        
        for row in results:
            _symbol = row[0]
            _company_name = row[1]
            
            if row[2] is None or row[2] == 0:
                _ipo_year ='1970'
            else:
                _ipo_year = row[2]
                
            if row[3] is None:
                _last_crawel_data_stock = str(_ipo_year) + '-01-01'
            else:
                _last_crawel_data_stock = row[3]
            
            print(_symbol)
            if "." in _symbol:
                print(_symbol)
            else:
                if "/" in _symbol:
                    print(_symbol)
                else:
                    print(_last_crawel_data_stock)
                    getStock(_symbol,_last_crawel_data_stock,today.strftime("%Y-%m-%d"))
                    
    except Exception as e:
        print("error for getStock() : " + str(e))
        mysql_conn.commit()
        mysql_conn.close()
        
        return {'error for getStock() ': str(e)}

### 5. 파일을 실행할때 처음 실행되는 코드 ###
if __name__ == '__main__':
    getCompany()







































