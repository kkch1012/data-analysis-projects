# 필수 라이브러리 설치 먼저!
# pip install selenium pandas webdriver-manager

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# 1. 드라이버 자동 설치 및 브라우저 실행
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')  # 창 최대화
# options.add_argument('--headless')  # 창 없이 실행하려면 주석 해제
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 2. 크롤링 대상 URL (카테고리 지정된 사람인 공공기관 채용공고 목록)
url = "https://www.saramin.co.kr/zf_user/jobs/public/list?cat_kewd=84%2C86%2C87%2C92&company_cd=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7%2C9%2C10&panel_type=domestic&search_optional_item=n&search_done=y&panel_count=y&preview=y"
driver.get(url)

# 3. 요소 로딩 대기
wait = WebDriverWait(driver, 10)
companies = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.list_item .col.company_nm a.str_tit, .list_item .col.company_nm span.str_tit')))
contents = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.list_body .col.notification_info .job_tit .str_tit')))
urls = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.list_body .col.notification_info .job_tit a.str_tit')))
dates = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.list_body .col.support_info .support_detail .date')))

# 4. 데이터 수집 및 정리
data = []
for i in range(min(len(companies), len(contents), len(urls), len(dates))):
    company = companies[i].text.strip()
    content = contents[i].text.strip()
    url_link = urls[i].get_attribute('href').strip()
    deadline = dates[i].text.strip()

    data.append({
        '회사명': company,
        '공고 내용': content,
        '채용공고 URL': url_link,
        '마감기한': deadline
    })

# 5. 브라우저 종료
driver.quit()

# 6. pandas DataFrame으로 변환
df = pd.DataFrame(data)

# 7. 출력 및 저장
print(df.head())  # 콘솔에 상위 5개 공고 출력
df.to_csv("saramin_jobs.csv", index=False, encoding='utf-8-sig')  # CSV로 저장
print("CSV 저장 완료: saramin_jobs.csv")
