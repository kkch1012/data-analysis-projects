# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:08:38 2025

@author: Admin
"""
import pandas as pd

uri = "./data/titanic.csv"
df = pd.read_csv(uri, sep='\t')

# 컬럼명을 소문자로
new_columns = ["passengerId",'survived','pclass','name','sex',
               'age','sibsp','parch','ticket','fare','cabin',
               'embarked']

df.columns = new_columns
# 필요한 컬럼만 가져오기
titanic_df = df[['survived','pclass','name','sex',
               'age','sibsp','parch','fare']]
tmp = []
for each in titanic_df['sex']:
    if each=='female':
        tmp.append(0)
    else:
        tmp.append(1)
titanic_df['gender']=tmp
titanic_df.drop(columns='sex',inplace=True)

# 2. 호칭을 숫자형으로 바꾸기

condition = lambda x:x.split(',')[1].split('.')[0].strip()
# name에서 호칭만
titanic_df['title']=titanic_df['name'].map(condition)
# 특수한 호칭은 Special로
Special=['Master','Don','Rev']
for each in Special:
    titanic_df['title']=titanic_df['title'].replace(each,'Special')
# name 열 삭제
titanic_df=titanic_df.drop('name',axis=1)
# 숫자형으로 만드는 함수 만들기
def convert_title(x):
    if x=="Special":
        return 1
    else:
        return 0
    
# special_title 열 만들어서 special 1  나머지는 0    
titanic_df['special_title'] = titanic_df['title'].apply(convert_title)
# 했으면 title 삭제
titanic_df.drop('title',axis=1,inplace=True)


titanic_df['sibpar'] = titanic_df['sibsp'] + titanic_df['parch']
titanic_df.drop(['sibsp','parch'],axis=1,inplace=True)
# 사용하지 않는 열 삭제

titanic_df['avgfare']=titanic_df['fare']/titanic_df['sibpar']
titanic_df['n_family']=titanic_df['sibpar']+1
# n_family 열 만들어 동반자에 1을 더하여 0으로 나누어지지 않도록 함
titanic_df['avgfare']=titanic_df['fare']/titanic_df['n_family']
titanic_df = titanic_df.drop(['fare','sibpar'], axis=1)


titanic_df.rename(columns={'gender':'sex','special_title':'title','avgfare':'fare','n_family':'num_family'},inplace=True)
titanic_df=titanic_df[['survived','pclass','sex','age','title','fare','num_family']]

titanic_df=titanic_df.dropna()
titanic_df
'''
Out[16]: 
     survived  pclass  sex   age  title       fare  num_family
0           0       3    1  22.0      0   3.625000           2
1           1       1    0  38.0      0  35.641650           2
2           1       3    0  26.0      0   7.925000           1
3           1       1    0  35.0      0  26.550000           2
4           0       3    1  35.0      0   8.050000           1
..        ...     ...  ...   ...    ...        ...         ...
150         0       2    1  51.0      1  12.525000           1
151         1       1    0  22.0      0  33.300000           2
152         0       3    1  55.5      0   8.050000           1
153         0       3    1  40.5      0   4.833333           3
155         0       1    1  51.0      0  30.689600           2

[126 rows x 7 columns]
'''

raw = titanic_df
np_raw = raw.values
type(np_raw)
# Out[19]: numpy.ndarray
train=np_raw[:100]
test=np_raw[100:]

y_train=[i[0] for i in train]
X_train=[j[1:] for j in train]
y_test=[i[0] for i in test]
X_test=[j[1:] for j in test]
len(X_train),len(y_train),len(y_test),len(X_test)


#Out[20]: (100, 100, 26, 26)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=5).fit(X_train,y_train)

model.fit(X_train,y_train)
DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=5)
print('Score:{}'.format(model.score(X_train,y_train)))
print('Score:{}'.format(model.score(X_test,y_test)))
'''
Score:0.84
Score:0.8846153846153846
'''

from sklearn.tree import export_graphviz

export_graphviz(
    model,
    out_file="titanic.dot",
    feature_names=['pclass','sex','age','title','fare','num_family'],
    class_names=['0','1'],
    rounded=True,
    filled=True)
'''
feature_names=['pclass','sex','age','title','fare','num_family']
의사결정 트리를 훈련하는데 사용된 컬럼 이름
생성된 그래프의 노드에 표시

class_names=['0','1']: 대상 클래스의 이름
                        트리의 리프 노드에 레이블을 지정하는 데 사용
                        생존(0)은 생존하지 못함, 1은 생존함)
rounded = True: 그래프의 노드 모서리가 둥글게
filled=True : 가가 노드 내에서 다수 클래스를 나타내는 색상으로 노드가 채워진다
'''
import graphviz
with open("titanic.dot") as f:
    dot_graph=f.read()
dot=graphviz.Source(dot_graph)
dot.format='png'
dot.render(filename='titani_tree',directory='image/decision_trees',cleanup=True)
dot


import graphviz
import os

graphviz_path = "C:\\Program Files\\Graphviz\\bin"
os.environ["PATH"] += os.pathsep + graphviz_path

with open("titanic.dot") as f:
    dot_graph=f.read()
dot=graphviz.Source(dot_graph)
dot.format='png'
dot.render(filename='titani_tree',directory='image/decision_trees',cleanup=True)



from sklearn.metrics import confusion_matrix
# 머신 러닝에서 분류 모델의 성능을 평가하는 데 사용

from sklearn.metrics import accuracy_score
# 분류 모델의 정확도를 계산하는 데 사용
y_pred=model.predict(X_test)
print("Test Accuracy is ",accuracy_score(y_test,y_pred)*100)
# Test Accuracy is  88.46153846153845
confusion_matrix(y_test,y_pred)

'''
Out[43]: 
array([[18,  1],
       [ 2,  5]], dtype=int64)

오른쪽 상단 요소(1): FP => 클래스 '0'의 인스턴스 1개를 클래스 '1'로 잘못 분류
왼쪽 하단 요소(2): FP => 클래스 '1'의 두 인스턴스를 클래스 '0'으로 잘못 분류
오른쪽 하단 요소(5): TP => 클래스 '1'의 5개 인스턴스를 정확하게 분류
'''


feature_names =['pclass','sex','age','title','fare','num_family']
Tom=[1,1,33,1,50,4]
Jane=[2, 0, 50, 0, 8, 1]

model.predict_proba([Tom])
# Out[45]: array([[0.64705882, 0.35294118]])
model.predict_proba([Jane])
# Out[46]: array([[0.16666667, 0.83333333]])


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C=10.0,
                             solver='liblinear',
                             random_state=13
                             )
'''
C=10.0 정규화 강도/ 상대적으로 약한 정규화
solver=liblinear : 최적화 문제에서 사용할 알고리즘을 지정
        => 소규모에서 중규모 데이터 세트와 L1 또는 L2 정규화에 적합한 선택
'''

log_reg.fit(X_train,y_train)


pred=log_reg.predict(X_train)
accuracy_score(y_train, pred)
# Out[51]: 0.78

pred=log_reg.predict(X_test)
accuracy_score(y_test, pred)
# Out[54]: 0.8846153846153846
confusion_matrix(y_test,pred)
'''
Out[55]: 
array([[18,  1],
       [ 2,  5]], dtype=int64)
'''

# 표준화한 데이터 이용

import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(data=titanic_df[['pclass','sex','age','title','fare','num_family']])
plt.show()


X=titanic_df

from sklearn.preprocessing import MinMaxScaler, StandardScaler
MMS = MinMaxScaler()
SS=StandardScaler()

SS.fit(X)
MMS.fit(X)

X_ss = SS.transform(X) # 평균이 약 0이고 표준 편차가 약 1인 특성을 생성
X_mms = MMS.transform(X) # 0과 1사이의 범위로 스케일링된 특성을 생성

X_ss_pd = pd.DataFrame(X_ss, columns=X.columns)
X_mms_pd = pd.DataFrame(X_mms, columns=X.columns)

sns.boxplot(data=X_mms_pd[['pclass','sex','age','title','fare','num_family']])
plt.show()

sns.boxplot(data=X_ss_pd[['pclass','sex','age','title','fare','num_family']])
plt.show()

y=raw['survived']
X=raw.drop(['survived'], axis=1)

from sklearn.model_selection import train_test_split

X_Train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
                                                    random_state=13)

import numpy as np
np.unique(y_train, return_counts=True)
# np.unique(y_train): 배열에 존재하는 모든 고유한 값
# return_counts=True : 각 고유 값이 배열에 나타나는 횟수도 반환

X_out = X_mms_pd
X_train, X_test, y_train, y_test = train_test_split(X_out, y,test_size=0.2,random_state=13)


log_reg = LogisticRegression(C=10.0,
                             solver='liblinear',
                             random_state=13
                             )


log_reg.fit(X_train,y_train)


pred=log_reg.predict(X_train)
accuracy_score(y_train, pred)
# Out[70]: 1.0

X_out = X_ss_pd
X_train, X_test, y_train, y_test = train_test_split(X_out, y,test_size=0.2,random_state=13)


log_reg = LogisticRegression(C=10.0,
                             solver='liblinear',
                             random_state=13
                             )


log_reg.fit(X_train,y_train)


pred=log_reg.predict(X_train)
accuracy_score(y_train, pred)

# Out[71]: 1.0

log_reg.coef_
'''
array([[ 4.73666302e+00, -2.53303155e-01, -5.08052874e-01,
        -1.93453169e-01,  3.13171381e-02,  2.68954893e-03,
        -1.70331998e-01]])
'''

train = [[25,100],[52,256],[38,152],[32,140],[25,150]]

x =[i[0] for i in train]
y =[j[1] for j in train]

def mean(x):
    return sum(x) / len(x)

def d_mean(x):
    x_mean = mean(x)
    return [i - x_mean for i in x]

d_mean(x)
'''
Out[73]: 
[-9.399999999999999,
 17.6,
 3.6000000000000014,
 -2.3999999999999986,
 -9.399999999999999]
'''






























