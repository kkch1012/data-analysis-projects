
import numpy as np

lst = [2, 3, 4]
num_array = np.array(lst)
# array([2,3,4])
'''
항목item 또는 원소element
배열 내부의 각 요소ㅗ는 인덱스index라고 불리는 정수들로 참조
넘파이에서 차원은 축 axis
(3, ) : 형상shape
(4, 3, 2)
-> 3개의 튜플 형식으로 표현되는 3차원 배열 내부의
    각 축이 가지는 최대 원소의 개수
'''

# 다차원 배열의 속성들

num_array.shape
# Out[3]: (3,)
num_array.ndim
# Out[4]: 1
num_array.dtype
# Out[5]: dtype('int32')
num_array.itemsize
# Out[6]: 4
num_array.size
# Out[7]: 3


'''
다차원 배열의 사칙연산
-> 사칙연산을 수행할 때
    개별 원소별로 덧셈,뺄셈, 곱셈 나눗셈이 이루어진다는것
'''
a = np.array([10,20,30])
b = np.array([1,2,3])
a+b

'''
넘파이 배열의 데이터 타입을 지정하는 두가지 방법
=> array(리스트,dtype = 타입)
타입: np.int32 /'int32'
'''
a = np.array([1,2,3,4],dtype=np.int32)
a = np.array([1,2,3,4],dtype='int32')

'''
브로드캐스팅

벡터화 연산
'''

a = np.array([10,20,30])
a*10
# 단일 값: 스칼라


# 2차원 배열과 1차원 배열 연산

b = np.array([[10,20,30],[40,50,60]])

c = np.array([2,3,4])
b+c

b.shape
b.ndim

'''
다차원 배열에 초기값 설정 함수
1. zeros(shape) : 모든 값은 0으로
2. ones(shape) : 모든 값은 1로
3. full(shape, 값) : 모든 값을 지정한 값
4. eye(숫자) :숫자만큼 행과 열이 만들어 짐
'''

zero =np.zeros((2,3))


one = np.ones((2,3))


full = np.full((2,3),123)


eye = np.eye(4)
'''
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
'''

'''
연속적인 값을 가지는 다차원 배열의 생성
arange(m,n)

arange(m,n,step)

m : 시작 값
n : 종료 값(-1)
step : 간격
'''
np.arange(0, 10)

np.arange(0, 10, 2)

np.arange(0.0, 1.0, 0.2)

'''
linspace(start, stop, num=구간)

'''

np.linspace(0,10,5)

np.linspace(0,10,4)

'''
(3,) : 1 => []
(1,3) : 2 => [ [ ] ]

'''


'''
다차원 배열의 축과 삽입
insert(어느 배열에, 데이터, 데이터)
'''
a = np.array([1, 3, 4])
np.insert(a, 1, 2)
# array([1, 2, 3, 4])

'''
2차원
insert(어느 배열에, 차원, 데이터, axis = 방향)
방향 : 0
'''

b = np.array([[1,1],[2,2],[3,3]])
np.insert(b,1,4,axis=1)
'''
flip(어느 배열을, axis=방향)
어느 배열을 지정한 방향으로 reverse
'''

c = np.array([[1,2,3],[4,5,6]])
np.flip(c,axis=1)


np.flip(c,axis=0)

# insert() / flip() : 원본 데이터에 영향을 미치지 않는다
# 파이썬 리스트 연산: 리스트와 리스트간의 연산
# 넘파이 다차원 배열 연산: 요소와 요소간의 연산
# 차원이 다른 넘파이 다차원 배열의 연산 수행자,
# 자동으로 브로드 캐스팅이 이루어지고
# 병렬 연산을 한다 <= 벡터화 연산

'''
파이썬의 리스트는 동일하지 않은 자료형
넘파이의 ndarray 객체는 동일한 자료형의 항목들만 저장

넘파이는 대용량의 배열과 행렬 연산을 수행
'''
arr_2d = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9],
                  [0,1,2]])
print(arr_2d[0][0])

print(arr_2d[2])

print(arr_2d[1:][0:2])




a= np.array([10,20,30])
a.max()
a.min()
a.mean()

a.astype(np.float64)

'''
넘파이 다차원 배열의 요소를 1차원으로 변경
flatten()
'''
b = np.array([[1,1], [2,2],[3,3]])
b.flatten()

# 배열의 정렬 : sort()
d = np.array([[35,24,55],[69,19,9],[4,1,11]])
d.sort() # 디폴트 값이 axis = 1

d = np.array([[35,24,55],
              [69,19,9],
              [4,1,11]])
d.sort(axis=0)

# append(다차원 배열, 다차원 배열)
# 첫번째 다차원 배열에 두번째 다차원 배열을 추가
# axis = 1
# 축을 명시하지 않으면 기본 1차원으로 .....
a = np.array([1,2,3])
b = np.array([[4,5,6],[7,8,9]])
np.append(a,b)
# Out[56]: array([1, 2, 3, 4, 5, 6, 7, 8, 9])

np.append([a],b, axis=0)


a = np.array([[1,2],
              [3,4]])
b = np.array([[10,20],
              [30,40]])
np.matmul(a,b)

# reshape()
# 차원을 변경 시켜주는 함수

sh = np.arange(12)
#  array([0,1,2,3,4,5,6,7,8,9,10,11])

r = sh.reshape(3',4)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''



































