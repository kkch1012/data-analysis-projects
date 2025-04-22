from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_pybo():
    return 'Hello Python'

'''
app = Flask(__name__)
FLask 클래스로 만든 객체

플라스크는 app 객체를 사용해 여러 가지 설정을 진행
app 객체를 전역으로 사용
프로젝트 규모가 커질수록 순환 참조(circular import) 오류가
발생할 확률이 높아진다.

순환 참조(circular import)
=> A 모듈이 B 모듈을 참조하고
=> B 모듈이 다시 A 모듈을 참조하는 경우

플라스크 공식 홈페이지에서는
"애플리케이션 팩토리(application factory를 사용하라"
팩토리: 디자인패턴(코드) 
'''