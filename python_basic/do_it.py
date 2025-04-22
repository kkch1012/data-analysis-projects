# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:07:05 2025

@author: Admin
"""

import game
game.sound.echo.echo_test()
# AttributeError: module 'game' 
# has no attribute 'sound'

import game.sound.echo.echo_test
# 오류 ModuleNotFoundError: No module named 
# 'game.sound.echo.echo_test'; 'game.sound.echo' 
# is not a package

import game.sound.echo
game.sound.echo.echo_test()

from game.sound.echo import echo_test

import game.sound.echo as ec
ec.echo_test()

import game
print(game.VERSION)
game.print_version_info()

import game
game.render_test()

from game.sound import *
echo.echo_test()




f = open('new.txt','r')

# FileNotFoundError: [Errno 2] No such file or directory: 'new.txt'

try:
    f = open('new.txt','r')
finally:
    f.close()
    
    
    
try:
    f = open('new.txt','r')
except:
    print('file not found........')
    
    
try:
    a =[1,2,3,4]
    print(a[3])
    4/0
except ZeroDivisionError as e:
    print(e)
except IndexError as e:
    print(e)
    
    # ------------------------------------
    
try:
    a=[1,2]
    print(a[3])
    4/0
except:
    print("오류가 발생했습니다.")
    
    
try:
    age = int(input("나이를 입력하세요: "))
except:
    print("정수를 입력하세요~~.")
else:
    if age<=18:
        print("미성년자는 출입금지입니다.")
    else:
        print("환영합니다.")
    
 #----------------------------------------------

class Bird:
    def fly(self):
        raise NotImplementedError
        print("fly")
        # raise NotImplementedError => 상속받는 Eagle은 재정의를 해야된다.
class Eagle(Bird):
    def fly(self):
        print("very fast")

eagle = Eagle()
eagle.fly()
    
# ======================================
class MyError(Exception):
    def __str__(self):
        return "허용되지 않는 별명입니다."

def say_nick(nick):
    if nick == '바보':
        raise MyError()
    print("당신의 별명은",nick,"입니다")
    
try:
    nick = input("별명이 무엇입니까? ")
    say_nick(nick)
except MyError as e:
    print(e)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    