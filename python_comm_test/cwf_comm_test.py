#!/usr/bin/env python
from turtle import *

#画一个三角形
'''
forward(100)
left(120)
forward(100)
left(120)
forward(100)
'''

#repr:不对字符串中的特殊字符做转意处理，会尽量保持字符串的原样
'''
print(repr('rfqwrca\n\r\t'))
'''

#print中的特殊开头字母f:可以直接在字符串中使用{变量}输出变量的值
'''
cwf_var = [1,2,3]
print(f"this is a var:{cwf_var}\n")
'''

#索引
'''
cwf_str='greeting'
print(cwf_str[-1])
'''

#向下取整
'''
print(6//4)
'''

#del语句
'''
cwf_222=[1,2,3,4,5,6,7,8,9,0]
del cwf_222[8]
print(cwf_222)
'''

#给切片赋值
'''
cwf_333=list('cwfabcdef')
cwf_333[7:] = list('012345676')#超过范围则会增加list的长度
print(cwf_333)
'''

#extend和拼接的差别
'''
cwf_444 = [1,2,3]
cwf_555 = [4,5,6]
print('+:{0}\n  {1}'.format(cwf_444+cwf_555, cwf_444))#返回一个新的list,对原list无影响
cwf_444.extend(cwf_555)
print('extend:{0}'.format(cwf_444))#会改变调用extend的list
'''

#多态的测试
'''
class a0():
    def __init__(self):
        pass

    def func_a(self):
        self.func_b()
    
class a1(a0):
    def __init__(self):
        super().__init__()
    
    def func_b(self):
        print("a1 func_b")

class a2(a1):
    def __init__(self):
        super().__init__()
    
    def func_b(self):
        print("a2 func_b")


cwf_a2 = a2()
cwf_a2.func_a()#print a2 func_b

cwf_a1 = a1()
cwf_a1.func_a()#print a1 func_b
'''


#list的切片是 deep copy
'''
cwf1 = [1,2,3,4,5]
cwf2 = cwf1[:]
cwf3 = cwf1[0:2]
cwf2[0] = 99
cwf3[1] = 98
print("cwf1:{0}\n".format(cwf1))
print("cwf2:{0}\n".format(cwf2)) 
print("cwf3:{0}\n".format(cwf3)) 
'''


#等待一个输入来结束进程，以方便查看输出
exit_in = input()