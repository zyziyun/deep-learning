"""
numerical differentiation(数值微分)：用数值方法近似求解函数的导数的过程

舍入误差（rounding error）。
指因省略小数的精细部分的数值（比如，小数点第 8 位以后的数值）而造成最终的计算结果上的误差。
在 Python 中，舍入误差可表示：np.float32(1e-50)（float32 类型（32 位的浮点数）来表示 1e-50，就会变成 0.0）


“真的导数”对应函数在 x 处的斜率（称为切线）
数值微分含有误差：导数对应的是 (x + h) 和 x 之间的斜率

为了减小这个误差，我们可以计算函数 f 在 (x + h) 和 (x - h) 之间的差分。因为这种计算方法以 x 为中心，计算它左右两边的差分，所以也称为中心差分（而 (x + h) 和 x 之间的差分称为前向差分）
"""

import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 #0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x


x = np.arange(0.0, 20.0, 0.1) # 以0.1为单位，从0到20的数组x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)


# 计算一下函数在 x = 5 和 x = 10 处的导数( f(x) 相对于 x 的变化量，对应函数的斜率 )
print(numerical_diff(function_1, 5)) # 0.1999999999990898
print(numerical_diff(function_1, 10)) # 0.2999999999986347
# 误差非常小

# 画对应的切线
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


tf = tangent_line(function_1, 5)
y2 = tf(x)
y3 = tangent_line(function_1, 10)(x)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()