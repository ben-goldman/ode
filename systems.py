from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt

k = 10000
l = 10
m = 1
g = 10
d = 0.1


def exp(t, y):
    return -10*y


def tetherball(t, y):
    r = y[0]
    r_ = y[1]
    th = y[2]
    th_ = y[3]
    fs = 0
    if r > l:
        fs = k * (l - r)

    rr = r*th_**2 + g*cos(th) + fs/m - (d/m) * r_
    # rr = 0
    thh = 1/r * (-g*sin(th) - 2*r_*th_ - (d/m) * th_ * r)
    return np.array([r_, rr, th_, thh])


def double_pendulum(t, y):
    th1 = y[0]
    th2 = y[1]
    p1 = y[2]
    p2 = y[3]
    th1_ = (6/m*l**2) * (2*p1 - 3*cos(th1 - th2) * p2)/(16 - 9*cos(th1 - th2)**2)
    th2_ = (6/m*l**2) * (8*p2 - 3*cos(th1 - th2) * p1)/(16 - 9*cos(th1 - th2)**2)
    p1_ = -1/2*m*l**2 * (th1_*th2_*sin(th1-th2) + 3*(g/l)*sin(th1))
    p2_ = -1/2*m*l**2 * (-th1_*th2_*sin(th1-th2) + (g/l)*sin(th2))

    return np.array([th1_, th2_, p1_, p2_])


def simple_pendulum(t, y):
    return np.array([y[1], -sin(y[0])])


def lorenz_attractor(t, s):
    Pr = 10
    b = 8/3
    r = 28
    X = s[0]
    Y = s[1]
    Z = s[2]
    X_dot = Pr*(Y - X)
    Y_dot = -X*Z + r*X - Y
    Z_dot = X*Y - b*Z

    return np.array([X_dot, Y_dot, Z_dot])
