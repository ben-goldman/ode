from math import sin, pi, cos
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
})

m = 1
l = 1
g = 1


def exp(t, y):

    return -10*y

def tetherball(t, y):
    k = 10000
    l = 10
    m = 1
    g = 10
    d = 0.1
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


# Euler's formula solver (1st order)
def euler(fn, y_0, h, n):
    out = np.zeros((n, len(y_0)))
    for i in tqdm(range(n)):
        y = out[i-1]
        if i == 0:
            out[0] = y_0
        else:
            t = h*i
            y = y + fn(t, y) * h
            out[i] = y
    return out


# 4th order Runge-Kutta solver
def runge(fn, y_0, h, n):
    out = np.zeros((n, len(y_0)))
    out[0] = y_0
    for i in tqdm(range(n)):
        y = out[i-1]
        if i == 0:
            out[0] = y_0
        else:
            t = h*i
            q1 = fn(t, y)
            q2 = fn(t + h/2, y + q1 * h/2)
            q3 = fn(t + h/2, y + q2 * h/2)
            q4 = fn(t + h/2, y + q3 * h/2)
            y = y + h/6 * (q1 + 2 * q2 + 2 * q3 + q4)
            out[i] = y
    return out


def adams_b2(fn, y_0, h, n):
    y_0 = np.array(y_0)
    out = np.zeros((n, len(y_0)))
    out[0] = y_0
    out[1] = y_0
    this = y_0
    for i in tqdm(range(n)):
        if i < 2:
            continue
        else:
            that = fn(h*(i-1), out[i-1])
            out[i] = out[i-1] + h*((3/2) * this -
                                   (1/2) * that)
            this = that
    return out


def adams_b4(fn, y_0, h, n):
    out = np.zeros((n, len(y_0)))
    out[0:4] = runge(fn, y_0, h, 4)
    a0 = fn(0, out[0])
    a1 = fn(1*h, out[1])
    a2 = fn(2*h, out[2])
    a3 = fn(3*h, out[3])
    for i in range(n):
        if i < 4:
            continue
        else:
            out[i] = out[i-1]
            + (h/24)*(55*a3 - 59*a2 + 37*a1 - 9*a0)
            a0 = a1
            a1 = a2
            a2 = a3
            a3 = fn(i*h, out[i])
    return out


def calc(solver, fn, y_0, n, t):
    h = t/n
    return solver(fn, y_0, h, n)


def polar_rect(r, th):
    x = r * sin(th)
    y = r * -cos(th)
    return (x, y)


def mapp(fn, s1, s2):
    o1 = np.zeros(len(s1))
    o2 = np.zeros(len(s2))
    for i in range(len(s1)):
        o1[i], o2[i] = fn(s1[i], s2[i])

    return (o1, o2)
