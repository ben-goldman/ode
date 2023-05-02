import numpy as np
from math import sin, cos


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
