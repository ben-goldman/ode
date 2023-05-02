import numpy as np
from tqdm import tqdm


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
