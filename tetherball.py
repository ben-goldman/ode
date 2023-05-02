from funcs import calc,  runge, tetherball, polar_rect, mapp
import numpy as np
import matplotlib.pyplot as plt
# from math import pi


t = 100
n = 1000000

ps_r = calc(runge, tetherball, [10, 0, 0, 2.55], n, t)

r = ps_r[:, 0]
r_ = ps_r[:, 1]
th = ps_r[:, 2]
th_ = ps_r[:, 3]

xs, ys = mapp(polar_rect, r, th)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), height_ratios=(2, 1))
ax1.plot(xs, ys)
ax1.set_xlim(-12, 12)
ax1.set_ylim(-12, 12)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")

ax2.plot(np.linspace(0, t, n), r, label=r"$r(t)$")
ax2.plot(np.linspace(0, t, n), r_, label=r"$r'(t)$")
ax2.plot(np.linspace(0, t, n), th, label=r"$\theta(t)$")
ax2.plot(np.linspace(0, t, n), th_, label=r"$\theta'(t)$")
ax2.set_xlabel("$t$")
ax2.legend()

# plt.savefig("fig.pdf")

plt.show()
