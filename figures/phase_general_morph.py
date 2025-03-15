# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import charu

L = 25
n = 512
x = np.linspace(-L, L, n)
X, Y = np.meshgrid(x, x)

for i in range(4):
    psi = np.load(f"../tests/data/general_2d_{i}.npy")
    fig, ax = plt.subplots()
    im = ax.pcolormesh(X, Y, rescale(psi), cmap="RdBu")
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.savefig(f"general_2d_{i}.png", crop=True, optimize=True)
    ax.clear()
