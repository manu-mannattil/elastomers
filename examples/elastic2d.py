# -*- coding: utf-8 -*-
"""Elastomer microphase separation in 2D."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from params import parameters
from pfmodel import PhaseFieldModel
from utils import *

L = kwargs["L"]
n = 256

Y = 800
kwargs = parameters(Y, kwargs=True)
phi_0 = 0.5
T = 20
pfm = PhaseFieldModel(T=T, phi_0=phi_0, n=n, **kwargs)

x = np.linspace(-L, L, n)
X, Y = np.meshgrid(x, x)
fig, ax = plt.subplots()
im = ax.pcolormesh(X, Y, rescale(pfm.psi), cmap="RdBu")
ax.set_aspect("equal")

def animate(i):
    pfm.evolve()
    im.set_array(rescale(pfm.psi))
    return [im]

anim = FuncAnimation(fig, animate, frames=1, interval=1, blit=True)
plt.show()
