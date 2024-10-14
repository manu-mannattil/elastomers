# -*- coding: utf-8 -*-
"""Cahn-Hilliard solver.

Our phase field model reduces to the Cahn-Hilliard model when
the longitudinal modulus M = 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pfmodel import PhaseFieldModel
from utils import *

L = 2
n = 128
x = np.linspace(-L, L, n)
X, Y = np.meshgrid(x, x)
pfm = PhaseFieldModel(a=-1,
                      b=1,
                      kappa=0.001,
                      phi_s=0.5,
                      phi_c=0.0,
                      M=0.0,
                      d=2,
                      n=n,
                      L=L,
                      energy=True)

fig, ax = plt.subplots()
im = ax.pcolormesh(X, Y, rescale(pfm.psi), cmap="RdBu")
ax.set_aspect("equal")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title(r"Cahn-Hilliard evolution")

def animate(i):
    pfm.evolve()
    im.set_array(rescale(pfm.psi))
    print(f"step = {i}; energy = {pfm.E}")
    return [im]

ani = FuncAnimation(fig, animate, frames=1, interval=1, blit=True)
plt.show()
