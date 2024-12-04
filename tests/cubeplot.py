# -*- coding: utf-8 -*-

"""Test plot a 3D scalar field."""

import numpy as np
import matplotlib.pyplot as plt
from utils import *

L = 1
n = 64

# Test data is a Gaussian centered at (x0, y0, z0).
x0, y0, z0 = 1.0, 0.0, 1.0
x = np.linspace(-L, L, n)
X, Y, Z = np.meshgrid(x, x, x)
phi = np.exp(-((X - x0) * (X - x0) + (Y - y0) * (Y - y0) + (Z - z0) * (Z - z0)))
phi += np.exp(-10 * (X * X + (Y + 1) * (Y + 1) + (Z - 0.5) * (Z - 0.5)))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set(xlabel="x", ylabel="y", zlabel="z", title="Meshgrid-like array")
cubeplot3d(ax, phi, rcount=32, visible_only=True, edge_style={"linestyle": "dashed", "color": "white"})
plt.show()

# Manually indexed data.
phi = np.empty(n**3)
for i in range(n):
    for j in range(n):
        for k in range(n):
            p = n * n * i + n * j + k
            x = -L + 2 * L / (n - 1) * k
            y = -L + 2 * L / (n - 1) * j
            z = -L + 2 * L / (n - 1) * i
            phi[p] = np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2))
            phi[p] += np.exp(-10 * (x * x + (y + 1) * (y + 1) + (z - 0.5) * (z - 0.5)))
phi = phi.reshape((n, n, n))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set(xlabel="x", ylabel="y", zlabel="z", title="Manually indexed array")
cubeplot3d(ax, phi, manual_index=True, rcount=32)
plt.show()
