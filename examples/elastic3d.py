# -*- coding: utf-8 -*-
"""Elastomer microphase separation in 3D."""

import numpy as np
from pfmodel import PhaseFieldModel
from params import parameters

Y = 800
kwargs = parameters(Y, kwargs=True)
phi_0 = 0.2
T = 20

L = kwargs["L"]
n = 256

pfm = PhaseFieldModel(T=T, phi_0=phi_0, n=n, d=3, **kwargs)

for i in range(5000):
    print(i)
    pfm.evolve()

np.save(f"data/Y_{Y}_phi_{phi_0}.npy", pfm.psi)
