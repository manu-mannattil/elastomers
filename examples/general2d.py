# -*- coding: utf-8 -*-
"""General equilibrium profiles in 2D.

This program produces equilibrium profiles to check the validity of the
general phase diagram in Fig. S2.
"""

import numpy as np
from pfmodel import PhaseFieldModel
from utils import *

n = 512
L = 50

T = -2.5
psi_0_list = [0, 0.15624191263393108, 0.2822829691326263, 0.43887768342537176]

for i, psi_0 in enumerate(psi_0_list):
    pfm = PhaseFieldModel(T=T,
                          phi_0=psi_0,
                          T_c=0,
                          a=1,
                          b=1,
                          M=np.e,
                          phi_c=0,
                          kappa=1,
                          h=1,
                          L=L,
                          n=n,
                          dt=1,
                          disorder=1.5)
    for j in range(1000):
        print(i, j)
        pfm.evolve()
    np.save(f"data/general_2d_{i}.npy", pfm.psi)
