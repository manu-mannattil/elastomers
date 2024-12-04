# -*- coding: utf-8 -*-
"""Plot the 3D elastomer morphologies"""

import numpy as np
import matplotlib.pyplot as plt
import charu
from utils import *

rc = {
    "charu.doc": "aps",
    "charu.square": 0,
    "charu.tex": True,
    "charu.tex.font": "cmbright",
    "figure.figsize": [150 * charu.pt, 150 / charu.golden * charu.pt],
}

with plt.rc_context(rc):
    Y, phi = 800, 0.2
    data = f"../tests/data//Y_{Y}_phi_{phi}.npy"
    psi = np.load(data)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d', 'proj_type': 'ortho'})
    ax.set_axis_off()

    cubeplot3d(ax,
               psi,
               rcount=256,
               visible_only=True,
               edge_style={
                   "linestyle": "solid", "color": "black"
               })
    plt.tight_layout()
    plt.savefig(
        f"3da.pdf",
        crop=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
        dpi=300,
    )
