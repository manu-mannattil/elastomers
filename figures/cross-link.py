# -*- coding: utf-8 -*-

"""Young's modulus vs. Cross-linker concentration.

This is a reproduction of Fig. S1 from Fernandez-Rico et al.  For some
reason, these authors decided to use a log-log scale in their original
plot, which makes the scaling less apparent.
"""

import numpy as np
import matplotlib.pyplot as plt
import charu

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "axes.axisbelow": False,
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    x, y = np.loadtxt("../experiments/figs1.dat", unpack=True)

    # Make the coefficient matrix; note the transpose.
    A = np.vstack([x[:-1], np.ones(len(x)-1)]).T
    m, c = np.linalg.lstsq(A, y[:-1])[0]

    ax.plot(x, y, "ro")
    ax.plot(x, m*x + c, "k--", zorder=-1000)

    ax.set_xlabel(r"cross-linker concentration (mol m$^{-3}$)")
    ax.set_ylabel(r"Young's modulus (kPa)")

    plt.tight_layout()
    plt.savefig(
        "cross-link.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
    )

    plt.show()
