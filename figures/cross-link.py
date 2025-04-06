# -*- coding: utf-8 -*-
"""Young's modulus vs. Cross-linker concentration.

This is a reproduction of Fig. S1 from Fernandez-Rico et al.  For some
reason, these authors decided to use a log-log scale in their original
plot, which makes the scaling ambiguous.
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

    ax.plot(x, y, "ro")
    ax.plot(x, y, "wo", mfc="w", ms=7, zorder=-10)
    ax.plot(x, y, "--", zorder=-20)
    ax.plot(x, y, "w-", zorder=-30, lw=10)

    # Linear fit.
    # A = np.vstack([x[:-1], np.ones(len(x) - 1)]).T
    # m, c = np.linalg.lstsq(A, y[:-1])[0]
    # ax.plot(x, m*x + c, "k--", zorder=-1000)

    ax.set_xlabel(r"cross-linker concentration $\propto \nu$ (mol m$^{-3}$)")
    ax.set_ylabel(r"$Y$ (kPa)")

    ax.set_xlim(4.5, 13.5)
    ax.plot([0, 15], [1.5 * 250, 1.5 * 250], "--", color="#666666", zorder=-100)

    ax.annotate(r"\footnotesize $3G_\textsf{\tiny N}/2$ for PDMS (expected)", 
                xy=(6, 370),
                xytext=(7, 610),
                ha="center", va="bottom",
                color="#666666",
                arrowprops=dict(arrowstyle="->", color="#999999", linewidth=0.5))


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
