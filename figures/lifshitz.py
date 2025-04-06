#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import charu

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    lp_shift = 0.5
    g_min, g_max = 1 - lp_shift, 1 + lp_shift

    # Disorder line.
    g1 = np.linspace(g_min, 1, 100)
    t1 = (1 - g1)**2/g1
    ax.plot(g1, t1, "-", color="#aaaaaa")

    # Microphase line.
    g2 = np.linspace(1, g_max, 100)
    t2 = (1 - g2)**2/g2
    ax.plot(g2, t2, "C3--")

    # Two-phase line.
    g3 = [g_min, 1]
    t3 = [0, 0]
    ax.plot(g3, t3, "C3--")

    # Lifshitz line.
    g3 = [1, 1]
    t3 = [0, 1]
    ax.plot(g3, t3, "-", color="#aaaaaa")

    # Triple line.
    g4 = np.linspace(1, g_max, 100)
    t4 = -(1 - g4)**2/g4*(2 + np.sqrt(6))
    ax.plot(g4, t4, "C0-")

    ax.set_xlim(g_min, g_max)
    ax.set_ylim(t4.min(), t1.max())

    ax.text(1.4, -0.05,
            "stripes",
            va="center", ha="center")

    ax.text(1.2, 0.28,
            r"B{\textmu}E $(q_\textsf{m} > 0)$",
            va="center", ha="center")

    ax.text(0.8, 0.28,
            r"B{\textmu}E $(q_\textsf{m} = 0)$",
            va="center", ha="center")

    ax.text(0.61, 0.061,
            r"disordered",
            va="center", ha="center")

    ax.text(0.86, -0.38,
            r"two-phase coexistence",
            va="center", ha="center")

    ax.scatter(1, 0, color="black", s=7, zorder=100)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\tau$", rotation=0, va="center", ha="center")

    plt.tight_layout()
    plt.savefig(
        "lifshitz.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
    )
    plt.show()
