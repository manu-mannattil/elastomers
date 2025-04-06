# -*- coding: utf-8 -*-

from params import parameters
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils import gprange
import charu

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "axes.axisbelow": False,
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    Y = 800 # Young's modulus in kPa

    # Choose phi to be the mean of phi_exp for computing kappa, h, and M.
    # This is mostly an estimate based on Fig. S7 of the Nat. Mat. paper.
    T_c = 70
    phi_c = 0.2
    kappa, h, M, zeta = parameters(Y)
    print(f"M = {M}, h = {h}")

    # Parameters in the Landau energy (estimated).
    a = 2.5e-2 # kPa/K
    b = 2 # kPa

    # Mean order parameter and temperature.
    psi_0 = 0.45 - phi_c # one of the experimental phi_0's
    T_m = np.round(T_c - (3*b*psi_0**2 + M*(1 + np.log(zeta))/zeta)/a)

    q_max = np.sqrt(np.log(zeta))/h
    q = np.linspace(q_max/2, 1.5 * q_max, 1000)

    num_lines = 7
    alpha_lines = gprange(1.0, 0.1, num_lines)
    dT_list = np.linspace(2, 10, num_lines)
    print(f"S(q) exact: T_max = {dT_list[-1] + T_m}")
    print(f"S(q) exact: T_min = {dT_list[0] + T_m}")
    for i, dT in enumerate(np.linspace(2, 10, num_lines)):
        T = T_m + dT # slightly above T_m
        tau = 2/kappa*(a*(T - T_c) + 3*b*psi_0**2 + M)

        S_q = a*(T - T_c) + 3*b*psi_0**2 + kappa*q**2 + M * np.exp(-h**2*q**2)
        S_0 = a*(T - T_c) + 3*b*psi_0**2 + M
        S_q = S_0/S_q

        ax.plot(q, S_q, color="C0", alpha=alpha_lines[i])

    ax.set_ylim(0, 1e5)
    ax.set_xlim(q_max/2, 1.5*q_max)
    ax.set_ylabel(r"$\left. S(\bm{q}) \middle/ S(0) \right.$")
    ax.set_xlabel(r"$q$ ($\textsf{\textmu m}^{-1})$")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,2))

    ax.arrow(
        x=0.25, y=0.25, dx=0, dy=0.40, transform=ax.transAxes,
        head_width=0.015, head_length=0.06, length_includes_head=True,
        lw=0.5,
        color="#999999",
    )

    ax.text(0.22, 0.42,
            r"cooling",
            transform=ax.transAxes,
            va="center",
            ha="center",
            rotation=90,
            color="#999999")

    plt.tight_layout()
    plt.savefig(
        "sfactor.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
    )

    plt.show()
