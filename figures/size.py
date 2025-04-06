# -*- coding: utf-8 -*-

import numpy as np
from params import parameters

# Parameters in the Landau energy (estimated).
a = 0.025 # kPa/K
b = 2 # kPa

# This is mostly an estimate based on Fig. S7 of the Nat. Mat. paper.
T_c = 70 # in Celsius
phi_c = 0.2

# Other parameters (affine model)
# B = 0.024 # kPa um^2
# n = 35 # number of cross-links we coarse-grain over.

# Other parameters (phantom model)
B = 0.012 # kPa um^2
n = 80 # number of cross-links we coarse-grain over.

# Interface parameter.
kappa = 0.013 # kPa um^2

def domain_size(Y):
    """Domain size (in μm) as a function of Y (in kPa)."""
    # Eq. (13), step by step (affine).
    # l = 3 * B * n**2 * phi_c**(-2 / 3)
    # l /= Y * np.log(B * n**2 / kappa * phi_c**(-7 / 3))

    # Eq. (13), step by step (phantom)
    l = 3 * B * n**2
    l /= Y * np.log(B * n**2 / kappa * phi_c**(-5 / 3))

    l = 2*np.pi*np.sqrt(l)
    return l

def T_m(Y, phi):
    """Microphase separation temperature (in C) as a function of Y (in kPa)."""
    kappa, h, M, zeta = parameters(Y)

    # This term appears in the minimized free energies.
    Q = kappa / h**2 * (1 + np.log(zeta))

    return T_c - (3 * b * (phi - phi_c)**2 + Q) / a

# Plotting -------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.ticker
import charu

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "figure.figsize": [262 * charu.pt, 262 * charu.pt],
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 2)

    labelpos = (0.075, 0.075)

    # Domain size ----------------------------------------------------------

    ax = axes[0]
    ax.set_box_aspect(1)

    ax.text(*labelpos, r"\textbf{(a)}", transform=ax.transAxes)

    Y_exp, size, err = np.loadtxt("../experiments/size.dat", usecols=(0, 1, 3), unpack=True)
    err = err / 2

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$Y$ (kPa)")
    ax.set_ylabel(r"$\Lambda$ (\textmu m)", labelpad=-3)
    ax.set_xlim(8, 1000)
    ax.set_xticks([10, 40, 180, 800])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([1, 10])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Experimental plot.
    ax.plot(Y_exp, size, "C0o", markerfacecolor="none", zorder=100)

    # Theoretical plot.
    Y_th = np.linspace(8, 1000, 100)
    ax.plot(Y_th, domain_size(Y_th), "--", color="#999999")

    ax.plot(Y_th[3:30], 1.45 * domain_size(Y_th[3:30]), ":", color="#666666", linewidth=0.5)
    ax.text(0.45,
            0.55,
            r"$\sim Y^{\small-1/2}$",
            transform=ax.transAxes,
            color="#666666",
            size=7,
            rotation=-45)

    # Microphase separation temperature ------------------------------------

    micro = np.load("../experiments/micro.npy")
    Y_list = np.array([800, 350, 180, 40, 10])

    ax = axes[1]

    ax.set_box_aspect(1)
    ax.text(*labelpos, r"\textbf{(b)}", transform=ax.transAxes)
    ax.set_xlabel("$Y$ (kPa)")
    ax.set_ylabel(r"$T_\mathrm{m}$ (${}^{\circ}\mathrm{C}$)", labelpad=8)
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlim(-20, 820)
    ax.set_ylim(10, 70)

    # These are T_m and phi_0 for an initial swelling temperature of 60 C.
    T_m_exp = micro[:, 1][:, 2]
    phi_exp = micro[:, 1][:, 1]
    ax.plot(Y_list, T_m_exp, "C0o", zorder=100)

    T_m_th = []
    for i, Y in enumerate(Y_list):
        T_m_th += [T_m(Y, phi_exp[i])]
    ax.plot(Y_list, T_m_th, "C3x", zorder=100)

    # Guideline.
    guide = np.array([-1000, 1000])
    ax.plot(guide, 65 - 0.05*guide, "--", zorder=-100, color="#aaaaaa")

    plt.tight_layout()
    plt.savefig(
        "size.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
