# -*- coding: utf-8 -*-
"""1D (general) phase-diagram using Maxwell construction.

This program computes the phase-boundary curves using a common tangent
construction.  The basic algorithm is as follows:

1. For a given temperature [measured as the parameter adT = a(T-T_c)],
   start with some value of the chemical potential μ.

2. Simultaneously solve df/dx = μ and dg/dy = μ for x and y.  Here f(x,
   aΔT) and g(y, aΔT) are two free-energy functions representing two
   different phases, and x and y represent the values of the order
   parameter in the two phases.  Check if the found x and y values
   satisfy the osmotic pressure condition, i.e., μ = [f(x) - g(y)]/(x
   - y).  Most likely it won't.

3. Optimize μ until a suitable triplet (μ, x, y) is found for a given
   temperature.

4. For a new temperature, start with Step 1.

This program reproduces a phase diagram similar to Fig. 1 of Thiele et
al., Phys. Rev. E 87, 042915 (2013) and Fig. 2 of Elder and Grant, Phys.
Rev.  E 70, 051605 (2004).
"""

import numpy as np
from scipy.optimize import minimize_scalar

# Parameters
T_c = 0
phi_c = 0
kappa, h, M = 1, 1, np.e
zeta = M * h**2 / kappa

# This term appears in the minimized free energies.
Q = kappa / h**2 * (1 + np.log(zeta))

# Parameters in the Landau energy.
a = 1
b = 1

def dfun(fun, x, args=(), h=1e-6):
    """Use the fourth-order central difference formula to compute dfun/dx."""
    f1 = fun(x + h, *args)
    f2 = fun(x + 2*h, *args)
    b1 = fun(x - h, *args)
    b2 = fun(x - 2*h, *args)

    return (-f2 + 8*f1 - 8*b1 + b2) / (12*h)

def F(x, adT):
    return adT + 3 * b * x**2 + Q

def f_unif(x, adT):
    return 0.5 * (adT+M) * x**2 + 0.25 * b * x**4

def f_stripe(x, adT):
    return -1 / (6*b) * F(x, adT)**2 + f_unif(x, adT)

def minimize(fun, bounds):
    return minimize_scalar(fun, bounds=bounds, method="bounded")

def stripe_unif(adT):
    x0 = np.sqrt(-(adT + Q) / (3*b))
    # Bounds for the stripe phase
    x1, x2 = 0, 0.95 * x0

    # Bounds for the uniform phase
    y1, y2 = 0.5 * x0, 2 * x0

    def fun(mu, find_mu=True):
        f = lambda x: f_stripe(x, adT) - x*mu
        g = lambda y: f_unif(y, adT) - y*mu

        x = minimize(f, bounds=(x1, x2)).x
        y = minimize(g, bounds=(y1, y2)).x

        if find_mu:
            a = f_stripe(x, adT) - f_unif(y, adT)
            b = x - y
            # a, b, and μ should have the same sign.
            if a * b * mu < 0:
                return 1e10
            return abs(a - mu*b)
        else:
            return x, y

    mu = minimize(fun, bounds=(0, 1)).x
    x, y = fun(mu, False)

    # Check if the x, y values are too close to the bounds.
    # If they are, we're still above the tricritical point.
    thres = 0.0001
    if abs(x1 - x) < thres or abs(x2 - x) < thres or abs(y1 - y) < thres or abs(y2 - y) < thres:
        return [mu, x0, x0]

    return [mu, x, y]

N = 50
r = np.linspace(-2 - 1e-3, -2.5, N) # adT
res = np.array([stripe_unif(_) for _ in r])
x, y = res[:, 1], res[:, 2]

# Tricritical point ----------------------------------------------------

# Guess the temperature of the tricritical point.
# This temperature must be __below__ the actual point.
rs = -2.195

# Extrapolate the binodals upward and find the intersection point.
r1, x1, y1 = r[r < rs][0], x[r < rs][0], y[r < rs][0]
r2, x2, y2 = r[r < rs][1], x[r < rs][1], y[r < rs][1]
# Plot interpolation points.
# ax.plot(x1, r1, "ro")
# ax.plot(y1, r1, "ro")
# ax.plot(x2, r2, "bo")
# ax.plot(y2, r2, "bo")

# Intersection point of two lines passing through (x1, r1), (x2, r2) and (y1, r1), (y2, r2).
m1 = (r2-r1) / (x2-x1)
b1 = r1 - m1*x1
m2 = (r2-r1) / (y2-y1)
b2 = r1 - m2*y1
xs, rs = (b2-b1) / (m1-m2), m1 * (b2-b1) / (m1-m2) + b1

print(f"Estimated tricritical point at ({xs:.2}, {rs:.2})")
# Second-order curve.
r_2nd = r[r > rs]
x_2nd = x[r > rs]
r_2nd = np.insert(r_2nd, -1, rs)
x_2nd = np.insert(x_2nd, -1, xs)
r_2nd = np.insert(r_2nd, 0, -2) # critical point
x_2nd = np.insert(x_2nd, 0, 0) # critical point
i = np.argsort(r_2nd)
x_2nd, r_2nd = x_2nd[i], r_2nd[i]

# First-order curves.
r_1st = r[r <= r1]
x_1st = x[r <= r1]
y_1st = y[r <= r1]
r_1st = np.insert(r_1st, 0, rs)
x_1st = np.insert(x_1st, 0, xs)
y_1st = np.insert(y_1st, 0, xs)

# Plotting ------------------------------------------------------------

import matplotlib.pyplot as plt
import charu

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    ax.plot(x_2nd, r_2nd, "C3--")
    ax.plot(-x_2nd, r_2nd, "C3--")

    shade = {"color": "C0", "alpha": 0.2}
    ax.plot(x_1st, r_1st, "C0")
    ax.plot(y_1st, r_1st, "C0")
    ax.fill_betweenx(r_1st, x_1st, y_1st, **shade)
    ax.plot(-x_1st, r_1st, "C0")
    ax.plot(-y_1st, r_1st, "C0")
    ax.fill_betweenx(r_1st, -x_1st, -y_1st, **shade)

    # Tricritical point.
    ax.scatter(xs, rs, color="black", s=8, zorder=100)
    ax.scatter(-xs, rs, color="black", s=8, zorder=100)

    # Labels.
    r_mark = -2.4
    _, x_mark, y_mark = stripe_unif(r_mark)
    ax.text(0.5 * (x_mark+y_mark), r_mark, "S + U", horizontalalignment="center")
    ax.text(-0.5 * (x_mark+y_mark), r_mark, "U + S", horizontalalignment="center")
    ax.text(0, -2.2, "S", horizontalalignment="center")
    ax.text(-0.4, -2.05, "U", horizontalalignment="center")
    ax.text(0.4, -2.05, "U", horizontalalignment="center")

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-2.5, -1.95)
    ax.set_xlabel(r"$\psi_0$")
    ax.set_ylabel(r"$T$")

    ax.set_xlabel(r"$\psi_0$")
    ax.set_ylabel(r"$T$")

    plt.tight_layout()
    plt.savefig(
        "phase1d.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
