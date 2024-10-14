# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq, fftshift
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import *

class PhaseFieldModel:

    def __init__(self,
                 a,
                 b,
                 kappa=0.001,
                 phi_s=1.0,
                 phi_c=1.0,
                 M=1.0,
                 h=0.1,
                 d=2,
                 n=128,
                 L=None,
                 dt=0.001,
                 randomness=0.1,
                 energy=True):
        if d not in (2, 3):
            raise ValueError(f"The dimension (d = {d}) must be 2 or 3.")

        # Choose a random initial condition.  Too much randomness can
        # result in overflows, whereas too little randomness will
        # sometimes prevent the system from reaching its true energy
        # minimum.
        self.psi = np.random.normal(size=(n, ) * d, scale=randomness)
        # Make sure that the mean value of the order parameter is exact.
        self.psi += phi_s - phi_c - self.psi.mean()

        # If the size of the box isn't given, choose it to be equal to
        # 10 times the size from linear analysis.
        if L is None:
            zeta = M * h**2 / (kappa * phi_s**2)
            size = 2 * np.pi * h / np.sqrt(np.log(zeta))
            L = np.round(5 * size)

        # Domain setup.
        x = np.linspace(-L, L, n)
        dx = x[1] - x[0]
        mesh = np.meshgrid(*(x, ) * d)

        # Wavenumber arrays.  The wavenumbers need to be multiplied by
        # 2pi to get usual physics conventions.
        q = 2 * np.pi * fftfreq(n, d=dx)
        if d == 2:
            q2 = q[:, None]**2 + q[None, :]**2
        else:
            q2 = q[:, None, None]**2 + q[None, :, None]**2 + q[None, None, :]**2

        # Coarse-graining kernel.
        K = np.exp(-np.sum(np.asarray(mesh)**2, axis=0) / (2 * h**2))
        K /= (2 * np.pi * h**2)**(d / 2) # normalization

        # DFTs assume that the "origin" of the kernel are at the "ends".
        # But the kernel we've defined above has an origin at the center.
        # So shift appropriately to put the origin at the "ends."
        K = fftshift(K)
        # The multiplication by dx^d is to turn a discrete DFT sum into
        # an integral.
        K_q = fftn(K) * (dx ** d)

        # Precomputable stuff that's used in each step.
        self.A = 1 - 3 * q2 * a * dt
        self.B = q2 * b * dt
        self.C = 1 + dt * q2 * (-2 * a + kappa * q2 + M * K_q / (phi_s * phi_s))

        # If energy computation is required, define some additional
        # instance variables.
        self.E = None
        self.energy = energy
        if energy:
            self.a = a
            self.b = b
            self.kappa = kappa
            self.n = n
            self.d = d
            self.N = n**d
            self.q2 = q2
            self.K_eff_q = M * K_q / (phi_s * phi_s)

    def evolve(self):
        psi_q = fftn(self.psi)
        psi3_q = fftn(self.psi**3)

        psi_q = (self.A * psi_q - self.B * psi3_q) / self.C
        self.psi = ifftn(psi_q).real

        # Energy computation.
        if self.energy:
            psi_q2 = psi_q * psi_q.conj()
            psi2 = self.psi*self.psi
            E = np.zeros((self.n, ) * self.d)

            # Compute Landau terms first.
            E += 1 / 2 * self.a * psi2
            E += 1 / 4 * self.b * psi2 * psi2

            # The interfacial energy and the elastic energy can be
            # computed using the DFT version of Parceval's theorem.
            E += 1 / 2 * (self.q2 * self.kappa * psi_q2 / self.N).real
            E += 1 / 2 * (self.K_eff_q * psi_q2 / self.N).real

            self.E = np.mean(E).real
