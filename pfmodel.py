# -*- coding: utf-8 -*-

import numpy as np

class PhaseFieldModel:

    def __init__(self,
                 T,
                 T_c=70,
                 phi_c=0.2,
                 phi_0=0.5,
                 a=0.025,
                 b=2,
                 kappa=0.013,
                 M=4000,
                 h=0.5,
                 d=2,
                 n=128,
                 L=5,
                 dt=1,
                 disorder=0.1):
        """Phase-field model for elastic microphase separation.

        Parameters
        ----------
        T : float
            Temperature (in Celsius).
        T_c : float
            Critical temperature (in Celsius).
        phi_c : float
            Critical polymer volume fraction.
        a : float
            Landau parameter a (in kPa).
        b : float
            Landau parameter b (in kPa).
        kappa : float
            Interfacial parameter (in kPa μm^2).
        M : float
            Rescaled longitudinal modulus (in kPa)
        h : float
            Coarse-graining length (in μm).
        d : int
            Dimension of box used for energy minimization.
        n : int
            Number of grid points along each axis of the box.
        L : float
            side length of the box (in μm)
        dt : float
            time step
        disorder: float
            Controls the randomness in the initial configuration.
        """
        if d not in (1, 2, 3):
            raise ValueError(f"The dimension (d = {d}) must be 1, 2, or 3.")

        # Choose a random initial condition.  Too much randomness can
        # result in overflows, whereas too little randomness will
        # sometimes prevent the system from reaching its true energy
        # minimum.
        self.psi = np.random.normal(size=(n, ) * d, scale=disorder)
        # Make sure that the mean value of the order parameter is exact.
        self.psi += phi_0 - phi_c - self.psi.mean()

        # Domain setup.
        x = np.linspace(-L, L, n)
        dx = x[1] - x[0]

        # Wavenumber arrays.  The wavenumbers need to be multiplied by
        # 2pi to get usual physics conventions.
        q = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        if d == 1:
            q2 = q**2
        elif d == 2:
            q2 = q[:, None]**2 + q[None, :]**2
        else:
            q2 = q[:, None, None]**2 + q[None, :, None]**2 + q[None, None, :]**2

        # Coarse-graining kernel.
        if d == 1:
            K = np.exp(-x**2 / (4 * h**2))
        else:
            mesh = np.meshgrid(*(x, ) * d)
            K = np.exp(-np.sum(np.asarray(mesh)**2, axis=0) / (4 * h**2))

        # Normalize the kernel.
        K /= (4 * np.pi * h**2)**(d / 2)

        # DFTs assume that the "origin" of the kernel are at the "ends".
        # But the kernel we've defined above has an origin at the center.
        # So shift appropriately to put the origin at the "ends."
        K = np.fft.fftshift(K)
        # The multiplication by dx^d is to turn a discrete DFT sum into
        # an integral.
        K_q = np.fft.fftn(K) * (dx**d)

        # Precomputable stuff that's used in each step.
        self.A = 1 - 3 * a * (T-T_c) * q2 * dt
        self.B = b * q2 * dt
        self.C = 1 + dt * q2 * (kappa*q2 - 2 * a * (T-T_c) + M*K_q)

    def evolve(self):
        """Evolves the energy-minimization equation in time."""
        psi_q = np.fft.fftn(self.psi)
        psi3_q = np.fft.fftn(self.psi**3)

        psi_q = (self.A * psi_q - self.B * psi3_q) / self.C
        self.psi = np.fft.ifftn(psi_q).real
