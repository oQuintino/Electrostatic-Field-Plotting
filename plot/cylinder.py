# shapes/coordinates file

import numpy as np

from functools import cached_property
from dataclasses import dataclass


@dataclass
class CoaxialCylinder:
    """
    Geometria do cilindro
    """

    r_i: float
    r_o: float
    L: float
    Δ: float = 1e-2  # espaçamento desejado entre vetores

    @cached_property
    def spaced_coordinates(self):
        N_r = int((self.r_o - self.r_i) / self.Δ)
        N_theta = int(2 * np.pi * ((self.r_i + self.r_o) / 2) / self.Δ)
        N_z = int(self.L / self.Δ)

        # garantir mínimo
        N_r = max(N_r, 5)
        N_theta = max(N_theta, 20)
        N_z = max(N_z, 5)

        r = np.linspace(self.r_i, self.r_o, N_r)
        theta = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)
        half_L = self.L / 2
        z = np.linspace(-half_L, half_L, N_z)

        return r, theta, z

    @cached_property
    def points(self):
        r, theta, z = self.spaced_coordinates

        rr, tt, zz = np.meshgrid(r, theta, z, indexing="ij")
        x = rr * np.cos(tt)
        y = rr * np.sin(tt)

        # flatten
        x_f = x.ravel()
        y_f = y.ravel()
        z_f = zz.ravel()

        return x_f, y_f, z_f

    @cached_property
    def coordinates(self):
        x, y, z = self.points

        r = np.sqrt(x**2 + y**2)

        return r, z

    def to_cartesian(self, Er: np.typing.NDArray, Ez: np.typing.NDArray):
        """
        Convert cylindrical field components (Er, Ez) defined at self.points
        into Cartesian vectors.

        Parameters
        ----------
        Er, Ez : arrays of shape (N,)
            Field components aligned with self.points ordering.
        """
        x, y, z = self.points

        r = np.sqrt(x**2 + y**2)
        r_safe = np.maximum(r, 1e-15)

        Ex = Er * (x / r_safe)
        Ey = Er * (y / r_safe)

        vectors = np.vstack((Ex, Ey, Ez)).T
        points = np.vstack((x, y, z)).T

        mag = np.linalg.norm(vectors, axis=1)
        mag_safe = np.maximum(mag, 1e-15)

        vectors_unit = vectors / mag_safe[:, None]

        return points, vectors_unit, mag
