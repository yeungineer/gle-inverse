"""Finite-dimensional bases for memory-kernel reconstruction."""

import numpy as np
from scipy.interpolate import BSpline


class Basis:
    def __init__(self, n_basis, t_max):
        self.n_basis = int(n_basis)
        self.t_max = float(t_max)

    def matrix(self, t):
        raise NotImplementedError

    def derivative_matrix(self, t):
        raise NotImplementedError

    def gram(self):
        raise NotImplementedError


class SplineBasis(Basis):
    def __init__(self, n_basis, t_max, quad_order=16, knot_power=1.25):
        super().__init__(n_basis, t_max)
        self.degree = 3
        self.quad_order = int(quad_order)
        self.knot_power = float(knot_power)
        if not np.isfinite(self.knot_power) or self.knot_power <= 0.0:
            raise ValueError("knot_power must be finite and positive.")
        n_internal = self.n_basis - self.degree - 1
        if n_internal < 0:
            raise ValueError("n_basis must be at least degree + 1.")
        fractions = np.arange(1, n_internal + 1, dtype=float) / (n_internal + 1)
        internal = self.t_max * fractions**self.knot_power
        self.knots = np.concatenate((np.zeros(self.degree + 1), internal, np.full(self.degree + 1, self.t_max)))
        self.splines = []
        for index in range(self.n_basis):
            coefficient = np.zeros(self.n_basis, dtype=float)
            coefficient[index] = 1.0
            self.splines.append(BSpline(self.knots, coefficient, self.degree, extrapolate=False))

    def matrix(self, t):
        t = np.asarray(t, dtype=float).reshape(-1)
        return np.column_stack([np.nan_to_num(spline(t), nan=0.0) for spline in self.splines])

    def derivative_matrix(self, t):
        t = np.asarray(t, dtype=float).reshape(-1)
        return np.column_stack([np.nan_to_num(spline.derivative()(t), nan=0.0) for spline in self.splines])

    def gram(self):
        nodes, weights = np.polynomial.legendre.leggauss(self.quad_order)
        gram = np.zeros((self.n_basis, self.n_basis), dtype=float)
        breaks = np.unique(self.knots)
        for left, right in zip(breaks[:-1], breaks[1:]):
            if right <= left:
                continue
            half = 0.5 * (right - left)
            midpoint = 0.5 * (right + left)
            points = midpoint + half * nodes
            values = self.matrix(points)
            derivatives = self.derivative_matrix(points)
            weighted = half * weights[:, None]
            gram += values.T @ (weighted * values)
            gram += derivatives.T @ (weighted * derivatives)
        return 0.5 * (gram + gram.T)


def get_basis(config):
    return SplineBasis(n_basis=int(config["num_basis"]), t_max=float(config["t_max"]))
