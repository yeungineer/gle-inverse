"""Lang 与 Lu 的 Prony–Sobolev 核估计。

https://github.com/QuanjunLang/GLE-memory-kernel-learning
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import hankel
from scipy.optimize import minimize_scalar

from model.data import get_force


def ensemble_correlation(a, v):
    return np.mean(a * v[:, :1], axis=0)


def prony(x, p, dt, constrained):
    N = x.size
    Y = hankel(x[:N - p], x[N - p - 1:])
    r = np.linalg.eigvals(np.linalg.pinv(Y[:, :-1]) @ Y[:, 1:])
    r = np.where(np.abs(r) >= 1.0, r / (np.abs(r) + 1e-6), r)
    lam = np.log(r) / dt
    Z = r[None, :] ** np.arange(N)[:, None]
    if constrained:
        C = np.vstack([-lam[1:] / lam[0], np.eye(p - 1)])
        D = Z @ C
        w = C @ np.linalg.solve(D.conj().T @ D, D.conj().T @ x)
    else:
        w = np.linalg.lstsq(Z, x.astype(complex), rcond=None)[0]
    bad = (r.real < 0.0) & (r.imag == 0.0)
    lam = np.concatenate([lam[~bad], lam[bad], np.conj(lam[bad])])
    w = np.concatenate([w[~bad], w[bad] / 2.0, w[bad] / 2.0])
    return w, lam


def exponential_sum(w, lam, t, power):
    return np.real(np.exp(np.outer(t, lam)) @ (w * lam**power))


def negative_curvature(lam, s, beta, xi):
    lam = np.atleast_1d(lam)[:, None]
    f = s**2 / (s**2 + lam**2)
    cf = 1.0 - f
    eta = np.linalg.norm(f * xi, axis=1)
    rho = np.linalg.norm(cf * beta, axis=1)
    f1 = -2.0 * f * cf / lam
    f2 = -f1 * (3.0 - 4.0 * f) / lam
    phi = np.sum(f * f1 * np.abs(xi) ** 2, axis=1)
    psi = np.sum(cf * f1 * np.abs(beta) ** 2, axis=1)
    dphi = np.sum((f1**2 + f * f2) * np.abs(xi) ** 2, axis=1)
    dpsi = np.sum((-f1**2 + cf * f2) * np.abs(beta) ** 2, axis=1)
    deta, drho = phi / eta, -psi / rho
    ddeta = dphi / eta - deta * (deta / eta)
    ddrho = -dpsi / rho - drho * (drho / rho)
    dlogeta, dlogrho = deta / eta, drho / rho
    ddlogeta = ddeta / eta - dlogeta**2
    ddlogrho = ddrho / rho - dlogrho**2
    return -(dlogrho * ddlogeta - ddlogrho * dlogeta) / (dlogrho**2 + dlogeta**2) ** 1.5


def l_curve_standard_form(A, b):
    lambdas = 10.0 ** np.linspace(-20.0, 2.0, 100)
    U, s, _ = np.linalg.svd(A)
    b_coord = U.T @ b
    s2 = np.sqrt(s)
    beta, xi = b_coord / s2, b_coord / s
    i = np.argmin(negative_curvature(np.sqrt(lambdas), s2, beta, xi))
    bounds = (np.sqrt(lambdas[max(i - 1, 0)]), np.sqrt(lambdas[min(i + 1, lambdas.size - 1)]))
    corner = minimize_scalar(lambda x: negative_curvature(x, s2, beta, xi)[0], bounds=bounds, method="bounded",
                             options={"xatol": 1e-4}).x
    kappa_max = -negative_curvature(corner, s2, beta, xi)[0]
    lam = corner**2 if kappa_max >= 0.0 else lambdas[0]
    return U @ (b_coord / (s + lam))


def rkhs_regularized_solution(A, b):
    U, s, _ = np.linalg.svd(A)
    sqrt_A = U @ np.diag(np.sqrt(s)) @ U.T
    return sqrt_A @ l_curve_standard_form(A @ A, sqrt_A @ b)


def volterra_convolution(h, phi, dx):
    full = np.convolve(h, phi)[:h.size]
    return dx * (full - 0.5 * (h[0] * phi + h * phi[0]))


def sobolev_regression(h, g, dg, comb, settings):
    T, n_grid, n_knots, omega = settings["t_max"], settings["grid_count"], settings["knot_count"], settings["omega"]
    x = np.linspace(0.0, T, n_grid)
    dx = x[1] - x[0]
    knots = np.concatenate([np.zeros(3), np.linspace(0.0, T, n_knots + 1), np.full(3, T)])
    basis = BSpline.design_matrix(x, knots, 3).toarray()
    h_x = h(x)
    conv = np.column_stack([volterra_convolution(h_x, basis[:, j], dx) for j in range(basis.shape[1])])
    conv_dx = np.gradient(conv, dx, axis=0)
    rho = np.exp(-omega * x)
    A_space = (conv * rho[:, None]).T @ conv * dx
    A_derivative = (conv_dx * rho[:, None]).T @ conv_dx * dx
    b_space = conv.T @ (g(x) * rho) * dx
    b_derivative = conv_dx.T @ (dg(x) * rho) * dx
    A = comb * A_derivative + (1.0 - comb) * A_space
    b = comb * b_derivative + (1.0 - comb) * b_space
    return knots, rkhs_regularized_solution(A, b)


def estimate_kernel_prony_sobolev(v, t, force_config, settings):
    r, N, p = settings["observation_ratio"], settings["correlation_length"], settings["modes"]
    v_obs = v[:, ::r][:, :N]
    dt_obs = r * (t[1] - t[0])
    w, lam = prony(ensemble_correlation(v_obs, v_obs), p, dt_obs, constrained=True)
    w_f, lam_f = np.zeros(0), np.zeros(0)
    if force_config["type"] != "zero":
        w_f, lam_f = prony(ensemble_correlation(get_force(force_config, v_obs), v_obs),
                           p + 2, dt_obs, constrained=False)
    h = lambda s: exponential_sum(w, lam, s, 0)
    g = lambda s: -exponential_sum(w, lam, s, 1) + exponential_sum(w_f, lam_f, s, 0)
    dg = lambda s: -exponential_sum(w, lam, s, 2) + exponential_sum(w_f, lam_f, s, 1)
    a1 = h(np.zeros(1))[0]
    a2 = np.real(-np.sum(w / lam))
    knots, c = sobolev_regression(h, g, dg, a1**2 / (a1**2 + a2**2), settings)
    return BSpline(knots, c, 3)
