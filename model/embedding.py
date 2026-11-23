"""Laguerre 基函数与速度历史积分。"""

import numpy as np
from scipy.linalg import expm
from scipy.special import eval_laguerre


def laguerre_basis(t, K, a):
    return np.array([np.sqrt(2.0 * a) * np.exp(-a * t) * eval_laguerre(k, 2.0 * a * t) for k in range(K)])


def kernel_from_coefficients(u, K, a, t):
    return u @ laguerre_basis(t, K, a)


def embed(v, t, K, a):
    dt = t[1] - t[0]
    R = -a * np.eye(K) - 2.0 * a * np.tril(np.ones((K, K)), -1)
    decay = expm(dt * R)
    gain = dt * laguerre_basis(np.array([dt]), K, a)[:, 0]
    L = v.shape[1] - 1
    z = np.zeros((v.shape[0], L, K))
    for l in range(1, L):
        z[:, l] = z[:, l - 1] @ decay.T + v[:, l - 1, None] * gain
    return z
