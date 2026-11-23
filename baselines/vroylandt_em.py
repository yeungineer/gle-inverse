"""Vroylandt 等人的似然 EM 核估计。

https://github.com/HadrienNU/GLE_AnalysisEM
"""

import numpy as np
from scipy.linalg import expm


def random_clever_gen(n, rng, max_ev, min_re_ev):
    while True:
        lamb, vec = np.linalg.eig(max_ev * rng.standard_normal(size=(n, n)) / n)
        Ap = np.real(vec @ np.diag(0.5 * np.abs(lamb.real) + 1j * lamb.imag) @ np.linalg.inv(vec))
        if np.all(np.linalg.eigvals(Ap).real > min_re_ev) and np.all(np.linalg.eigvals(Ap + Ap.T) > min_re_ev):
            return Ap


def initial_friction(d, rng, max_ev, min_re_ev):
    A = np.zeros((1 + d, 1 + d))
    A[1:, 1:] = random_clever_gen(d, rng, max_ev, min_re_ev)
    A[:1, :1] = random_clever_gen(1, rng, max_ev * 1e-2, min_re_ev * 1e-2)
    A[1, 0] = -1.0
    A[0, 1] = 1.0
    return A


def filter_smoother(v, mutilde, R, Q, mu0, sig0):
    M, L, d = v.shape[0], v.shape[1] - 1, mu0.size
    muf = np.zeros((M, L + 1, d))
    Sigf = np.zeros((L + 1, d, d))
    muf[:, 0] = mu0
    Sigf[0] = sig0
    for t in range(1, L + 1):
        mutemp = mutilde[:, t - 1] + muf[:, t - 1] @ R.T
        Sigtemp = Q + R @ Sigf[t - 1] @ R.T
        gain = Sigtemp[1:, 0] / Sigtemp[0, 0]
        muf[:, t] = mutemp[:, 1:] + (v[:, t] - mutemp[:, 0])[:, None] * gain[None, :]
        sig = Sigtemp[1:, 1:] - np.outer(Sigtemp[1:, 0], Sigtemp[1:, 0]) / Sigtemp[0, 0]
        Sigf[t] = 0.5 * sig + 0.5 * sig.T
    mus, Sigs = muf[:, L].copy(), Sigf[L].copy()
    muh = np.zeros((M, L, 2 * d))
    Sigh = np.zeros((L, 2 * d, 2 * d))
    for t in range(L - 1, -1, -1):
        Rs = Sigf[t] @ R.T @ np.linalg.inv(Q + R @ Sigf[t] @ R.T)
        mu_rev = muf[:, t] + v[:, t + 1, None] * Rs[None, :, 0] - (muf[:, t] @ R.T + mutilde[:, t]) @ Rs.T
        sig_rev = Sigf[t] - Rs @ R @ Sigf[t]
        mu_t = mu_rev + mus @ Rs[:, 1:].T
        sig_t = Rs[:, 1:] @ Sigs @ Rs[:, 1:].T + sig_rev
        muh[:, t] = np.concatenate([mus, mu_t], axis=1)
        Sigh[t, :d, :d] = Sigs
        Sigh[t, d:, :d] = Rs[:, 1:] @ Sigs
        Sigh[t, :d, d:] = Sigh[t, d:, :d].T
        Sigh[t, d:, d:] = sig_t
        mus, Sigs = mu_t, sig_t
    return muh, Sigh


def sufficient_statistics(v, f, muh, Sigh, d):
    x, dx = v[:, :-1], np.diff(v, axis=1)
    h_next, h = muh[:, :, :d], muh[:, :, d:]
    dh = h_next - h
    S_nn, S_tn, S_nt, S_tt = (Sigh[:, :d, :d].mean(0), Sigh[:, d:, :d].mean(0), Sigh[:, :d, d:].mean(0),
                              Sigh[:, d:, d:].mean(0))
    mean = lambda a, b: np.einsum("mti,mtj->ij", a, b) / (a.shape[0] * a.shape[1])
    mu_nn, mu_tn, mu_nt, mu_tt = mean(h_next, h_next), mean(h, h_next), mean(h_next, h), mean(h, h)
    x1, dx1, f1 = x[:, :, None], dx[:, :, None], f[:, :, None]
    xx = np.block([[mean(x1, x1), mean(x1, h)], [mean(h, x1), S_tt + mu_tt]])
    xdx = np.block([[mean(x1, dx1), mean(x1, dh)], [mean(h, dx1), S_tn + mu_tn - S_tt - mu_tt]])
    dxdx_hh = S_nn + mu_nn - S_tn - S_nt - mu_tn - mu_nt + S_tt + mu_tt
    dxdx = np.block([[mean(dx1, dx1), mean(dh, dx1).T], [mean(dh, dx1), dxdx_hh]])
    bkx = np.concatenate([mean(f1, x1), mean(f1, h)], axis=1)
    bkdx = np.concatenate([mean(f1, dx1), mean(f1, dh)], axis=1)
    hS = (0.5 * d * (1.0 + np.log(2.0 * np.pi)) + 0.5 * np.mean(np.log(np.linalg.det(Sigh)))
          - 0.5 * np.mean(np.log(np.linalg.det(Sigh[:, d:, d:]))))
    return {"xx": xx, "xdx": xdx, "dxdx": dxdx, "bkx": bkx, "bkdx": bkdx, "bkbk": mean(f1, f1),
            "mu0": h[:, 0].mean(0), "hS": hS}


def residual_second_moment(stat, friction, dt):
    e = np.zeros((friction.shape[0], 1))
    e[0, 0] = dt
    bkbk, bkdx, bkx = e @ stat["bkbk"] @ e.T, e @ stat["bkdx"], e @ stat["bkx"]
    m = stat["dxdx"] + friction @ stat["xdx"] + (friction @ stat["xdx"]).T - bkdx.T - bkdx
    return m + friction @ stat["xx"] @ friction.T - friction @ bkx.T - (friction @ bkx.T).T + bkbk


def m_step(stat, dt):
    e = np.zeros((stat["xx"].shape[0], 1))
    e[0, 0] = dt
    friction = -(stat["xdx"].T - e @ stat["bkx"]) @ np.linalg.inv(stat["xx"])
    residual = residual_second_moment(stat, friction, dt)
    return friction, 0.5 * (residual + residual.T)


def log_likelihood(stat, friction, Q, dt):
    n = friction.shape[0]
    quad = -np.trace(np.linalg.inv(Q) @ (0.5 * residual_second_moment(stat, friction, dt)))
    return quad - 0.5 * (n * np.log(2.0 * np.pi) + np.log(np.linalg.det(Q))) + stat["hS"]


def fit_em(v, dt, force, settings):
    d = settings["hidden_dimension"]
    rng = np.random.RandomState(settings["seed"])
    A = initial_friction(d, rng, max_ev=(1.0 / 50) / dt, min_re_ev=(0.5 / v.shape[1]) / dt)
    friction = A * dt
    Q = friction + friction.T
    mu0, sig0 = np.zeros(d), np.eye(d)
    f = force[:, :-1]
    best, best_ll, lower_bound = friction, -np.inf, -np.inf
    history = []
    for n_iter in range(settings["max_iterations"]):
        R = np.eye(1 + d)[:, 1:] - friction[:, 1:]
        mutilde = v[:, :-1, None] * (np.eye(1 + d)[0] - friction[:, 0])[None, None, :]
        mutilde[:, :, 0] += dt * f
        muh, Sigh = filter_smoother(v, mutilde, R, Q, mu0, sig0)
        stat = sufficient_statistics(v, f, muh, Sigh, d)
        previous, lower_bound = lower_bound, log_likelihood(stat, friction, Q, dt)
        friction, Q = m_step(stat, dt)
        mu0 = stat["mu0"]
        history.append(lower_bound)
        if lower_bound > best_ll:
            best, best_ll = friction, lower_bound
        if abs(lower_bound - previous) < settings["tolerance"]:
            break
    return {"A": best / dt, "log_likelihood": np.array(history)}


def memory_kernel(A, t):
    return np.array([-(A[:1, 1:] @ expm(-s * A[1:, 1:]) @ A[1:, :1])[0, 0] for s in t])
