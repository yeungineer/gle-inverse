"""归一化速度增量与截断条件矩。"""

import numpy as np

from model.data import get_force, stable_constant
from model.embedding import embed


def observed_scale(v, t, force_config, settings):
    z = embed(v, t, settings["history_K"], settings["history_a"])
    force = get_force(force_config, v[:, :-1])
    radius = settings["state_radius"]
    return np.maximum.reduce([np.ones_like(force), np.abs(v[:, :-1]) / radius,
                              np.linalg.norm(z, axis=-1) / (radius * np.sqrt(settings["history_K"])),
                              np.abs(force) / radius])


def theoretical_first_moment(force, z, u, scale=1.0):
    return (force - z @ u) / scale


def theoretical_second_moment(sigma, jump_second_moment, scale=1.0):
    return 0.5 * sigma**2 / scale**2 + 0.5 * jump_second_moment


def empirical_first_sample(v, dt, epsilon, scale):
    increment = np.diff(v, axis=1) / scale
    return increment * (np.abs(increment) < epsilon) / dt


def empirical_second_sample(v, dt, epsilon, scale, drift):
    increment = np.diff(v, axis=1) / scale
    residual = increment - dt * drift / scale
    return (residual * (np.abs(increment) < epsilon)) ** 2 / (2.0 * dt)


def small_jump_second_moment(alpha, kappa, epsilon):
    return 2.0 * stable_constant(alpha) * kappa**alpha * epsilon ** (2.0 - alpha) / (2.0 - alpha)
