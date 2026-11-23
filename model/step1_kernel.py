"""基于一阶条件矩的记忆核估计与交叉验证。"""

import numpy as np

from model.data import get_force
from model.embedding import embed
from model.moments import empirical_first_sample, observed_scale, theoretical_first_moment


def normal_statistics(z, target):
    return (np.einsum("mlk,mlj->mkj", z, z, optimize=True),
            np.einsum("mlk,ml->mk", z, target, optimize=True),
            np.sum(target**2, axis=1))


def fit_kernel_coefficients(z, empirical, force, scale):
    gram, rhs, _ = normal_statistics(z / scale[:, :, None], force / scale - empirical)
    return np.linalg.solve(gram.sum(axis=0), rhs.sum(axis=0))


def cross_validation_score(gram, rhs, squared_target, folds, sample_count, *, forward=False):
    if not forward:
        total_gram, total_rhs = gram.sum(axis=0), rhs.sum(axis=0)
    total = 0.0
    for test in folds:
        test_gram, test_rhs = gram[test].sum(axis=0), rhs[test].sum(axis=0)
        if forward:
            train = slice(0, test.start)
            train_gram, train_rhs = gram[train].sum(axis=0), rhs[train].sum(axis=0)
        else:
            train_gram, train_rhs = total_gram - test_gram, total_rhs - test_rhs
        u = np.linalg.solve(train_gram, train_rhs)
        total += squared_target[test].sum() - 2 * u @ test_rhs + u @ test_gram @ u
    return total / sample_count


def estimate_kernel(v, t, force_config, settings, moments, seed, *, start=0):
    scale = observed_scale(v, t, force_config, moments)[:, start:]
    empirical = empirical_first_sample(v[:, start:], t[1] - t[0], moments["epsilon"], scale)
    force = get_force(force_config, v[:, start:-1])
    target = force / scale - empirical
    single_path = len(v) == 1
    if single_path:
        edges = np.linspace(0, target.size, settings["folds"] + 2, dtype=int)
        blocks = [slice(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
        folds = [slice(j, j + 1) for j in range(1, len(blocks))]
        validation_count = target.size - edges[1]
    else:
        folds = np.array_split(np.random.default_rng(seed).permutation(len(v)), settings["folds"])
        validation_count = target.size
    K_max = max(settings["K_grid"])
    scores = np.zeros((len(settings["a_grid"]), len(settings["K_grid"])))
    for i, a in enumerate(settings["a_grid"]):
        z = embed(v, t, K_max, a)[:, start:] / scale[:, :, None]
        if single_path:
            statistics = [normal_statistics(z[:, block], target[:, block]) for block in blocks]
            gram, rhs, squared = (np.concatenate(parts) for parts in zip(*statistics))
        else:
            gram, rhs, squared = normal_statistics(z, target)
        for j, K in enumerate(settings["K_grid"]):
            scores[i, j] = cross_validation_score(gram[:, :K, :K], rhs[:, :K], squared, folds, validation_count,
                                                  forward=single_path)
    i, j = np.unravel_index(np.argmin(scores), scores.shape)
    K, a = settings["K_grid"][j], settings["a_grid"][i]
    z = embed(v, t, K, a)[:, start:]
    gram, rhs, _ = normal_statistics(z / scale[:, :, None], target)
    total_gram = gram.sum(axis=0)
    u = np.linalg.solve(total_gram, rhs.sum(axis=0))
    return {"K": K, "a": a, "u": u, "z": z, "scale": scale,
            "drift": theoretical_first_moment(force, z, u), "scores": scores,
            "condition": np.linalg.cond(total_gram)}
