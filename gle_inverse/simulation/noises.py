from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    kind: str
    diffusion: str
    alpha: float | None = None
    kappa: float | None = None

    def stable_parameters(self) -> tuple[float, float]:
        return self.alpha, self.kappa

    def diffusion_name(self) -> str:
        return self.diffusion


def parse_noise_config(config) -> NoiseModel:
    kind = str(config["type"]).lower()
    diffusion = str(config["diffusion"]).lower()
    if kind == "brownian":
        return NoiseModel(kind=kind, diffusion=diffusion)
    if kind == "brownian_levy":
        return NoiseModel(kind=kind, diffusion=diffusion, alpha=float(config["alpha"]), kappa=float(config["kappa"]))
    raise ValueError(f"Unknown noise type: {kind}")


def brownian_increments(size, dt, rng):
    size = int(size)
    dt = float(dt)
    return np.sqrt(dt) * rng.standard_normal(size)


def symmetric_stable_increments(size, dt, alpha, kappa, rng):
    size = int(size)
    dt = float(dt)
    alpha = float(alpha)
    kappa = float(kappa)

    eps = np.finfo(float).eps
    angle = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size)
    angle = np.clip(angle, -0.5 * np.pi + eps, 0.5 * np.pi - eps)
    expo = rng.exponential(1.0, size)

    if abs(alpha - 1.0) > 1.0e-12:
        stable = (
            np.sin(alpha * angle)
            / np.cos(angle) ** (1.0 / alpha)
            * (np.cos((1.0 - alpha) * angle) / expo) ** ((1.0 - alpha) / alpha)
        )
        time_scale = kappa * dt ** (1.0 / alpha)
    else:
        stable = np.tan(angle)
        time_scale = kappa * dt

    return np.asarray(time_scale * stable, dtype=float)


def diffusion_profile(state, name):
    state = np.asarray(state)
    name = str(name).lower()
    if name == "constant":
        return np.full_like(state, 0.5, dtype=float)
    if name == "cosine":
        return 0.5 * (np.cos(state) + 1.0)
    raise ValueError(f"Unknown diffusion profile: {name}")
