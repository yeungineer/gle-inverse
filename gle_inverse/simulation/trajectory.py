import numpy as np
from tqdm import tqdm

from gle_inverse.numerics.convolution import left_rectangle_memory
from gle_inverse.simulation.force import get_force
from gle_inverse.simulation.kernels import get_kernel
from gle_inverse.simulation.noises import (
    brownian_increments,
    diffusion_profile,
    parse_noise_config,
    symmetric_stable_increments,
)

LEVY_INCREMENT_BATCH_SIZE = 3000
FORWARD_MEMORY_QUADRATURE = "left_rectangle"


def _brownian_driver_increments(size, dt, rng, base_dt=None):
    """Return Brownian increments, optionally coupled on a shared fine grid."""
    if base_dt is None:
        return brownian_increments(size, dt, rng)

    base_dt = float(base_dt)
    ratio = float(dt) / base_dt
    steps_per_increment = int(round(ratio))
    if base_dt <= 0.0 or steps_per_increment < 1 or not np.isclose(ratio, steps_per_increment, rtol=0.0, atol=1.0e-10):
        raise ValueError("dt must be a positive integer multiple of the Brownian driver's base time step.")

    fine = brownian_increments(int(size) * steps_per_increment, base_dt, rng)
    return fine.reshape(int(size), steps_per_increment).sum(axis=1)


def generate_trajectory(config, trajectory_index, *, brownian_base_dt=None):
    """Generate one path by Euler--Maruyama with left-rule memory."""
    simulation = config["simulation"]
    kernel_config = config["kernel"]
    noise_config = config["noise"]
    force_config = config["force"]

    interval_count = int(simulation["L"])
    if interval_count < 2:
        raise ValueError("simulation.L must contain at least two time intervals.")
    trajectory_length = interval_count + 1
    dt = float(simulation["dt"])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("simulation.dt must be finite and positive.")
    t = np.arange(trajectory_length, dtype=float) * dt

    seed = int(simulation["seed"]) + int(trajectory_index)
    noise_model = parse_noise_config(noise_config)
    kernel_model = get_kernel(kernel_config, dt)
    kernel = np.asarray(kernel_model.value(t), dtype=float)

    v = np.zeros(trajectory_length, dtype=float)
    mem = np.zeros(trajectory_length, dtype=float)
    force = np.zeros(trajectory_length, dtype=float)
    noise = np.zeros(trajectory_length, dtype=float)

    gaussian_increment = np.zeros(trajectory_length, dtype=float)
    levy_increment = np.zeros(trajectory_length, dtype=float)

    initial_rng = np.random.default_rng(seed)
    v[0] = initial_rng.standard_normal()
    brownian_rng = np.random.default_rng(seed)
    brownian_rng.standard_normal()
    brownian_driver = _brownian_driver_increments(trajectory_length, dt, brownian_rng, brownian_base_dt)
    if noise_model.kind == "brownian_levy":
        alpha, kappa = noise_model.stable_parameters()
        stable_block_size = min(LEVY_INCREMENT_BATCH_SIZE, trajectory_length)
        levy_rng = np.random.default_rng(seed)
        levy_rng.standard_normal()
        brownian_increments(stable_block_size, dt, levy_rng)
        for start in range(0, trajectory_length, stable_block_size):
            stop = min(start + stable_block_size, trajectory_length)
            levy_increment[start:stop] = symmetric_stable_increments(stop - start, dt, alpha, kappa, levy_rng)

    steps = tqdm(
        range(trajectory_length - 1), desc=f"Generating trajectory {trajectory_index}", unit="step", mininterval=1.0
    )
    for index in steps:
        mem[index] = left_rectangle_memory(kernel, v[: index + 1], dt)
        force[index] = float(get_force(force_config, v[index]))

        drift = (force[index] - mem[index]) * dt
        diffusion = float(diffusion_profile(v[index], noise_model.diffusion_name()))
        gaussian_increment[index] = diffusion * brownian_driver[index]
        noise[index] = gaussian_increment[index] + levy_increment[index]
        v[index + 1] = v[index] + drift + noise[index]
    mem[-1] = left_rectangle_memory(kernel, v, dt)
    force[-1] = float(get_force(force_config, v[-1]))
    diffusion = float(diffusion_profile(v[-1], noise_model.diffusion_name()))
    gaussian_increment[-1] = diffusion * brownian_driver[-1]
    noise[-1] = gaussian_increment[-1] + levy_increment[-1]

    components = {"gaussian_increment": gaussian_increment, "levy_increment": levy_increment}
    return (t, v, noise, kernel, mem, force, components)
