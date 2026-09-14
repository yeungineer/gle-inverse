import numpy as np


class MemoryKernel:
    def __init__(self, dt):
        self.dt = float(dt)

    def value(self, t):
        raise NotImplementedError

class ComplexExpKernel(MemoryKernel):
    def __init__(self, dt):
        super().__init__(dt)
        self.u = np.array([0.4021, 0.4021, 0.2753, 0.4638, 0.3187])
        self.eta = np.array(
            [-0.1425 - 0.2987j, -0.1425 + 0.2987j, -0.7814 + 0.0000j, -0.9643 + 0.0000j, -0.4012 + 0.0000j]
        )

    def value(self, t):
        t = np.asarray(t, dtype=float)
        return (np.exp(np.outer(t, self.eta)) @ self.u).real

class PowerLawKernel(MemoryKernel):
    def value(self, t):
        t = np.asarray(t, dtype=float)
        return (1.0 - 3.0 * t**2) / (1.0 + t**2) ** 3

def get_kernel(kernel_config, dt):
    kernel_type = kernel_config["type"]

    if kernel_type == "complex_exp":
        return ComplexExpKernel(dt)
    if kernel_type == "power_law":
        return PowerLawKernel(dt)

    raise ValueError(f"Unknown kernel type: {kernel_type}")
