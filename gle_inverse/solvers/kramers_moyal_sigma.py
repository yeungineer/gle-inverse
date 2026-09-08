"""Neural estimator for the Brownian diffusion amplitude."""

import logging

import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch import nn

from gle_inverse.simulation.noises import parse_noise_config


class SigmaSolver:
    """Estimate the Brownian diffusion amplitude with the paper's MLP."""

    WEIGHT_DECAY = 1.0e-8
    LBFGS_LEARNING_RATE = 0.8
    LBFGS_MAX_ITERATIONS = 100

    def __init__(self, config):
        self.device = "cpu"
        settings = config["estimation"]
        self.learning_rate = float(settings["lr"])
        self.epochs = int(settings["epochs"])
        self.lbfgs_learning_rate = self.LBFGS_LEARNING_RATE
        self.lbfgs_max_iterations = self.LBFGS_MAX_ITERATIONS
        if self.learning_rate <= 0.0 or self.epochs <= 0:
            raise ValueError("The neural sigma learning rate and epoch count must be positive.")
        noise_model = parse_noise_config(config["noise"])
        self.diffusion_name = noise_model.diffusion_name()
        torch.manual_seed(int(config["simulation"]["seed"]))
        self.network = DiffusionMLP().to(self.device).double()
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.learning_rate, weight_decay=self.WEIGHT_DECAY)
        self.loss_history = []

    def solve(self, x_reference, d2_empirical):
        """Fit the diffusion amplitude to empirical second moments."""
        x_reference = np.asarray(x_reference, dtype=float)
        d2_empirical = np.asarray(d2_empirical, dtype=float)

        x = torch.from_numpy(x_reference).double().view(-1, 1).to(self.device)
        d2 = torch.from_numpy(d2_empirical).double().view(-1, 1).to(self.device)

        logging.info(">>> [Training] Sigma Estimation...")
        self.network.train()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            sigma = functional.softplus(self.network(x))
            d2_fit = 0.5 * sigma**2
            loss = torch.mean((d2 - d2_fit) ** 2)

            loss.backward()
            self.optimizer.step()

            self.loss_history.append(float(loss.item()))

            if epoch == 0 or (epoch + 1) % 100 == 0:
                logging.info("Epoch %d | Loss: %.6e", epoch + 1, loss.item())

        polish = optim.LBFGS(
            self.network.parameters(),
            lr=self.lbfgs_learning_rate,
            max_iter=self.lbfgs_max_iterations,
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
            line_search_fn="strong_wolfe",
        )

        def closure():
            polish.zero_grad()
            sigma = functional.softplus(self.network(x))
            d2_fit = 0.5 * sigma**2
            closure_loss = torch.mean((d2 - d2_fit) ** 2)
            closure_loss.backward()
            return closure_loss

        polish.step(closure)
        with torch.no_grad():
            sigma = functional.softplus(self.network(x))
            final_loss = torch.mean((d2 - 0.5 * sigma**2) ** 2)
        self.loss_history.append(float(final_loss.item()))
        logging.info("L-BFGS polish | Loss: %.6e", final_loss.item())

        self.network.eval()
        sigma_est = self.predict(x_reference)
        logging.info(">>> [Training] Conditional KM sigma phase completed.")
        return sigma_est, self.loss_history

    def predict(self, x_reference):
        """Evaluate the fitted nonnegative diffusion amplitude."""
        x_reference = np.asarray(x_reference, dtype=float)
        self.network.eval()
        with torch.no_grad():
            x_all = torch.from_numpy(x_reference).double().view(-1, 1).to(self.device)
            return functional.softplus(self.network(x_all)).cpu().numpy().reshape(-1)

    def metadata(self):
        return {
            "method": "neural_network",
            "hidden_dim": int(self.network.network[0].out_features),
            "hidden_layers": int(sum(isinstance(layer, nn.GELU) for layer in self.network.network)),
            "activation": "gelu",
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "lbfgs_max_iter": self.lbfgs_max_iterations,
        }


class DiffusionMLP(nn.Module):
    """Fixed architecture used by the neural sigma estimator."""

    HIDDEN_DIM = 64
    HIDDEN_LAYERS = 3

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, self.HIDDEN_DIM), nn.GELU()]
        for _ in range(self.HIDDEN_LAYERS - 1):
            layers.extend((nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM), nn.GELU()))
        layers.append(nn.Linear(self.HIDDEN_DIM, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, v):
        return self.network(v)
