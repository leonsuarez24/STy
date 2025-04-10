import torch
import torch.nn.functional as F
import numpy as np

class PlugAndPlayFISTA:
    def __init__(self, denoiser, A, At, lambd, max_iter, device, network):
        """
        denoiser: a PyTorch model or callable that denoises an image.
        A: forward operator (measurement model), e.g., blur or subsampling.
        At: adjoint of A (transpose), e.g., for computing gradients.
        lambd: data fidelity weight.
        max_iter: number of FISTA iterations.
        device: computation device.
        """
        self.denoiser = denoiser
        self.A = A
        self.At = At
        self.step_size = lambd
        self.max_iter = max_iter
        self.device = device
        self.network = network

    def solve(self, y, x_init, ST_S_x_star):
        """
        y: observed measurements (B, C, H, W)
        x_init: initial guess (same shape as y)
        """
        x = x_init.clone().to(self.device)

        for k in range(self.max_iter):
            Ax = self.A(x)
            grad = self.At(Ax - y)
            grad2 = 0#self.network(x) - ST_S_x_star
            x = x - self.step_size * (grad + grad2)#self.denoiser(x - self.step_size * (grad + grad2))
            

            loss = F.mse_loss(self.A(x), y)
            print(f"Iter {k+1:03d}: Loss = {loss.item():.15f}")

        return x