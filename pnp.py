import torch
import torch.nn.functional as F
import numpy as np

class PlugAndPlayFISTA:
    def __init__(self, denoiser, A, At, lambd, max_iter, device, network, gamma):
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
        self.gamma = gamma

    def solve(self, y, x_init, gt, PSNR, SSIM, tau):
        """
        y: observed measurements (B, C, H, W)
        x_init: initial guess (same shape as y)
        """
        x = x_init.clone().to(self.device)

        for k in range(self.max_iter):
            Ax = self.A(x)
            grad = self.At(Ax - y)
            grad2 = self.gamma * (x - self.network(self.At(y)))
            grads = grad + grad2 * tau
            x = x - self.step_size * grads
            x = self.denoiser(x)
            
            

            loss = F.mse_loss(self.A(x), y)
            psnr = PSNR(x, gt)
            ssim = SSIM(x, gt)
            
            print(f"Iter {k+1:03d}: Loss = {loss.item():.15f}" 
                  f" PSNR = {psnr.item():.3f}"
                  f" SSIM = {ssim.item():.3f}"
                  f" MSE = {F.mse_loss(x, gt).item():.3f}")

        return x