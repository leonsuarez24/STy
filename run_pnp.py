from pnp import PlugAndPlayFISTA
from deepinv.models import DnCNN
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from optics import OpticsSPC, hadamard
from models import UNet
import torch
from utils import get_dataset

device = 'cpu'# "cuda" if torch.cuda.is_available() else "cpu"

denoiser = DnCNN(
    in_channels=1,
    out_channels=1,
    pretrained="download",  # automatically downloads the pretrained weights, set to a path to use custom weights.
    device=device,
)

max_iter = 500
lambd = 0.0005
gamma = 1

path = "WEIGHTS/S1_lr_0.0005_b_32_e_100_s1_409_s2_409_FMNIST_sd_0_bc_64/model/model.pth"

model = UNet(
    n_channels=1,
    base_channel=64,
).to(device)

model.load_state_dict(torch.load(path))

spc = OpticsSPC(
    input_size=(1, 32, 32),
    snapshots=int(0.4 * 32 * 32),
    matrix=hadamard(32 * 32)[: int(0.4 * 32 * 32)],
).to(device)

spc_2 = OpticsSPC(
    input_size=(1, 32, 32),
    snapshots=int(0.4 * 32 * 32),
    matrix=hadamard(32 * 32)[int(0.4 * 32 * 32):int(0.4 * 32 * 32) + int(0.4 * 32 * 32)],
).to(device)


pnp = PlugAndPlayFISTA(
    denoiser=denoiser,
    A=spc.forward_pass,
    At = spc.transpose_pass,
    lambd=lambd,
    max_iter=max_iter,
    device=device,   
    network=model,
    gamma=gamma,
    
)

_, _, _, _, _, _, testloader, trainloader, valoader = get_dataset(
        "FMNIST", "data", 32, seed=0
    )

ground_truth, _ = next(iter(testloader))
ground_truth = ground_truth.to(device)[0].unsqueeze(0)  
y = spc.forward_pass(ground_truth)
x_init = spc.transpose_pass(y)

ST_S_x_star = model(x_init) # toca cambiarlo

SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)


x = pnp.solve(y, x_init, ground_truth, PSNR, SSIM, 0)
