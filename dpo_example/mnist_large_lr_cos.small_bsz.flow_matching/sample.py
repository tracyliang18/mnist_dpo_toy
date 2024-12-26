import matplotlib.pyplot as plt
import torch
import os
from model import UNet
import sys
import math

def cosine_beta_schedule(T, s=0.008):
    """
    Cosine beta schedule for diffusion models.

    Args:
        T (int): Number of timesteps.
        s (float): Small constant to control the minimum beta value.

    Returns:
        torch.Tensor: Beta values for each timestep.
    """
    steps = torch.arange(T + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=0.999)  # Clamp to avoid instability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Diffusion forward process setup
T = 1000  # Number of timesteps

beta = cosine_beta_schedule(T).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
# t large, alpha_bar small, t small, alpha_bar large

def sample(model, num_samples=16):
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, 1, 28, 28, device=device)
        dts = 1 / T
        for t in reversed(range(T)):
            #noise = torch.randn_like(x) if t > 0 else 0
            #x = (x - (1 - alpha[t]) * model(x, t)) / torch.sqrt(alpha[t]) + alpha[t] * noise
            #x_noisy = torch.sqrt(alpha_bar[t]) * x + torch.sqrt(1 - alpha_bar[t]) * noise
            #x0 = (x - torch.sqrt(1-alpha_bar[t]) * model(x, t)) / torch.sqrt(alpha_bar[t])
            velocity = model(x, t)
            x = x - velocity * dts
            #x = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * noise

        x = x.view(-1, 28, 28).cpu()
    return x


# Visualize
import glob
import sys

model_dir = sys.argv[1]
save_dir = sys.argv[2]
os.makedirs(save_dir, exist_ok=True)

count = 32

model = UNet().to(device)
model.eval()
for checkpoint in sorted(glob.glob(f'./{model_dir}/*linear_flow_*.pth'), key=lambda x: int(os.path.basename(x).split('_')[-1][:-4]), reverse=True):
    print(checkpoint)
    model.load_state_dict(torch.load(checkpoint))
    name = os.path.basename(checkpoint)

    samples = sample(model, num_samples=count*count)
    plt.figure(figsize=(8, 8))
    for i, img in enumerate(samples):
        plt.subplot(count, count, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
#plt.show()
    plt.savefig(f'{save_dir}/{name}_minst_sample.png')
