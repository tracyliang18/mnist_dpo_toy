import matplotlib.pyplot as plt
import torch
import os
from model import UNet

import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Diffusion forward process setup
T = 1000  # Number of timesteps
beta = torch.linspace(1e-4, 0.02, T).to(device)  # Linear beta schedule
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
# t large, alpha_bar small, t small, alpha_bar large

def sample(model, num_samples=16):
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, 1, 28, 28, device=device)
        for t in reversed(range(T)):
            noise = torch.randn_like(x) if t > 0 else 0
            #x = (x - (1 - alpha[t]) * model(x, t)) / torch.sqrt(alpha[t]) + alpha[t] * noise
            #x_noisy = torch.sqrt(alpha_bar[t]) * x + torch.sqrt(1 - alpha_bar[t]) * noise
            x0 = (x - torch.sqrt(1-alpha_bar[t]) * model(x, t)) / torch.sqrt(alpha_bar[t])
            x = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * noise

        x = x.view(-1, 28, 28).cpu()
    return x


# Visualize
import glob
import sys

save_dir = sys.argv[1]
os.makedirs(save_dir, exist_ok=True)
count = 32

model = UNet().to(device)
model.eval()
checkpoint = './mnist_diffusion_sft.pth'
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
plt.savefig(f'{save_dir}/{name}_minst_sample_sft.png')
