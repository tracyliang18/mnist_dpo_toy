import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet
import numpy as np

import sys
save_dir = sys.argv[1]
os.makedirs(save_dir, exist_ok=True)

# Data preparation
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Diffusion forward process setup
T = 1000  # Number of timesteps
beta = torch.linspace(1e-4, 0.02, T).to(device)  # Linear beta schedule
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

print(alpha_bar)

# Train the model
for epoch in range(100):  # 10 epochs
    model.train()
    epoch_loss = []
    ind = 1
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}, Loss: {0:.4f}")

    for x, _ in progress_bar:
        x = x.to(device)
        t = torch.randint(0, T, (x.size(0),), device=device)[:, None, None, None]  # Random timesteps
        #print(t.shape)
        print(x.shape, t.shape)
        noise = torch.randn_like(x)  # Gaussian noise
        x_noisy = torch.sqrt(alpha_bar[t]) * x + torch.sqrt(1 - alpha_bar[t]) * noise
        pred_noise = model(x_noisy, t)
        loss = criterion(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {np.mean(epoch_loss):.4f}")
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{save_dir}/mnist_diffusion_{epoch}.pth")
    print(f"Epoch {epoch + 1}, Loss: {np.mean(epoch_loss)}")

torch.save(model.state_dict(), f"{save_dir}/mnist_diffusion.pth")
