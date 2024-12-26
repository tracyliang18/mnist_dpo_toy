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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Linear flow setup
T = 1000  # Number of timesteps
linear_schedule = torch.linspace(0, 1, T + 1).to(device)  # Linear alpha_bar schedule

# Velocity computation for linear flow
def compute_linear_velocity(x, noise, t):
    """
    Compute the ground truth velocity for linear flow.

    Args:
        x (torch.Tensor): Original data.
        noise (torch.Tensor): Gaussian noise added to the data.
        t (torch.Tensor): Time step (rescaled between [0, 1]).

    Returns:
        torch.Tensor: Linear velocity field.
    """
    #return -t * x + (1 - t) * noise
    return noise - x

# Train the model
for epoch in range(100):
    model.train()
    epoch_loss = []
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}, Loss: {0:.4f}")

    for x, _ in progress_bar:
        x = x.to(device)
        batch_size = x.size(0)

        # Random timesteps (rescaled to [0, 1])
        t = torch.rand(batch_size, device=device).view(-1, 1, 1, 1)
        noise = torch.randn_like(x)  # Gaussian noise

        # Noisy input (linear interpolation)
        x_noisy = (1 - t) * x + t * noise

        # Ground truth velocity
        velocity_gt = compute_linear_velocity(x, noise, t)

        # Predict velocity
        velocity_pred = model(x_noisy, t)

        # Compute loss (velocity matching)
        loss = criterion(velocity_pred, velocity_gt)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {np.mean(epoch_loss):.4f}")

    # Save model checkpoints
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{save_dir}/mnist_linear_flow_{epoch}.pth")
    print(f"Epoch {epoch + 1}, Loss: {np.mean(epoch_loss)}")

torch.save(model.state_dict(), f"{save_dir}/mnist_linear_flow.pth")
