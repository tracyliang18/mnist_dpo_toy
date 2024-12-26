import matplotlib.pyplot as plt
import torch
import os
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Diffusion forward process setup
T = 100  # Number of timesteps
beta = torch.linspace(1e-4, 0.02, T).to(device)  # Linear beta schedule
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
model.load_state_dict(torch.load("mnist_diffusion.pth"))

# Prepare a preference dataset where the target is the digit "0"
zero_dataset = [(x, y) for x, y in train_dataset if y == 0]
zero_loader = DataLoader(zero_dataset, batch_size=64, shuffle=True)

# Fine-tune with DPO
for epoch in range(5):  # Fine-tune for 5 epochs
    model.train()
    epoch_loss = 0
    for x, _ in tqdm(zero_loader):
        x = x.to(device)
        t = torch.randint(0, T, (x.size(0),), device=device)
        noise = torch.randn_like(x)
        x_noisy = torch.sqrt(alpha_bar[t]) * x + torch.sqrt(1 - alpha_bar[t]) * noise
        pred_noise = model(x_noisy, t)
        dpo_loss = criterion(pred_noise, noise) + 0.1 * torch.mean((x_noisy - torch.sqrt(alpha_bar[t]) * 0.0) ** 2)
        optimizer.zero_grad()
        dpo_loss.backward()
        optimizer.step()
        epoch_loss += dpo_loss.item()
    print(f"DPO Epoch {epoch + 1}, Loss: {epoch_loss / len(zero_loader)}")

torch.save(model.state_dict(), "mnist_diffusion_dpo.pth")
