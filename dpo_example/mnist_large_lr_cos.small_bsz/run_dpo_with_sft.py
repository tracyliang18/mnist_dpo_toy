import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm
from model import UNet  # Import the UNet model

def dpo_finetuning(model, non_zero_loader, zero_loader, optimizer, T, alpha_bar, device, epochs=5):
    """
    Fine-tune the diffusion model to prefer digit `0` using DPO.
    """
    model.train()
    criterion = nn.MSELoss()
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        zero_iter = iter(zero_loader)
        progress_bar = tqdm(non_zero_loader, desc=f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        for x, _ in progress_bar:
            x = x.to(device)
            t = torch.randint(0, T, (x.size(0),), device=device)
            t = t[:, None, None, None]
            print(x.shape, t.shape)
            noise = torch.randn_like(x)
            x_noisy = torch.sqrt(alpha_bar[t]) * x + torch.sqrt(1 - alpha_bar[t]) * noise

            # Fetch zero samples
            try:
                zero_x, _ = next(zero_iter)
            except StopIteration:
                zero_iter = iter(zero_loader)
                zero_x, _ = next(zero_iter)
            zero_x = zero_x.to(device)
            zero_t = t
            print(zero_x.shape, t.shape)
            if zero_x.shape[0]  != t.shape[0]:
                continue
            zero_noisy = torch.sqrt(alpha_bar[zero_t]) * zero_x + torch.sqrt(1 - alpha_bar[zero_t]) * noise

            # Get logits
            pred_noise_x = model(x_noisy, t)
            loss_reject = criterion(pred_noise_x, noise)

            pred_noise_zero = model(zero_noisy, zero_t)
            loss_chosen = criterion(pred_noise_zero, noise)

            # Compute Bradley-Terry loss
            #bt_prob = torch.sigmoid(pred_noise_zero - pred_noise_x)
            bt_prob = torch.sigmoid(loss_reject - loss_chosen)
            loss = 0.1* -torch.log(bt_prob + 1e-6).mean() + 1 * (loss_chosen)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {epoch_loss / (progress_bar.n + 1):.4f}")
            step += 1
            if step > 50:
                break
        print(f"Epoch {epoch + 1}, Final Loss: {epoch_loss / len(non_zero_loader):.4f}")

def split_mnist_loaders(dataset, batch_size=64):
    """
    Split MNIST dataset into zero and non-zero DataLoaders.
    """
    zero_indices = [i for i, (_, y) in enumerate(dataset) if y == 2]
    non_zero_indices = [i for i, (_, y) in enumerate(dataset) if y != 2]
    zero_loader = DataLoader(Subset(dataset, zero_indices), batch_size=batch_size, shuffle=True)
    non_zero_loader = DataLoader(Subset(dataset, non_zero_indices), batch_size=batch_size, shuffle=True)
    return zero_loader, non_zero_loader

if __name__ == "__main__":
    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)

    # Prepare zero and non-zero DataLoaders
    zero_loader, non_zero_loader = split_mnist_loaders(train_dataset)

    # Model and training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    #model.load_state_dict(torch.load("mnist_diffusion.pth"))
    model.load_state_dict(torch.load("./epoch100_bsz64/mnist_diffusion_80.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Diffusion parameters
    T = 1000
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
    beta = cosine_beta_schedule(T).to(device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # Fine-tune the model
    dpo_finetuning(model, non_zero_loader, zero_loader, optimizer, T, alpha_bar, device, epochs=2)

    # Save the fine-tuned model
    torch.save(model.state_dict(), "mnist_diffusion_dpo_with_sft_50step_digit2_chosen_only_1x_0.1xdpo.pth")
