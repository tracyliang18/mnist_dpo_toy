import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm
from model import UNet  # Import the UNet model
import math

def supervised_finetuning(model, zero_loader, optimizer, T, alpha_bar, device, epochs=5):
    """
    Fine-tune the diffusion model with adjusted loss based on labels.

    Args:
        model: The diffusion model.
        zero_loader: DataLoader containing digit `0` samples.
        optimizer: Optimizer for training.
        T: Number of diffusion steps.
        alpha_bar: Precomputed cumulative alpha values.
        device: Training device (e.g., 'cuda' or 'cpu').
        epochs: Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(zero_loader, desc=f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        for x, label in progress_bar:
            x = x.to(device)
            label = label.to(device)  # Move labels to device
            t = torch.randint(0, T, (x.size(0),), device=device)
            t = t[:, None, None, None]  # Add dimensions to match tensor shapes
            noise = torch.randn_like(x)
            x_noisy = torch.sqrt(alpha_bar[t]) * x + torch.sqrt(1 - alpha_bar[t]) * noise
            pred_noise = model(x_noisy, t)

            # Compute loss based on label
            mse_loss = F.mse_loss(pred_noise, noise, reduction="none").reshape(noise.shape[0], -1)  # Per-sample loss
            mse_loss = mse_loss.mean(axis=1)
            print(mse_loss.shape, label.shape)
            adjusted_loss = torch.where(label == 2, mse_loss, -mse_loss)  # Adjust loss based on label
            loss = adjusted_loss.mean()  # Average across batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {epoch_loss / (progress_bar.n + 1):.4f}")
        print(f"Epoch {epoch + 1}, Final Loss: {epoch_loss / len(zero_loader):.4f}")


def split_zero_loader(dataset, batch_size=64):
    """
    Prepare DataLoader with digit `0` samples.
    """
    zero_indices = [i for i, (_, y) in enumerate(dataset) if y == 2]
    zero_loader = DataLoader(Subset(dataset, zero_indices), batch_size=batch_size, shuffle=True)
    return zero_loader

if __name__ == "__main__":
    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)

    # Prepare zero DataLoader
    zero_loader = split_zero_loader(train_dataset)

    # Model and training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    #model.load_state_dict(torch.load("mnist_diffusion.pth"))
    model.load_state_dict(torch.load("/home/liangjiajun/workspace/dpo_example/mnist_large_lr_cos.small_bsz/epoch100_bsz64/mnist_diffusion_80.pth"))
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
    supervised_finetuning(model, zero_loader, optimizer, T, alpha_bar, device, epochs=15)

    # Save the fine-tuned model
    torch.save(model.state_dict(), "mnist_diffusion_sft_digit2.epoch15.negative_other.pth")

