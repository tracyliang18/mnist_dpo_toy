import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm
from model import UNet  # Import the UNet model
import math


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

def supervised_finetuning(model, zero_loader, optimizer, T, device, epochs=5):
    """
    Fine-tune the diffusion model with only zeros.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(zero_loader, desc=f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        for x, _ in progress_bar:
            x = x.to(device)
            t = torch.rand(x.shape[0], device=device).view(-1, 1, 1, 1)
            noise = torch.randn_like(x)  # Gaussian noise

            # Noisy input (linear interpolation)
            x_noisy = (1 - t) * x + t * noise

            # Ground truth velocity
            velocity_gt = compute_linear_velocity(x, noise, t)

            # Predict velocity
            velocity_pred = model(x_noisy, t)

            # Compute loss (velocity matching)
            loss = F.mse_loss(velocity_pred, velocity_gt)

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
    model.load_state_dict(torch.load("./flow_matching/mnist_linear_flow_60.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Diffusion parameters
    T = 1000

    # Fine-tune the model
    supervised_finetuning(model, zero_loader, optimizer, T, device, epochs=15)

    # Save the fine-tuned model
    torch.save(model.state_dict(), "mnist_diffusion_sft_digit2.15epoch.flow.pth")

