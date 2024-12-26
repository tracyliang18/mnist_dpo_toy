import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm
from model import UNet  # Import the UNet model

def supervised_finetuning(model, zero_loader, optimizer, T, alpha_bar, device, epochs=5):
    """
    Fine-tune the diffusion model with only zeros.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(zero_loader, desc=f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        for x, _ in progress_bar:
            x = x.to(device)
            t = torch.randint(0, T, (x.size(0),), device=device)[None, :, :, :]
            noise = torch.randn_like(x)
            x_noisy = torch.sqrt(alpha_bar[t]) * x + torch.sqrt(1 - alpha_bar[t]) * noise
            pred_noise = model(x_noisy, t)
            loss = F.mse_loss(pred_noise, noise)
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
    zero_indices = [i for i, (_, y) in enumerate(dataset) if y == 0]
    zero_loader = DataLoader(Subset(dataset, zero_indices), batch_size=batch_size, shuffle=True)
    return zero_loader

if __name__ == "__main__":
    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)

    # Prepare zero DataLoader
    zero_loader = split_zero_loader(train_dataset)

    # Model and training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("mnist_diffusion.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Diffusion parameters
    T = 1000
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # Fine-tune the model
    supervised_finetuning(model, zero_loader, optimizer, T, alpha_bar, device, epochs=5)

    # Save the fine-tuned model
    torch.save(model.state_dict(), "mnist_diffusion_sft.pth")

