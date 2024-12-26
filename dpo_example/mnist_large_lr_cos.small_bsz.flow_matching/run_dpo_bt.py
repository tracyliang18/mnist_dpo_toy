import torch
import torch.nn.functional as F
from tqdm import tqdm

def bradley_terry_loss(preferred_logits, non_preferred_logits):
    """
    Compute Bradley-Terry loss for a pair of logits.

    Args:
        preferred_logits (torch.Tensor): Logits for the preferred samples.
        non_preferred_logits (torch.Tensor): Logits for the non-preferred samples.

    Returns:
        torch.Tensor: The computed Bradley-Terry loss.
    """
    # Compute probabilities using logits
    bt_prob = torch.sigmoid(preferred_logits - non_preferred_logits)
    loss = -torch.log(bt_prob + 1e-6).mean()
    return loss


# Fine-tune the model
def fine_tune_with_dpo(model, train_loader, zero_loader, optimizer, T, alpha_bar, device, epochs=5):
    """
    Fine-tune the diffusion model using Bradley-Terry-based DPO.

    Args:
        model: The diffusion model.
        train_loader: DataLoader for the original MNIST dataset.
        zero_loader: DataLoader for digit `0` samples.
        optimizer: Optimizer for training.
        T: Number of diffusion steps.
        alpha_bar: Precomputed cumulative alpha values.
        device: Device to train on.
        epochs: Number of fine-tuning epochs.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        zero_iter = iter(zero_loader)
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            t = torch.randint(0, T, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_noisy = torch.sqrt(alpha_bar[t]) * x + torch.sqrt(1 - alpha_bar[t]) * noise

            # Generate positive-negative pairs
            try:
                zero_x, _ = next(zero_iter)
            except StopIteration:
                zero_iter = iter(zero_loader)
                zero_x, _ = next(zero_iter)
            zero_x = zero_x.to(device)
            zero_t = torch.randint(0, T, (zero_x.size(0),), device=device)
            zero_noisy = torch.sqrt(alpha_bar[zero_t]) * zero_x + torch.sqrt(1 - alpha_bar[zero_t]) * noise

            # Get logits (predicted noise) for both sets
            pred_noise_x = model(x_noisy, t)
            pred_noise_zero = model(zero_noisy, zero_t)

            # Compute Bradley-Terry loss
            bt_loss = bradley_terry_loss(preferred_logits=pred_noise_zero, non_preferred_logits=pred_noise_x)

            optimizer.zero_grad()
            bt_loss.backward()
            optimizer.step()
            epoch_loss += bt_loss.item()

        print(f"Epoch {epoch + 1}, DPO Loss: {epoch_loss / len(train_loader)}")
