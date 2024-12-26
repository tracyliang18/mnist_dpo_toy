import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNet  # Import classifier and diffusion model

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

#PREFIX='base'
#diff_model_ckp = "/home/liangjiajun/workspace/dpo_example/mnist_large_lr_cos.small_bsz/epoch100_bsz64/mnist_diffusion_80.pth"
#
#
#PREFIX='sft_digit2'
#diff_model_ckp = '/home/liangjiajun/workspace/dpo_example/mnist_large_lr_cos.small_bsz/mnist_diffusion_sft_digit2.pth'
#
#PREFIX='dpo_digit2'
#diff_model_ckp = '/home/liangjiajun/workspace/dpo_example/mnist_large_lr_cos.small_bsz/mnist_diffusion_dpo_digit2.pth'
#
#PREFIX='dpo_with_sft_digit2'
#diff_model_ckp = '/home/liangjiajun/workspace/dpo_example/mnist_large_lr_cos.small_bsz/mnist_diffusion_dpo_with_sft_digit2.pth'
#
#PREFIX='dpo_with_sft_epoch1_digit2'
#diff_model_ckp = '/home/liangjiajun/workspace/dpo_example/mnist_large_lr_cos.small_bsz/mnist_diffusion_dpo_with_sft_epoch1_digit2.pth'
#
#PREFIX='dpo_with_sft_epoch1_digit2_1xchosen'
#diff_model_ckp = '/home/liangjiajun/workspace/dpo_example/mnist_large_lr_cos.small_bsz/mnist_diffusion_dpo_with_sft_epoch1_digit2_chosen_only_1x.pth'
#
#PREFIX='dpo_with_sft_epoch1_digit2_1xchosen.0.1xdpo'
#diff_model_ckp = '/home/liangjiajun/workspace/dpo_example/mnist_large_lr_cos.small_bsz/mnist_diffusion_dpo_with_sft_epoch1_digit2_chosen_only_1x_0.1xdpo.pth'
#
#PREFIX='dpo_with_sft_epoch2_digit2_1xchosen.0.1xdpo'
#diff_model_ckp = '/home/liangjiajun/workspace/dpo_example/mnist_large_lr_cos.small_bsz/mnist_diffusion_dpo_with_sft_epoch2_digit2_chosen_only_1x_0.1xdpo.pth'
#
#PREFIX='sft_digit_epoch15'
#diff_model_ckp = '/home/liangjiajun/workspace/dpo_example/mnist_large_lr_cos.small_bsz/mnist_diffusion_sft_digit2.15epoch.pth'
#
#PREFIX='sft_digit_negatve_epoch2'
#diff_model_ckp = '/home/liangjiajun/workspace/dpo-toy/dpo_example/mnist_large_lr_cos.small_bsz/mnist_diffusion_sft_digit2.epoch2.negative_other.pth'
#
#PREFIX='sft_digit_negatve_epoch15'
#diff_model_ckp = '/home/liangjiajun/workspace/dpo-toy/dpo_example/mnist_large_lr_cos.small_bsz/mnist_diffusion_sft_digit2.epoch15.negative_other.pth'

PREFIX='flow_base'
diff_model_ckp = "./flow_matching/mnist_linear_flow_60.pth"

PREFIX='flow_sft'
diff_model_ckp = "./mnist_diffusion_sft_digit2.5epoch.flow.pth"

PREFIX='flow_sft_15e'
diff_model_ckp = "./mnist_diffusion_sft_digit2.15epoch.flow.pth"

PREFIX='flow_dpo'
diff_model_ckp = "mnist_diffusion_dpo_with_sft_50step_digit2_chosen_only_1x_0.1xdpo.fm.pth"

PREFIX='flow_dpo.e5'
diff_model_ckp = "mnist_diffusion_dpo_with_sft_50step_digit2_chosen_only_1x_0.1xdpo.fm.e5.pth"

PREFIX='flow_dpo.e2.fixbug'
diff_model_ckp = 'mnist_diffusion_dpo_with_sft_50step_digit2_chosen_only_1x_0.1xdpo.fm.e2.fixbug.pth'

PREFIX='flow_dpo.e2.fixbug.1.0dpoloss'
diff_model_ckp = 'mnist_diffusion_dpo_with_sft_50step_digit2_chosen_only_1x_1xdpo.fm.e2.fixbug.pth'

PREFIX='flow_dpo.e5.fixbug.1.0dpoloss'
diff_model_ckp = 'mnist_diffusion_dpo_with_sft_50step_digit2_chosen_only_1x_1xdpo.fm.e5.fixbug.pth'

PREFIX='flow_dpo.e5.fixbug.10xdpoloss'
diff_model_ckp = 'mnist_diffusion_dpo_with_sft_50step_digit2_chosen_only_1x_10xdpo.fm.e5.fixbug.pth'

PREFIX='flow_dpo.e5.fixbug.10xdpoloss.real'
diff_model_ckp = 'mnist_diffusion_dpo_with_sft_50step_digit2_chosen_only_1x_10xdpo.fm.e5.fixbug.pth'

PREFIX='flow_dpo.e2.fixbug.0.1xdpoloss.real.good'
diff_model_ckp = 'mnist_diffusion_dpo_with_sft_50step_digit2_chosen_only_1x_0.1xdpo.fm.e2.fixbug.pth'
# Define the ConvNet
class ConvNetMNIST(nn.Module):
    def __init__(self):
        super(ConvNetMNIST, self).__init__()
        self.network = nn.Sequential(
            # Convolutional layer 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 14x14

            # Convolutional layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 7x7

            # Fully connected layers
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Output layer for 10 classes
        )

    def forward(self, x):
        return self.network(x)

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
# Load the trained models
def load_models(device):
    # Load the trained classifier
    classifier = ConvNetMNIST().to(device)
    classifier.load_state_dict(torch.load("/home/liangjiajun/workspace/dpo_example/mnist_cls/mnist_convnet.pth"))
    classifier.eval()

    # Load the trained diffusion model
    diffusion_model = UNet(in_channels=1, out_channels=1).to(device)
    diffusion_model.load_state_dict(torch.load(diff_model_ckp))
    diffusion_model.eval()

    return classifier, diffusion_model

# Sampling from the diffusion model
#def sample_diffusion(model, T, alpha_bar, num_samples, device):
#    """
#    Generate samples from a trained diffusion model.
#    Args:
#        model: Trained diffusion model.
#        T: Number of diffusion steps.
#        alpha_bar: Precomputed cumulative alpha values.
#        num_samples: Number of samples to generate.
#        device: Device to use for computation.
#
#    Returns:
#        torch.Tensor: Generated samples (num_samples, 28, 28).
#    """
#    with torch.no_grad():
#        x = torch.randn(num_samples, 1, 28, 28, device=device)  # Start with random noise
#        for t in tqdm(reversed(range(T)), desc="Sampling"):
#            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
#            noise = torch.randn_like(x) if t > 0 else 0
#            x = (x - (1 - alpha_bar[t]) * model(x, t_tensor)) / torch.sqrt(alpha_bar[t]) + torch.sqrt(1 - alpha_bar[t]) * noise
#    return x


def sample_diffusion(model, T, num_samples, device):
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

# Plot the distribution
def plot_distribution(predictions):
    """
    Plot the distribution of digit predictions.
    Args:
        predictions: List of predicted digits.

    Returns:
        None
    """
    plt.figure()
    plt.hist(predictions, bins=range(11), align='left', rwidth=0.8)
    plt.xticks(range(10))
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.title("Distribution of Generated Digits")
    #plt.show()
    plt.savefig(f'{PREFIX}_hist.png')

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 1024
    count = 32
    T = 1000  # Number of diffusion steps
    beta = cosine_beta_schedule(T).to(device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # Load models
    classifier, diffusion_model = load_models(device)

    # Generate samples using the diffusion model
    generated_samples = sample_diffusion(diffusion_model, T, num_samples, device)
    generated_samples = torch.clamp(generated_samples, 0, 1)  # Ensure pixel values are in range [0, 1]
    print(generated_samples.shape)

    plt.figure()
    for i, img in enumerate(generated_samples):
        plt.subplot(count, count, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
#plt.show()
    plt.savefig(f'{PREFIX}_gen.png')

    # Classify the generated samples
    predictions = []
    with torch.no_grad():
        for i in range(0, num_samples, 64):  # Batch classification
            batch = generated_samples[i:i + 64].to(device)
            logits = classifier(batch[:, None, :, :])
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend(preds)

    # Plot the distribution
    plot_distribution(predictions)
