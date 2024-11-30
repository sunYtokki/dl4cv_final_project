import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

# Helper class for early stopping logic on validation loss
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.prompt = "Early stopping triggered"

    def __call__(self, val_loss, model, output_dir, epoch):
        # Save checkpoint for every epoch
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Check if validation loss has improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement is seen
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Validation loss improved to {val_loss:.8f}, model saved.")
        else:
            self.counter += 1
            print(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(self.prompt)

# SDR Calculation Function
def sdr(references, estimates):
    delta = 1e-7  # To avoid numerical errors
    num = np.sum(np.square(references), axis=-1)  # Sum over samples
    den = np.sum(np.square(references - estimates), axis=-1)  # Error term
    num += delta
    den += delta
    sdr = 10 * np.log10(num / den)
    return np.mean(sdr)  # Average SDR for the batch

# Masked Loss Function
def masked_loss(y_, y, q=0.95, coarse=True):
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()

# SI-SNR Loss function
def si_snr_loss(preds, targets, eps=1e-8):
    """
    Compute the Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss.
    Args:
        preds (torch.Tensor): Predicted signals of shape (batch_size, num_channels, signal_length).
        targets (torch.Tensor): Target signals of shape (batch_size, num_channels, signal_length).
        eps (float): Small value to avoid division by zero.
    Returns:
        torch.Tensor: The SI-SNR loss value.
    """
    # Ensure zero-mean signals
    preds_mean = torch.mean(preds, dim=-1, keepdim=True)
    targets_mean = torch.mean(targets, dim=-1, keepdim=True)
    preds = preds - preds_mean
    targets = targets - targets_mean

    # Compute the scaling factor
    scale = torch.sum(preds * targets, dim=-1, keepdim=True) / (torch.sum(targets ** 2, dim=-1, keepdim=True) + eps)
    s_target = scale * targets
    e_noise = preds - s_target

    # Compute SI-SNR
    si_snr = 10 * torch.log10((torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps))

    # Return the negative SI-SNR loss (to minimize)
    return -torch.mean(si_snr)

def normalize(mix, targets):
    mean = mix.mean()
    std = mix.std()
    if std != 0:
        mix = (mix - mean) / std
        targets = (targets - mean) / std
    return mix, targets


# Training function
def train_model(
    model, train_loader, val_loader,
    epochs=100, learning_rate=9.0e-05, output_dir="./model/",
    device="cpu", early_stopping=None, loss_type="si_snr"
):
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True)  # Use 'max' for SDR

    # Select loss function
    if loss_type == "si_snr":
        loss_fn = si_snr_loss
    elif loss_type == "masked":
        loss_fn = masked_loss
    else:
        raise ValueError("Invalid loss_type. Choose 'si_snr' or 'masked'.")

    for epoch in range(epochs):
        print("-" * 20)
        print(f"Epoch {epoch + 1}/{epochs}")

        # ---- Training Phase ----
        model.train()
        train_loss = 0.0
        for mix, targets in tqdm(train_loader):
            mix, targets = mix.to(device), targets.to(device)
            mix, targets = normalize(mix, targets)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(mix)

            # Compute loss
            loss = loss_fn(outputs, targets)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.8f}")

        # ---- Validation Phase ----
        model.eval()
        val_loss = 0.0
        total_sdr = 0.0
        num_batches = 0
        with torch.no_grad():
            for mix, targets in val_loader:
                mix, targets = mix.to(device), targets.to(device)

                # Forward pass
                outputs = model(mix)

                # Compute loss
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

                # SDR calculation
                targets_np = targets.cpu().numpy()
                outputs_np = outputs.cpu().numpy()
                batch_sdr = sdr(targets_np, outputs_np)
                total_sdr += np.mean(batch_sdr)
                num_batches += 1

        # Calculate average validation loss and SDR
        val_loss /= len(val_loader)
        avg_sdr = total_sdr / num_batches

        # Print epoch results
        print(f"Validation Loss: {val_loss:.8f}")
        print(f"Average SDR: {avg_sdr:.8f} dB")

        # Scheduler step 
        scheduler.step(val_loss)

        # Early stopping 
        if early_stopping:
            early_stopping(val_loss, model, output_dir, epoch)  # Minimize negative SDR
            if early_stopping.early_stop:
                break

    return model