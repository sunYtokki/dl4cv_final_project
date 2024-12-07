import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from utils import upload_model_to_gcs

# Helper class for early stopping logic on validation SDR
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = -np.inf  # Initialize to negative infinity for maximizing
        self.early_stop = False
        self.prompt = "Early stopping triggered"

    def __call__(self, val_metric, model, output_dir, epoch):
        # Save checkpoint for every epoch
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Check if validation SDR has improved
        if val_metric > self.best_metric + self.min_delta:
            self.best_metric = val_metric
            self.counter = 0  # Reset counter if improvement is seen
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Validation SDR improved to {val_metric:.8f} dB, model saved.")
        else:
            self.counter += 1
            print(f"No improvement in validation SDR. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(self.prompt)


# SDR Calculation Function (per sample)
def sdr(reference, estimate):
    """
    Compute SDR between reference and estimate signals.

    reference: numpy array of shape [batch_size, num_channels, signal_length]
    estimate: numpy array of same shape as reference

    Returns:
        sdr: numpy array of shape [batch_size], SDR per sample
    """
    delta = 1e-7  # To avoid numerical errors

    # Compute numerator and denominator
    num = np.sum(reference ** 2, axis=(1, 2)) + delta
    den = np.sum((reference - estimate) ** 2, axis=(1, 2)) + delta

    sdr = 10 * np.log10(num / den)
    return sdr  # shape [batch_size]

# Masked Loss Function


def masked_loss(y_, y, q=0.95):
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    loss = torch.mean(loss, dim=(-1, -2)) 
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()

def normalize(mix, targets):
    # Normalize mix and targets using mix statistics to preserve relative amplitudes
    mean = mix.mean()
    std = mix.std()
    if std != 0:
        mix = (mix - mean) / std
        targets = (targets - mean) / std
    return mix, targets

# Training function with progress bar logging
def train_model(
    model, train_loader, val_loader,
    epochs=100, learning_rate=9.0e-05, output_dir="./model/",
    device="cpu", early_stopping=None, loss_type="masked", normalize_data=True,
    gcs_bucket_name=None, gcs_model_dir=None, stem_names=None
):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    model.to(device)  # Move model to the specified device

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True)  # Set mode to 'max' for SDR

    # Select loss function
    if loss_type == "masked":
        loss_fn = masked_loss
    else:
        raise ValueError("Invalid loss_type. Choose 'masked'.")

    for epoch in range(epochs):
        print("-" * 20)
        print(f"Epoch {epoch + 1}/{epochs}")

        # ---- Training Phase ----
        model.train()
        train_loss = 0.0
        total_samples = 0

        # Initialize the progress bar
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (mix, targets) in enumerate(pbar):
            mix, targets = mix.to(device), targets.to(device)
            if normalize_data:
                mix, targets = normalize(mix, targets)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(mix)

            # Compute loss
            loss = loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss * mix.size(0)
            total_samples += mix.size(0)

            # Update the progress bar
            avg_loss = train_loss / total_samples
            pbar.set_postfix({'loss': 100 * batch_loss, 'avg_loss': 100 * avg_loss})

        train_loss /= total_samples
        print(f"Training Loss: {train_loss:.8f}")

        # ---- Validation Phase ----
        model.eval()
        val_loss = 0.0
        total_sdr_stems = None  # To accumulate SDRs per stem
        num_samples = 0
        with torch.no_grad():
            for mix, targets in val_loader:
                mix, targets = mix.to(device), targets.to(device)
                if normalize_data:
                    mix, targets = normalize(mix, targets)

                # Forward pass
                outputs = model(mix)

                # Compute loss
                loss = loss_fn(outputs, targets)

                val_loss += loss.item() * mix.size(0)

                # SDR calculation per stem
                targets_np = targets.cpu().numpy()  # shape: [batch_size, num_stems, num_channels, signal_length]
                outputs_np = outputs.cpu().numpy()

                batch_size = targets_np.shape[0]
                num_samples += batch_size

                num_stems = targets_np.shape[1]

                if total_sdr_stems is None:
                    total_sdr_stems = [0.0] * num_stems

                for i in range(num_stems):
                    target_stem = targets_np[:, i, :, :]  # shape: [batch_size, num_channels, signal_length]
                    output_stem = outputs_np[:, i, :, :]
                    batch_sdr = sdr(target_stem, output_stem)  # shape: [batch_size]
                    total_sdr_stems[i] += np.sum(batch_sdr)  # Sum over batch

        # Calculate average validation loss
        val_loss /= num_samples

        # Calculate average SDR per stem
        avg_sdr_stems = [total_sdr_stem / num_samples for total_sdr_stem in total_sdr_stems]

        # Calculate overall average SDR
        avg_sdr = np.mean(avg_sdr_stems)

        # Print epoch results
        print(f"Validation Loss: {val_loss:.8f}")
        for i, stem_sdr in enumerate(avg_sdr_stems):
            stem_name = stem_names[i] if stem_names else f"Stem {i}"
            print(f"Average SDR for {stem_name}: {stem_sdr:.8f} dB")
        print(f"Overall Average SDR: {avg_sdr:.8f} dB")

        # Scheduler step with SDR
        scheduler.step(avg_sdr)

        # Early stopping based on SDR
        if early_stopping:
            early_stopping(avg_sdr, model, output_dir, epoch)  # Maximize SDR
            if early_stopping.early_stop:
                break
        else:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

            if gcs_bucket_name and gcs_model_dir:
                upload_model_to_gcs(model, gcs_bucket_name, gcs_model_dir, epoch)

    return model