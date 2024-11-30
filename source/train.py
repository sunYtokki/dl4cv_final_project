import os
import numpy as np

import torch.nn as nn
import torch
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
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Check if validation loss has improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model:
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Validation loss improved to {val_loss:.8f}, model saved.")
        else:
            self.counter += 1
            print(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(self.prompt)

# SDR Calculation Function
def compute_sdr(references, estimates):
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

# Training Function
def train_model(
    model, train_loader, val_loader,
    epochs=100, learning_rate=9.0e-05, output_dir="./model/",
    device="cpu", early_stopping=None
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

    for epoch in range(epochs):
        print("-" * 20)
        print(f"Epoch {epoch + 1}/{epochs}")

        # ---- Training Phase ----
        model.train()
        train_loss = 0.0
        for mix, targets in tqdm(train_loader):
            mix, targets = mix.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(mix)
            loss = masked_loss(outputs, targets)
            train_loss += loss.item()

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
                loss = masked_loss(outputs, targets)
                val_loss += loss.item()

                # Compute SDR
                targets_np = targets.cpu().numpy()  # Shape: [batch_size, num_stems, chunk_size]
                outputs_np = outputs.cpu().numpy()  # Shape: [batch_size, num_stems, chunk_size]

                # Compute SDR for each batch
                for b in range(targets_np.shape[0]):  # Iterate over batch
                    batch_sdr = compute_sdr(targets_np[b], outputs_np[b])
                    total_sdr += batch_sdr

                num_batches += targets_np.shape[0]

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
            early_stopping(val_loss, model, output_dir, epoch)
            if early_stopping.early_stop:
                break

    return model