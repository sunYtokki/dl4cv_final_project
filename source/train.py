import os
import numpy as np

import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm.auto import tqdm

# Helper class for early stopping logic on validation loss
# It also saves the best model path
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        # Initialize EarlyStopping with patience (epochs to wait for improvement) and min_delta (minimum change to qualify as improvement)
        self.patience = patience  # Number of epochs to wait after last improvement before stopping
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.prompt = "Early stopping triggered"

    def __call__(self, val_loss, model, output_dir):
        # Check if validation loss has improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement is seen
            if model:
                # Save the best model's weights
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Validation loss improved to {val_loss:.8f}, model saved.")
        else:
            self.counter += 1
            print(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(self.prompt)

def masked_loss(y_, y, q=0.95, coarse=True):
    # shape = [num_sources, batch_size, num_channels, chunk_size]
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()

# Define the training function
def train_model(
    model, train_loader, val_loader,
    epochs=100, learning_rate=9.0e-05, output_dir="./model/",
    device="cpu", early_stopping=None 
):
    # Loss and optimizer
    # criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True)

    for epoch in range(epochs):
        print("-" * 20)
        print(f"Epoch {epoch + 1}/{epochs}")

        # ---- Training ----
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader)
        for i, (mix, targets) in enumerate(pbar):
            mix, targets = mix.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # print(f"mix shape: {mix.shape}")
            # print(f"targets shape: {targets.shape}")
            outputs = model(mix)
            # print(f"output shape: {outputs.shape}")

            # Compute loss
            # loss = criterion(outputs, targets)

            loss = masked_loss(
                outputs,
                targets
            )

            train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.8f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mix, targets in val_loader:
                mix, targets = mix.to(device), targets.to(device)

                # Forward pass
                outputs = model(mix)

                # Compute loss
                # loss = criterion(outputs, targets)
                loss = masked_loss(
                    outputs,
                    targets
                )
                val_loss += loss.item()

        # Calculate average validation loss and accuracy for the epoch
        val_loss /= len(val_loader)

        # Print epoch results
        print(f"Val Loss: {val_loss:.8f}")

        # Update learning rate
        scheduler.step(val_loss)
    
        # Early stopping
        if early_stopping:
            early_stopping(val_loss, model, output_dir)
            if early_stopping.early_stop:
                break


    # ---- Testing ----
    # TODO: move testing to separate file and update to SDR evaluation
    # print("Testing the model...")
    # model.eval()
    # test_loss = 0.0
    # with torch.no_grad():
    #     for mix, targets in test_loader:
    #         mix, targets = mix.to(device), targets.to(device)

    #         # Forward pass
    #         outputs = model(mix)

    #         # Compute loss
    #         loss = criterion(outputs, targets)
    #         test_loss += loss.item()

    # test_loss /= len(test_loader)
    # print(f"Test Loss: {test_loss:.4f}")
    # torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    # print("Final model saved!")

    return model