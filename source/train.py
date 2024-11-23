import os
import torch.nn as nn
import torch

# Define the training function
def train_model(
    model, train_loader, val_loader, test_loader,
    epochs=100, learning_rate=1e-4, output_dir="./model/",
    device="cpu"
):
    # Loss and optimizer
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # For tracking the best validation loss
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 20)

        # ---- Training ----
        model.train()
        train_loss = 0.0
        for mix, targets in train_loader:
            mix, targets = mix.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # print(f"mix shape: {mix.shape}")
            # print(f"targets shape: {targets.shape}")
            outputs = model(mix)
            # print(f"output shape: {outputs.shape}")

            # Compute loss
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.4f}")

        # ---- Validation ----
        # TODO: implement early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mix, targets in val_loader:
                mix, targets = mix.to(device), targets.to(device)

                # Forward pass
                outputs = model(mix)

                # Compute loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print("Best model saved!")

    # ---- Testing ----
    # TODO: move testing to separate file and update to SDR evaluation
    print("Testing the model...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for mix, targets in test_loader:
            mix, targets = mix.to(device), targets.to(device)

            # Forward pass
            outputs = model(mix)

            # Compute loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    print("Final model saved!")

    return model