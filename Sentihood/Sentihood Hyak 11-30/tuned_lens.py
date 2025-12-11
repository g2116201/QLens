import torch
import torch.nn as nn
from tqdm import tqdm, trange
import os
import matplotlib.pyplot as plt
import numpy as np

class TunedLens(nn.Module):
    # A bias-only tuned lens for intermediate logit extraction

    def __init__(self, d_model, last_norm, unembedding_layer, name):
        """
        Args:
        d_model: dimensionality of the latent space of the original model

        last_norm: The layer norm of the original model's last layer

        unembedding_layer: The original model's linear prediction layer that
                           maps between the latent space and the class/vocabulary
                           space
        """
        super(TunedLens, self).__init__()

        self.bias = nn.Parameter(torch.zeros(d_model)) # Creating an intial bias vector
        self.layer_norm = last_norm
        self.unembedding = unembedding_layer
        self.name = name # Either 'embedding' or 'attention; Used for save file naming

        # Freezing all parameters except those of the bias vector
        for param in self.layer_norm.parameters():
            param.requires_grad = False

        for param in self.unembedding.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Forward pass adds trainable bias before applying original model layer norm and prediction layers
        z = x + self.bias
        z = self.layer_norm(z)

        # Logits are derived from CLS token located at the start of each input sequence
        cls_token_state = z[:, 0, :]
        intermediate_logits = self.unembedding(cls_token_state)

        return intermediate_logits

    def train_model(self, train_dataloader, criterion, optimizer, num_epochs, device, lens_dir, scheduler):
        """
        Trains the TunedLens model.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            criterion (callable): The loss function to minimize.
            optimizer (Optimizer): The PyTorch optimizer.
            num_epochs (int): Number of training epochs.
            proj_dir (str): Directory for saving checkpoint and loss plot.
        """
        self.to(device)

        # Detecting per-batch scheduler to change learning rate appropriately later
        per_batch_scheduler = scheduler is not None and hasattr(scheduler, "step") and getattr(scheduler, "_step_count", None) is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

        train_losses = []

        print("Starting training loop...")

        for epoch in range(num_epochs):
            self.train() # Set model to training mode
            lens_epoch_loss = 0
            n_batches = 0

            # Training Loop
            for batch_number, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")): # Using tqdm for a progress bar

                # Zero the gradients
                optimizer.zero_grad()

                residual = batch['residual'].to(device)
                final_logits = batch['final_logits'].to(device)

                # Forward pass
                lens_logits = self(residual).squeeze(dim = 1)

                # Calculating the loss
                lens_loss = criterion(final_logits.float(), lens_logits).to(device)
                lens_epoch_loss += lens_loss.item()

                # Backward pass to compute gradients
                lens_loss.backward()

                # Updating model parameters
                optimizer.step()

                # if scheduler is per-batch (e.g., OneCycleLR) step every batch
                if scheduler is not None and per_batch_scheduler:
                    scheduler.step()

                n_batches += 1

            # step epoch-based schedulers here
            if scheduler is not None and not per_batch_scheduler:
                try:
                    scheduler.step()
                except Exception:
                    # Some schedulers require a metric; we do not call those automatically
                    pass

            # Obtaining learning rate
            if scheduler is not None and hasattr(scheduler, "get_last_lr"):
                current_lr = float(scheduler.get_last_lr()[0])
            else:
                current_lr = float(optimizer.param_groups[0]['lr'])

            # Calculate and record average epoch loss
            average_epoch_loss = lens_epoch_loss / max(1, n_batches)
            train_losses.append(average_epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_epoch_loss:.4f}, Learning Rate: {current_lr:.4f}")

        print("Training finished!")

        # Saving model checkpoint
        checkpoint = {
            'epoch': num_epochs,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_epoch_loss,
        }
        # Saving checkpoint to the specified directory
        torch.save(checkpoint, lens_dir + self.name + '_lens_checkpoint.pth')
        print(f"Checkpoint for Epoch {num_epochs} saved to {lens_dir}{self.name}_lens_checkpoint.pth")

        # Creating a graph of loss vs epochs and saving
        plt.figure()
        plt.plot(np.array(range(1, num_epochs + 1)), np.array(train_losses), "-o")
        plt.grid(True)
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch Number")

        # Saving plot to the specified directory
        plt.savefig(lens_dir + self.name[0].upper() + self.name[1:] + "_lens_training_loss.png")
        plt.show()

        return train_losses

    def evaluate_model(self, test_dataloader, criterion, device):
        """
        Evaluates the TunedLens model on a test set.

        Args:
            test_dataloader (DataLoader): DataLoader for test data.
            criterion (callable): The loss function (e.g., kl_loss).
            device (torch.device): The device to evaluate on ('cpu' or 'cuda').

        Returns:
            dict: A dictionary containing the average test loss and accuracy.
        """
        self.to(device)
        self.eval() # Set model to evaluation mode

        val_loss = 0
        matching = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc = self.name[0].upper() + self.name[1:] + " Lens Evaluation"):
                # Moving data to device and preprocessing
                residual = batch['residual'].to(device)
                final_logits = batch['final_logits'].to(device)

                # Forward pass
                lens_logits = self(residual).squeeze(dim = 1)

                # Calculating the loss
                val_loss += criterion(final_logits.float(), lens_logits).to(device).item() # Use .item()

                # Converting logits to predictions
                lens_predicted = (lens_logits.argmax(dim=-1)).float()
                model_predicted = (final_logits.argmax(dim=-1)).float()

                # Counting correct matches
                matching += (lens_predicted == model_predicted).sum().item()

                # Counting total samples
                total += lens_predicted.size(0)

        avg_val_loss = val_loss / len(test_dataloader)
        accuracy = matching / total

        print(f"\nTest Loss: {avg_val_loss:.4f}, Accuracy against base model: {accuracy:.4f}")
        print("Test complete!")

        return {
            "avg_val_loss": avg_val_loss,
            "accuracy": accuracy
            }