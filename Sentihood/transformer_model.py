import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from tqdm import tqdm, trange
import os
import matplotlib.pyplot as plt
import numpy as np

class TransformerModel(nn.Module):
    """
    A single-layer Transformer model using PyTorch's nn.TransformerEncoderLayer.
    """
    def __init__(self, d_embedding, d_model, dim_feedforward, num_heads, dropout_rate, tokenizer_len):
        """
        Args:
            d_embedding (int): The dimension of the model's input embeddings
            d_model (int): The dimension of the model's feature space.
            num_heads (int): The number of attention heads.
            dim_feedforward (int): The dimension of the inner MLP layer.
            dropout_rate (float): The dropout rate.
        """
        super(TransformerModel, self).__init__()

        # Loading a GPT-2 Model to extract its embedding matricies
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_model.resize_token_embeddings(tokenizer_len)

        self.word_embeddings_layer = gpt2_model.transformer.wte
        self.positional_embeddings_layer = gpt2_model.transformer.wpe

        self.compression = nn.Linear(d_embedding, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout_rate, batch_first = True)
        self.w1 = nn.Linear(d_model, dim_feedforward)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.w2 = nn.Linear(dim_feedforward, d_model)
        self.attention_norm = nn.LayerNorm(d_model)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.w1_dropout = nn.Dropout(dropout_rate)
        self.w2_dropout = nn.Dropout(dropout_rate)

        # A final linear layer to project the output to a single value
        self.classification_layer = nn.Linear(d_model, 2)

    def forward(self, x, pad_mask, return_probs = True):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
                              Contains sequence embeddings derived from GPT-2
            pad_mask (torch.Tensor): Input tensor of shape (batch_size, seq_len).
                                     Specifies which tokens to ignore during attention.
            return_probs (boolean): Defaults to True, otherwise logits are returned

        Returns:
            torch.Tensor: The final output of the model, shape (batch_size, 1).
        """

        # Compression layer reduces dimensionality of embeddings to reduce model size
        compressed_embeddings = self.compression(x)

        # Attention layer
        attention_output = self.attention(compressed_embeddings, compressed_embeddings, compressed_embeddings, key_padding_mask = pad_mask)[0]
        y = compressed_embeddings + self.attention_dropout(attention_output)
        y = self.attention_norm(y)

        # MLP layer
        mlp_output = self.w1_dropout(nn.functional.gelu(self.w1(y)))
        mlp_output = self.w2_dropout(self.w2(mlp_output))
        y = y + mlp_output
        y = self.mlp_norm(y)

        # Get classification logits through the final layer
        final_logits = self.classification_layer(y[:, 0, :])

        # Returning either the raw logits or softmaxed probabilities
        if not return_probs:
          return final_logits

        else:
          # Pass the result through the softmax
          output_probs = nn.functional.softmax(final_logits, dim = 1)

          return output_probs
    
    """
    get_embeddings() returns GPT-2 embeddings for an input token sequence
    
    Args:
        input_ids (torch.Tensor): token ids of input sentence
    
    Returns:
        combined_embeddings (torch.Tensor): GPT-2 embeddings for input token IDs
    """
    def get_embeddings(self, input_ids):
        self.word_embeddings_layer.to(input_ids.device)
        self.positional_embeddings_layer.to(input_ids.device)
        
        token_embeddings = self.word_embeddings_layer(input_ids)
        positional_embeddings = self.positional_embeddings_layer(torch.arange(input_ids.shape[1], device=input_ids.device))
        
        combined_embeddings = token_embeddings + positional_embeddings
        
        return combined_embeddings

    def train_model(self, train_dataloader, criterion, optimizer, num_epochs, device, model_dir, lr_scheduler):
    
        train_losses = []
    
        print("Starting training loop...")
        
        # Beginning training loop
        for epoch in range(num_epochs):
            self.train() # Set model to training mode
        
            epoch_loss = 0
        
            for batch_number, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                # Zero the gradients
                optimizer.zero_grad()
            
                # Extracting input_ids and class labels from batch
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                pad_mask = batch['pad_mask'].to(device)
            
                # Getting GPT-2 embeddings for input sentences
                embeddings = self.get_embeddings(input_ids).squeeze(1).to(device)
            
                # Passing GPT-2 embeddings through model to extract probabilities
                probs = self(embeddings, pad_mask)
            
                # Calculate the loss
                loss = criterion(probs, labels.float()).to(device)
                epoch_loss += loss
            
                # Backward pass: compute gradients
                loss.backward()
            
                # Update model parameters
                optimizer.step()

            # Updating learning rate per epoch
            lr_scheduler.step()

            # Obtaining current LR to print
            if hasattr(lr_scheduler, "get_last_lr"):
                current_lr = float(lr_scheduler.get_last_lr()[0])
            else:
                current_lr = float(optimizer.param_groups[0]['lr'])
            
            average_epoch_loss = epoch_loss / len(train_dataloader)
            train_losses.append(average_epoch_loss.cpu().detach().item())
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_epoch_loss:.4f}, Learning Rage: {current_lr:.4f}")
        
        print("Training finished!")
        
        # Saving model checkpoint
        checkpoint = {
                    'epoch': num_epochs,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,  # Assuming 'loss' is the final loss of the epoch
                }
        torch.save(checkpoint, model_dir + 'model_checkpoint.pth')
        print(f"Checkpoint for Epoch {num_epochs} saved")
        
        # Creating a graph of loss vs epochs and saving
        plt.plot(np.array(range(1, num_epochs + 1)), np.array(train_losses), "-o")
        plt.grid(True)
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch Number")
        
        plt.savefig(model_dir + "model_training_loss.png")
        plt.show()

    def eval_model(self, test_dataloader, criterion, device):
        # Conduncting evalutation on the test set
        self.eval()
        
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc = "Model Evaluation"):
                # Extracting input_ids and class labels from batch
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                pad_mask = batch['pad_mask'].to(device)
        
                # Getting GPT-2 embeddings for input sentences
                embeddings = self.get_embeddings(input_ids).squeeze(1).to(device)
        
                # Passing GPT-2 embeddings through model to extract probabilities
                probs = self(embeddings, pad_mask)
        
                # Computing validation loss
                val_loss += criterion(probs, labels.float()).item()
        
                total += labels.size(0)
                predicted = (probs > 0.5).float() # Convert logits to predictions
                correct += (predicted == labels).all(dim=1).sum().item()
        
        avg_val_loss = val_loss / len(test_dataloader)
        accuracy = correct / total
        print(f"\nTest Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        print("Test complete!")