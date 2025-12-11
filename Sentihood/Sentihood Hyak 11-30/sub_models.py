import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    # A component of the original Transformer consisting of only its compression layer
    def __init__(self, compression, device):
        super(EmbeddingModel, self).__init__()

        self.compression = compression
        self.device = device

        # Moving EmbeddingModel to inputted device
        self.to(self.device)

        # Freezing embedding_model parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through embedding model

        Args:
            x: torch.Tensor of input GPT-2 emeddings

        Returns:
            torch.Tensor: compressed embeddigs
        """
        x = x.to(self.device)
        return self.compression(x)

class AttentionModel(nn.Module):
    # A component of the original Transformer consisting of only its compression and attention layers
    def __init__(self, compression, attention, device):
        super(AttentionModel, self).__init__()

        self.compression = compression
        self.attention = attention
        self.device = device

        # Moving EmbeddingModel to inputted device
        self.to(self.device)

        # Freezing model parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, pad_mask):
        """
        Forward pass through attention model

        Args:
        x: torch.Tensor of input GPT-2 emeddings

        Returns residual stream state after the attention layer
        """
        x = x.to(self.device)
        compressed_embeddings = self.compression(x)
        attention_output = self.attention(compressed_embeddings, compressed_embeddings, compressed_embeddings, key_padding_mask = pad_mask)[0]
        res_stream = compressed_embeddings + attention_output

        return res_stream