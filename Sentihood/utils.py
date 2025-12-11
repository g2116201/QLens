import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
kl_loss() computes the Kullback-Leibler (KL) divergence loss between
two sets of logits in accordance with the original Tuned Lens paper.

Args:
    base_logits (torch.Tensor): The logits from the base Transformer.
    lens_logits (torch.Tensor): The logits from the Tuned Lens model.

Returns:
    torch.Tensor: The mean KL divergence loss over the batch.
"""
def kl_loss(base_logits, lens_logits):
    # Getting the log-softmax of the teacher's logits
    log_p = nn.functional.log_softmax(base_logits, dim=-1)

    # Getting log-softmax of the student's logits
    log_q = nn.functional.log_softmax(lens_logits, dim=-1)

    # Exponentiating log_p to get the probability distribution P
    p = torch.exp(log_p)

    # Computing the KL divergence for each sample in the batch
    kl = torch.sum(p * (log_p - log_q), dim=-1)

    # Return the mean of the KL divergences over the batch
    return torch.mean(kl)

class GPT2EmbeddingUtility:
    '''Used to extract initial GPT-2 embeddings to be passed to submodels''' 
    def __init__(self, tokenizer):
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        gpt2_model.resize_token_embeddings(len(tokenizer))
        
        self.word_embeddings_layer = gpt2_model.transformer.wte
        self.positional_embeddings_layer = gpt2_model.transformer.wpe
        
    
    def get_embeddings(self, input_ids):

        token_embeddings = self.word_embeddings_layer(input_ids)
        positional_embeddings = self.positional_embeddings_layer(torch.arange(input_ids.shape[1], device=input_ids.device))
            
        combined_embeddings = token_embeddings + positional_embeddings
            
        return combined_embeddings