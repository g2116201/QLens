import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel
from sub_models import AttentionModel
from utils import GPT2EmbeddingUtility

class TunedLensDataset(Dataset):
    # Customed Dataset class for training tuned lenses
    def __init__(self, texts, tokenizer, sub_model, final_model, device):
        # Sentihood instances to classify
        self.texts = texts

        # Tokenizer and Embedding to prepare inputs for models
        self.tokenizer = tokenizer
        self.embedding = GPT2EmbeddingUtility(tokenizer)

        # Submodel and final model for intermediate residual and final output generations
        self.sub_model = sub_model
        self.final_model = final_model

        # Device for computations
        self.device = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenizing the text
        encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors='pt'
                )

        # Prepending CLS token
        input_ids = encoding['input_ids']
        cls_token_id = self.tokenizer.cls_token_id
        cls_token_tensor = torch.tensor([[cls_token_id]], dtype = torch.long)
        input_ids = torch.cat([cls_token_tensor, input_ids], dim = 1)

        # Preparing pad mask
        pad_mask = torch.zeros(1, 1 + encoding['attention_mask'].shape[1], dtype = torch.bool).to(self.device)

        # Obtaining GPT-2 embeddings via get_embeddings function
        embeddings = self.embedding.get_embeddings(input_ids).to(self.device)

        # Using sub model to obtain intermediate resitual
        if isinstance(self.sub_model, AttentionModel):
            residual = self.sub_model(embeddings, pad_mask)
        else:
            residual = self.sub_model(embeddings)

        # Using original model to obtain final_logits to train lens against
        final_logits = self.final_model(embeddings, pad_mask, return_probs = False)

        return {
                'residual': residual,
                'final_logits': final_logits,
            }

def lens_collate_fn(batch):
    residuals = [item['residual'].squeeze() for item in batch]
    logits = [item['final_logits'].squeeze() for item in batch]

    # Padding residuals with zero vectors to allow for batching
    max_seq_len = max([item.shape[0] for item in residuals])
    for i in range(len(residuals)):
        S = residuals[i].shape[0]
        if S < max_seq_len:
            extension = torch.zeros(max_seq_len - S , residuals[i].shape[1]).to(residuals[i].device)
            residuals[i] = torch.cat([residuals[i], extension])

    batch_residuals = torch.stack(residuals, dim = 0)
    batch_logits = torch.stack(logits, dim = 0)

    return {
            'residual': batch_residuals,
            'final_logits': batch_logits
        }