import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Dataset class for training base Transformer model on the Sentihood Dataset
class SentihoodDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenizing the text without padding
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Appending CLS token to the beginning of each sequence
        input_ids = encoding['input_ids']
        
        cls_token_id = self.tokenizer.cls_token_id
        cls_token_tensor = torch.tensor([[cls_token_id]], dtype = torch.long)
        input_ids = torch.cat([cls_token_tensor, input_ids], dim = 1)

        # Gernating an appropriate key padding mask
        pad_mask = torch.zeros((1, encoding['attention_mask'].shape[1] + 1), dtype = torch.bool)

        return {
            'input_ids': input_ids,
            'labels': label.squeeze(),
            'pad_mask': pad_mask
        }

def sentihood_collate_fn(batch, pad_id):
    # Separating the batch into components
    input_ids = [item['input_ids'].squeeze() for item in batch]
    labels = [item['labels'] for item in batch]
    pad_masks = [item['pad_mask'].squeeze() for item in batch]

    # Pading the input_ids
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value = pad_id)

    # Adjusting the padding masks
    S = padded_input_ids.shape[1]
    for i in range(len(pad_masks)):
        if pad_masks[i].shape[0] < S:
            mask_extension = torch.ones(S - pad_masks[i].shape[0], dtype = torch.bool)
            pad_masks[i] = torch.cat([pad_masks[i], mask_extension], dim = 0)

    # Stacking pad_masks
    pad_masks = torch.stack(pad_masks)
    
    # Stacking labels
    labels = torch.stack(labels)

    return {
        'input_ids': padded_input_ids,
        'labels': labels,
        'pad_mask': pad_masks
    }