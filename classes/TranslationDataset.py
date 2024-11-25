import math
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    src_tokens = [item['input_ids'] for item in batch]
    tgt_tokens = [item['labels'] for item in batch]
    src_padded = pad_sequence(src_tokens, batch_first=True, padding_value=3)
    tgt_padded = pad_sequence(tgt_tokens, batch_first=True, padding_value=3)
    return {'input_ids': src_padded, 'labels': tgt_padded}
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = 3   # Default to 3 if pad token is not set

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx] if self.tgt_texts else ""

        src_encoding = self.tokenizer.encode(src_text)
        tgt_encoding = self.tokenizer.encode(tgt_text)
        src_tokens = torch.tensor(src_encoding.ids[:self.max_length] + [3] * (self.max_length - len(src_encoding.ids)))
        tgt_tokens = torch.tensor(tgt_encoding.ids[:self.max_length] + [3] * (self.max_length - len(tgt_encoding.ids)))

        return {
            'input_ids': src_tokens,
            'labels': tgt_tokens
        }