import json
import pandas as pd
import torch

from classes.CustomTokenizer import CustomTokenizer
from torch.utils.data import DataLoader
from classes.TranslationDataset import TranslationDataset

class DataLoaderProcessor:
    def __init__(self, file_path, tokenizer_path, max_len=32, batch_size=32,vocab_size=1500000,device="cpu"):
        self.file_path = file_path
        self.tokenizer,self.vocab_size = CustomTokenizer(vocab_size=vocab_size).load(tokenizer_path)
        self.max_len = max_len
        self.batch_size = batch_size
        self.df = self.load_data()
        self.dataset = self.create_dataset()
        self.dataloader = self.create_dataloader()
        self.device=device

    def load_data(self):
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))  # Преобразуем строку в словарь
        return pd.DataFrame(data)

    def create_dataset(self):
        src_texts = self.df['src'].tolist()
        tgt_texts = self.df['dst'].tolist()
        return TranslationDataset(src_texts, tgt_texts, self.tokenizer, self.max_len)

    def create_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
