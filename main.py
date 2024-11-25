import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer

from classes.CustomTokenizer import CustomTokenizer
from classes.DataLoaderProcessor import DataLoaderProcessor
from classes.Trainer import Trainer
from classes.TranslationDataset import TranslationDataset, collate_fn
from classes.Translator import TransformerTranslator
import torch

def main():
    batch_size=32
    seq_len = 32
    vocab_size=1500000
    device = torch.device("cuda")
    data_processor = DataLoaderProcessor(file_path='alien_translation/data/train', tokenizer_path='alien_translation/custom_tokenizer.json', max_len=seq_len,
                                         batch_size=batch_size,vocab_size=vocab_size)
    vocab_size=data_processor.vocab_size
    model = TransformerTranslator(vocab_size=vocab_size, d_model=1024, num_heads=16, num_layers=32,device=device).to(
        data_processor.device)

    model.to(device)
    # Инициализация тренера
    trainer = Trainer(model=model, dataloader=data_processor.dataloader, vocab_size=vocab_size, num_epochs=10,
                      device=data_processor.device)

    # Запуск тренировки
    trainer.train()


if __name__ == "__main__":
    main()
