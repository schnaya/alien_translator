import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from classes.Translator import TransformerTranslator


class Trainer:
    def __init__(self, model, dataloader, vocab_size, learning_rate=1e-4, num_epochs=10, device=None):
        self.model = model
        self.dataloader = dataloader
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=3)
        self.best_loss = float("inf")
        self.no_improvement_epochs = 0

    def train(self):
        teacher_forcing_start_ratio = 1.0
        teacher_forcing_end_ratio = 0.5

        for epoch in range(self.num_epochs):
            teacher_forcing_ratio = teacher_forcing_start_ratio - (
                    teacher_forcing_start_ratio - teacher_forcing_end_ratio) * (epoch / self.num_epochs)

            print(f"Epoch {epoch + 1}/{self.num_epochs}")

            epoch_loss = 0.0  # Сумма ошибок за эпоху
            num_batches = len(self.dataloader)
            with tqdm(self.dataloader, desc=f"Training Epoch {epoch + 1}") as pbar:
                for batch in pbar:
                    src = batch['input_ids'].to(self.device)
                    tgt = batch['labels'].to(self.device)
                    outputs = self.model.training_step(src, tgt, teacher_forcing_ratio).to(self.device)
                    loss = self.criterion(outputs[1:].reshape(-1, self.vocab_size), tgt[:, 1:].reshape(-1))
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Суммируем ошибку для подсчета среднего по эпохе
                    epoch_loss += loss.item()

                    # Обновление строки прогресс-бара
                    pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

            # Средняя ошибка за эпоху
            epoch_loss /= num_batches
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.no_improvement_epochs = 0
                torch.save(self.model.state_dict(), f"best_model_epoch_{epoch + 1}.pth")
                print(f"Model saved at epoch {epoch + 1} with loss {self.best_loss:.4f}")
            else:
                self.no_improvement_epochs += 1

            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {epoch_loss:.4f}, Best Loss: {self.best_loss:.4f}")

            # Ранняя остановка
            if self.no_improvement_epochs >= 1:  # patience
                print(f"Early stopping at epoch {epoch + 1}")
                break
        torch.save(self.model.state_dict(), "final_model.pth")
        print("Model saved after completing all epochs")

