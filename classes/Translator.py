import math
import random


import torch
import torch.nn as nn

class TransformerTranslator(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, device='cpu'):
        super(TransformerTranslator, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.get_positional_encoding(d_model, 1000).to(device)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1,device=self.device),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1,device=self.device),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"


    def get_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)



    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src).transpose(0, 1)
        if src_mask is None:
            src_mask = (src == 0)
        enc_output = self.encoder(src_emb, src_key_padding_mask=src_mask)

        if tgt is not None:
            tgt_emb = self.embedding(tgt).transpose(0, 1)

            if tgt_mask is None:
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(self.device)
            dec_output = self.decoder(tgt_emb, enc_output, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
            output = self.fc(dec_output.transpose(0, 1))
            return output
        else:
            return enc_output

    def training_step(self, src, tgt, teacher_forcing_ratio):
        
        src = src.to(self.device) 
        tgt = tgt.to(self.device)
        batch_size, seq_len = src.size()
        outputs = torch.zeros(seq_len, batch_size, self.fc.out_features).to(self.device)
        src_emb = self.embedding(src)
        src_mask = (src == 0).to(self.device)

        memory = self.encoder(src_emb, src_key_padding_mask=src_mask).to(self.device)

        input_t = tgt[:, 0].unsqueeze(0).to(self.device)

        for t in range(1, seq_len):
            tgt_emb = self.embedding(input_t).to(self.device)
            tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(self.device)
            dec_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask).to(self.device)
            logits = self.fc(dec_output).to(self.device)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            outputs[t] = logits[-1, :, :]

            if use_teacher_forcing:
                input_t = tgt[:, t].unsqueeze(0)
            else:
                pred_token = logits[-1, :, :].argmax(dim=-1)
                input_t = pred_token.unsqueeze(0)
        return outputs

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def greedy_decode(self, src, src_mask, max_len, start_symbol, eos_symbol=3):
        src = src.transpose(0, 1)
        memory = self.encoder(src, src_key_padding_mask=src_mask)
        memory = memory.transpose(0, 1)
        batch_size = src.size(1)
        ys = torch.full((1, batch_size), start_symbol, dtype=torch.long).to(self.device)

        for _ in range(max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(0)).to(self.device)
            out = self.decoder(
                ys.transpose(0, 1),
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_mask
            )
            out = self.fc(out.transpose(0, 1))
            prob = out[-1, :, :].softmax(dim=-1)  # Только последний токен
            next_word = torch.argmax(prob, dim=-1).unsqueeze(0)
            ys = torch.cat([ys, next_word], dim=0)
            if (next_word == eos_symbol).all():
                break

        return ys.transpose(0, 1)  # Вернем батч в первой размерности

