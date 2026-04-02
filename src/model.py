import torch
from torch import nn
import pytorch_lightning as pl

label_pad_idx = -100

class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size = emb_dim, hidden_size = hidden_dim,
                            num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.3)
        self.output_dim = hidden_dim * 2
        
    def forward(self, seq_ids, mask):
        emb = self.embedding(seq_ids)
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, 
                                                   batch_first=True, 
                                                   enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out
    
class SSModel(pl.LightningModule):
    def __init__(self, encoder, lr=5e-4):
        super().__init__()
        self.encoder = encoder
        self.q3_head = nn.Linear(encoder.output_dim, 3)
        self.q8_head = nn.Linear(encoder.output_dim, 8)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=label_pad_idx)
        self.lr = lr
    
    def forward(self, seq_ids, mask):
        features = self.encoder(seq_ids, mask)
        q3_logits = self.q3_head(features)
        q8_logits = self.q8_head(features)
        
        return q3_logits, q8_logits
    
    def training_step(self, batch, batch_idx):
        q3_logits, q8_logits = self(batch['seq_ids'], batch['mask'])
        loss_q3 = self.loss_fn(q3_logits.view(-1,3), batch['sst3_ids'].view(-1))
        loss_q8 = self.loss_fn(q8_logits.view(-1, 8), batch['sst8_ids'].view(-1))
        loss = (loss_q3 + loss_q8) / 2
        
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)