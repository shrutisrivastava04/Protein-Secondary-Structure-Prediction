import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.dataset import ProteinSSDataset, collate_fn
from src.model import SequenceEncoder, SSModel
from src.preprocess import aa_vocab
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def train():
    df = pd.read_csv("data/raw/train_pp.csv")
    train_records, val_records = train_test_split(df.to_dict("records"), test_size=0.1, random_state=42)
    train_ds = ProteinSSDataset(train_records, mode='train')
    val_ds = ProteinSSDataset(val_records, mode='train')
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    encoder=SequenceEncoder(vocab_size=len(aa_vocab))
    model=SSModel(encoder)
    checkpoint_cb = ModelCheckpoint(dirpath="models/", filename="best-model", monitor="train/loss", 
                                    mode='min', save_top_k=1)
    trainer=Trainer(max_epochs=10, callbacks=[checkpoint_cb])
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()