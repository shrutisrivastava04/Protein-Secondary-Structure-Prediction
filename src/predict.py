import torch
from src.model import SSModel, SequenceEncoder
from src.preprocess import (clean_sequence, encode_sequence, decode_q3, decode_q8, aa_vocab)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

encoder = SequenceEncoder(vocab_size=len(aa_vocab))
model = SSModel.load_from_checkpoint("models/best-model.ckpt", encoder=encoder)

model.to(device)
model.eval()

def predict_structure(sequence: str):
    sequence = clean_sequence(sequence)
    seq_ids = torch.tensor(encode_sequence(sequence)).unsqueeze(0).to(device)
    mask = torch.ones_like(seq_ids).bool()
    
    with torch.no_grad():
        q3_logits, q8_logits = model(seq_ids, mask)
        
    q3_preds = q3_logits.argmax(dim=-1)[0]
    q8_preds = q8_logits.argmax(dim=-1)[0]
    
    return {"q3": decode_q3(q3_preds), "q8": decode_q8(q8_preds)}
