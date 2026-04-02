import torch
from torch.utils.data import Dataset
from src.preprocess import (clean_sequence, encode_sequence, encode_labels_q3, encode_labels_q8, pad_idx)

label_pad_idx = -100

class ProteinSSDataset(Dataset):
    def __init__(self, records, mode='train'):
        self.records = records
        self.mode = mode
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        r = self.records[idx]
        seq = clean_sequence(r["seq"])
        seq_ids = torch.tensor(encode_sequence(seq), dtype=torch.long)
        
        item = {"id": r["id"], "seq_ids": seq_ids}
        
        if self.mode != "test":
            item["sst3_ids"] = torch.tensor(encode_labels_q3(r["sst3"]), dtype=torch.long)
            item["sst8_ids"] = torch.tensor(encode_labels_q8(r["sst8"]), dtype=torch.long)
            
        return item
    
def collate_fn(batch):
    max_len = max(len(x["seq_ids"]) for x in batch)
    batch_seq_ids, batch_mask = [], []
    batch_sst3, batch_sst8 = [], []
    has_labels = "sst3_ids" in batch[0]
        
    for x in batch:
        seq = x['seq_ids']
        pad_len = max_len - len(seq)
            
        padded_seq = torch.cat([seq, torch.full((pad_len,), pad_idx)])
        mask = torch.cat([torch.ones(len(seq)), torch.zeros(pad_len)])
            
        batch_seq_ids.append(padded_seq)
        batch_mask.append(mask)
            
        if has_labels:
            batch_sst3.append(torch.cat([x['sst3_ids'], torch.full((pad_len,), label_pad_idx)]))
            batch_sst8.append(torch.cat([x['sst8_ids'], torch.full((pad_len,), label_pad_idx)]))
                
    output = {"seq_ids": torch.stack(batch_seq_ids), "mask": torch.stack(batch_mask).bool()}
        
    if has_labels:
        output['sst3_ids'] = torch.stack(batch_sst3)
        output['sst8_ids'] = torch.stack(batch_sst8)
            
    return output