# Vocabulary & Special Tokens
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
pad_token = "<PAD>"
unk_token = "<UNK>"
mask_token = "*"

aa_vocab = [pad_token, unk_token, mask_token] + amino_acids

aa_to_idx = {a: i for i, a in enumerate(aa_vocab)}
idx_to_aa = {i: a for a, i in aa_to_idx.items()}

pad_idx = aa_to_idx[pad_token]
unk_idx = aa_to_idx[unk_token]

# Label Mapping (Q3 & Q8)

q3_labels = ['H', 'E', 'C']
q3_to_idx = {l: i for i, l in enumerate(q3_labels)}
idx_to_q3 = {i: l for l, i in q3_to_idx.items()}

q8_labels = ['H', 'G', 'I', 'E', 'B', 'T', 'S', 'C']
q8_to_idx = {l: i for i, l in enumerate(q8_labels)}
idx_to_q8 = {i: l for l, i in q8_to_idx.items()}
# Sequence Cleaning

def clean_sequence(seq: str) -> str:
    """ 
    Replacing invalid amino acids with mask_token (*)
    """
    return "".join([c if c in amino_acids else mask_token for c in seq])

# Encoding Functions
def encode_sequence(seq: str):
    """ 
    Converting sequence string to list of indices
    """
    return [aa_to_idx.get(c, unk_idx) for c in seq]

def encode_labels_q3(labels: str):
    return [q3_to_idx[c] for c in labels]

def encode_labels_q8(labels:str):
    return [q8_to_idx[c] for c in labels]

# Decoding Functions
def decode_q3(indices):
    return "".join([idx_to_q3[int(i)] for i in indices])

def decode_q8(indices):
    return "".join([idx_to_q8[int(i)] for i in indices])