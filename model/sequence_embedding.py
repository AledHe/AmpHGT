"""
Simple BiLSTM to incorporate sequence information.
"""

import torch
import torch.nn as nn

def read_vocab(vocab_file: str) -> list:
    # append "<PAD>", "<UNK>", to the beginning of the vocab
    vocab = ["<PAD>", "<UNK>"]
    with open(vocab_file, 'r', encoding="utf-8") as f:
        for line in f:
            vocab.append(line.strip())

    token2id = { aa: idx for idx, aa in enumerate(vocab) }
    vocab_size = len(token2id)
    return vocab, token2id, vocab_size

VOCAB, TOKEN2ID, VOCAB_SIZE = read_vocab("model/vocab.txt")

def parse_sequence(seq: str):
    tokens = []
    i = 0
    while i < len(seq):
        if seq[i] == '<':
            j = i
            while j < len(seq) and seq[j] != '>':
                j += 1
            tokens.append(seq[i:j+1])  # '<D-Allo-ILE>'
            i = j + 1
        else:
            # for single character, directly append to tokens
            tokens.append(seq[i])
            i += 1
    
    tokens = [t.strip() for t in tokens if t.strip() != '']
    return tokens

def encode_sequence(tokens, token2id, max_len=100):
    """
    map tokens to ids and pad to max_len
    """
    unk_idx = token2id.get("<UNK>", 1)
    pad_idx = token2id.get("<PAD>", 0)

    ids = [token2id.get(t, unk_idx) for t in tokens]

    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids
    
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=150, hidden_dim=150, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.1)
    
    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len]
        return:
            outputs: [batch_size, seq_len, hidden_dim * 2]
            (h, c): LSTM's hidden state and cell state
        """
        # get embedding from vocab
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        outputs, (h, c) = self.lstm(x) 
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # h: [num_layers*2, batch_size, hidden_dim]
        # c: [num_layers*2, batch_size, hidden_dim]

        # split the hidden state and cell state
        # [num_layers, 2, batch_size, hidden_dim]
        h = h.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
        c = c.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)

        last_h_forward = h[-1, 0, :, :] # the last layer, forward direction, [batch_size, hidden_dim]
        last_h_backward = h[-1, 1, :, :] # the last layer, backward direction, [batch_size, hidden_dim]

        fused_h = torch.cat([last_h_forward, last_h_backward], dim=-1)  # [batch_size, hidden_dim*2]

        return fused_h