"""
Simple Cov+Transformer to incorporate sequence information.
"""

import dgl
import torch
import torch.nn as nn

def read_vocab(vocab_file: str) -> list:
    # prepend "<PAD>", "<UNK>", "<CLS>" and "<MASK>" to the vocab
    vocab = ["<PAD>", "<UNK>", "<CLS>", "<MASK>"]
    with open(vocab_file, 'r', encoding="utf-8") as f:
        for line in f:
            vocab.append(line.strip())

    token2id = { aa: idx for idx, aa in enumerate(vocab) }
    vocab_size = len(token2id)
    return vocab, token2id, vocab_size

VOCAB, TOKEN2ID, VOCAB_SIZE = read_vocab("model/vocab.txt")

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x, seq_len):
        """
        x: [batch_size, seq_len, embed_dim]
        seq_len: integer
        """
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0) # [1, seq_len]
        pos_emb = self.pos_embedding(positions)  # [1, seq_len, embed_dim]
        return x + pos_emb # [batch_size, seq_len, embed_dim]
    
class MultiScaleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
                self.activation,
                self.dropout
            )
            self.branches.append(branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x.transpose(1, 2)).transpose(1, 2)  # [B, L, C]

        # multi cov
        branch_outputs = []
        for branch in self.branches:
            # x: [B, L, C_in]
            x_conv = x.transpose(1, 2)  # [B, C_in, L]
            out = branch(x_conv)        # [B, C_out, L]
            out = out.transpose(1, 2)   # [B, L, C_out]
            branch_outputs.append(out)

        # stack the outputs
        merged = torch.stack(branch_outputs, dim=0).sum(dim=0)  # [B, L, C_out]

        output = self.activation(merged + residual)
        return output
    
class ConvTransformerEncoder(nn.Module):
    """
    1D Conv + Transformer Encoder adapted for graph node features
    """
    def __init__(
        self,
        embed_dim: int = 256,
        conv_channels: int = 256,
        kernel_size: int = 5,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        ):
        """
        args:
        - embed_dim: word vector dimension
        - conv_channels: number of convolution output channels
        - kernel_size: convolution kernel size
        - num_heads: number of Transformer multi-head attention
        - num_layers: number of TransformerEncoderLayer layers
        - ff_dim: hidden layer dimension of FeedForward network in Transformer
        - dropout: Dropout probability
        """
        super().__init__()
        
        # 1D Convolution
        self.conv1d = MultiScaleConv(
            in_channels=embed_dim,
            out_channels=conv_channels,
            kernel_sizes=kernel_size
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, key_padding_mask=None):
        """
        x: [batch_size, seq_len, embed_dim]
        mask: [batch_size, seq_len, seq_len], 1 -> include, 0 -> exclude
        """
        residual = x

        # Convolution over embedding dimension
        x = self.conv1d(x)     # [batch_size, conv_channels, seq_len]

        x = x + residual

        out = self.transformer_encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        return out
    
class ResidueEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        max_len=201,
        token2id=None,
        conv_channels=256,
        kernel_size=[1, 3, 5, 7],
        num_heads=4,
        num_layers=2,
        ff_dim=1024,
        dropout=0.4
        ):
        super().__init__()
        if token2id:
            self.token2id = token2id
        else:
            self.token2id = TOKEN2ID
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0  # or token2id["<PAD>"]
        )
        self.pos_emb = PositionEmbedding(max_len, embed_dim)
        self.encoder = ConvTransformerEncoder(
            embed_dim=embed_dim,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout
        )
        self.init_wb()

    def init_wb(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Conv1d):
                # 卷积层使用 He 初始化 + ReLU修正
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Transformer 中的线性层使用 Xavier 初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _prepare_residue_batch(self, bg):
        # unbatch
        graphs = dgl.unbatch(bg)
        batch_size = len(graphs)

        # gather all subgraphs
        all_sorted_labels = []
        all_inverse_indices = []
        max_len = 0

        for g in graphs:
            if g.num_nodes('rsd') == 0:
                # store empty placeholders
                all_sorted_labels.append(torch.tensor([], device=g.device, dtype=torch.long)) # [0]
                all_inverse_indices.append(torch.tensor([], device=g.device, dtype=torch.long)) # [0]
                continue

            # sort by 'pos' and insert cls.
            sorted_pos, sorted_idx = torch.sort(g.nodes['rsd'].data['pos'])
            original_labels = g.nodes['rsd'].data['label'][sorted_idx]
            cls_token_id = self.token2id.get('<CLS>', 2)

            cls_token = torch.tensor([cls_token_id], device=original_labels.device) # [1]
            sorted_labels = torch.cat([cls_token, original_labels]) # [1 + num_nodes]

            # generate inverse index so we can restore the original order
            _, inverse_idx = torch.sort(sorted_idx) # [num_nodes]

            all_sorted_labels.append(sorted_labels)
            all_inverse_indices.append(inverse_idx)
            max_len = max(max_len, len(sorted_labels))

        # prepare padded labels & attention mask
        padded_labels = torch.full(
            (batch_size, max_len),
            fill_value=self.token2id['<PAD>'] if self.token2id else 0,
            device=bg.device,
            dtype=torch.long
        ) # [batch_size, max_len]
        key_padding_mask = torch.ones(
            (batch_size, max_len),
            device=bg.device,
            dtype=torch.bool
        ) # [batch_size, max_len], True means to be mask

        # fill each row
        for i, sorted_labels in enumerate(all_sorted_labels):
            seq_len = len(sorted_labels)
            if seq_len == 0:
                continue
            padded_labels[i, :seq_len] = sorted_labels
            key_padding_mask[i, :seq_len] = False

        return padded_labels, key_padding_mask, all_inverse_indices, max_len, graphs

    def forward(self, bg):
        # prepare batched residues
        padded_labels, key_padding_mask, all_inverse_indices, max_len, graphs = \
            self._prepare_residue_batch(bg)
        
        # padded_labels Shape: [batch_size, max_len]
        # attention_mask Shape: [batch_size, max_len, max_len]

        # embedding + positional
        x = self.embedding(padded_labels)          # [batch_size, max_len, embed_dim]
        x = self.pos_emb(x, seq_len=max_len)       # add position embedding

        # conv + Transformer
        encoded = self.encoder(x, key_padding_mask=key_padding_mask)  # [batch_size, max_len, conv_channels]

        # recover node-wise features and gather CLS vectors
        rsd_features = []
        cls_features = []

        for i, g in enumerate(graphs):
            seq_len = len(all_inverse_indices[i]) + 1  # +1 for CLS
            if seq_len <= 1:
                # no real residue nodes
                continue

            sub_enc = encoded[i, :seq_len]  # [seq_len, conv_channels]
            cls_feat = sub_enc[0] # [conv_channels]
            node_feat = sub_enc[1:] # [seq_len-1, conv_channels]
            # reorder node_feat to the original index
            node_feat = node_feat[all_inverse_indices[i]] # [num_nodes, conv_channels]
            rsd_features.append(node_feat)
            cls_features.append(cls_feat)

        # concatenate final node features, stack CLS 
        if len(rsd_features) > 0:
            rsd_features = torch.cat(rsd_features, dim=0) # [total_num_nodes, conv_channels]
            cls_features = torch.stack(cls_features, dim=0) # [num_valid_graphs, conv_channels]
        else:
            rsd_features = torch.zeros((0, encoded.shape[-1]), device=encoded.device)
            cls_features = torch.zeros((0, encoded.shape[-1]), device=encoded.device)

        return rsd_features, cls_features # [total_num_nodes, conv_channels], [num_valid_graphs, conv_channels]

if __name__ == "__main__":
    # Test the ResidueEncoder
    g = dgl.heterograph({
        ('rsd', 's', 'rsd'): ([], [])
    }, num_nodes_dict={'rsd': 2})

    # 添加节点特征
    g.nodes['rsd'].data['label'] = torch.tensor([TOKEN2ID['A'], TOKEN2ID['<ORN>']])
    g.nodes['rsd'].data['pos'] = torch.tensor([0, 1])

    print(g)

    # 创建一个包含单个子图的批处理图
    bg = dgl.batch([g])

    # 使用ResidueEncoder处理
    residue_encoder = ResidueEncoder(vocab_size=VOCAB_SIZE, embed_dim=256)
    output = residue_encoder(bg)

    print(output.shape)  # [2, conv_channels] for there is two residue, Ala, Orn in the graph

    # 测试样例：创建一个包含两个分子的批次图
    g1 = dgl.heterograph({('rsd','s','rsd'): [(0,1), (1,0)]}, num_nodes_dict={'rsd':2})
    g1.nodes['rsd'].data['label'] = torch.LongTensor([1,2])
    g1.nodes['rsd'].data['pos'] = torch.LongTensor([1,2])

    g2 = dgl.heterograph({('rsd','s','rsd'): [(0,1), (1,0)]}, num_nodes_dict={'rsd':2})
    g2.nodes['rsd'].data['label'] = torch.LongTensor([3,4])
    g2.nodes['rsd'].data['pos'] = torch.LongTensor([1,2])

    bg = dgl.batch([g1, g2])
    rsd_emb = ResidueEncoder()(bg)
    print(bg.nodes['rsd'].data['label'])  # 预期输出 [1,2,3,4]
    print(rsd_emb.shape)  # 预期形状 (4, hidden_dim)