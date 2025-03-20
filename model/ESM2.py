"""
adaptation of esm2 for ambiguous/ncAA feature embedding.
"""

import torch
import torch.nn as nn
import dgl
import esm
from .sequence_embedding import TOKEN2ID

class ESMResidueEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "/root/autodl-tmp/projects/amphgt/ESMpt/esm2_t33_650M_UR50D.pt", # for this, hid_dim 1280
                 ):
        super().__init__()
        # load esm
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.id2token = {idx: tok for tok, idx in TOKEN2ID.items()}

        self.hidden_dim = self.model.embed_dim # hidden_dim in ESM2 is 1280. might project to 300.

    def _token_id_to_esm_char(self, token_id: int) -> str:
        token_str = self.id2token[token_id]
        if token_str in esm.data.proteinseq_toks["toks"]:
            return token_str
        else:
            return "<mask>"

    def _prepare_batch(self, bg: dgl.DGLGraph) -> tuple[list, list, list]:
        graphs = dgl.unbatch(bg)
        batch_sequences = []
        batch_inv_indices = []

        for g in graphs:
            if g.num_nodes('rsd') == 0:
                batch_sequences.append(("<CLS>", [""]))
                batch_inv_indices.append(torch.tensor([], dtype=torch.long))
                continue

            # sort rsd by pos
            sorted_pos, sorted_idx = torch.sort(g.nodes['rsd'].data['pos'])
            original_labels = g.nodes['rsd'].data['label'][sorted_idx]

            # get amino acids sequence character. replace ncaas with mask
            esm_seq = []
            for token_id in original_labels.tolist():
                esm_char = self._token_id_to_esm_char(token_id)
                esm_seq.append(esm_char)
            
            esm_seq_str = ''.join(esm_seq)
            batch_sequences.append(("", esm_seq_str))  # ESM will prepend <cls>
            batch_inv_indices.append(torch.argsort(sorted_idx))  # inverse index for restore original sequecne.

        return batch_sequences, batch_inv_indices
    
    @torch.no_grad()
    def forward(self, bg: dgl.DGLGraph) -> tuple[torch.Tensor, torch.Tensor]:
        batch_sequences, batch_inv_indices = self._prepare_batch(bg)
        
        # batch_converter has added <cls>
        labels, strs, tokens = self.batch_converter(batch_sequences)
        tokens = tokens.to(bg.device)
        
        # get hidden representations
        results = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        hidden_states = results["representations"][self.model.num_layers]
        
        # parse rsd feature and cls for every graph.
        rsd_features = []
        cls_features = []
        for i, g in enumerate(dgl.unbatch(bg)):
            if g.num_nodes('rsd') == 0:
                # when null
                cls_feat = torch.zeros(self.hidden_dim, device=bg.device)
                rsd_features.append(torch.zeros(0, self.hidden_dim, device=bg.device))
                cls_features.append(cls_feat)
                continue
            
            num_rsd = g.num_nodes('rsd')
            
            seq_features = hidden_states[i, :num_rsd + 2]  # +2 for CLS and EOS
            
            cls_feat = seq_features[0]
            cls_features.append(cls_feat)
            
            # extract residue feature and restore origin sequence
            rsd_feats = seq_features[1:-1]  # [num_rsd, hidden_dim]
            rsd_feats = rsd_feats[batch_inv_indices[i]]
            rsd_features.append(rsd_feats)
        
        rsd_features = torch.cat(rsd_features, dim=0) if rsd_features else torch.zeros(0, self.hidden_dim, device=bg.device)
        cls_features = torch.stack(cls_features, dim=0)

        # rsd_features, torch.Size([8025, 1280]), [num_nodes, hid_dim]
        # cls_features, torch.Size([128, 1280]), [B, hid_dim]
        
        return rsd_features, cls_features