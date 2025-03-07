import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ESM2 import ESMResidueEncoder
from .sequence_embedding import ResidueEncoder, VOCAB_SIZE

class ESM_CovT_Ensemble(nn.Module):
    def __init__(
        self,
        hid_dim,
        device,
        ):
        super().__init__()
        self.esm_encoder = ESMResidueEncoder().to(device=device)  # 冻结
        self.cov_encoder = ResidueEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=hid_dim,
            conv_channels=hid_dim,
            ff_dim=hid_dim * 4,
        ).to(device=device)

        self.esm_dim = ESMResidueEncoder().hidden_dim
        self.conv_dim = hid_dim

        self.esm_proj = nn.Sequential(
            nn.Linear(self.esm_dim, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
        )

        # Gating
        self.gate_linear = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, bg: dgl.DGLGraph):
        self.esm_encoder.eval()
        with torch.no_grad():
            rsd_esm, cls_esm = self.esm_encoder(bg)
            # rsd_esm: [N, esm_dim], cls_esm: [B, esm_dim]

        rsd_cov, cls_cov = self.cov_encoder(bg)  
        # rsd_cov: [N, hid_dim], cls_cov: [B, hid_dim]

        rsd_esm_proj = self.esm_proj(rsd_esm)  # [N, hid_dim]
        cls_esm_proj = self.esm_proj(cls_esm)  # [B, hid_dim]

        # gating
        # residue-level
        if rsd_cov.size(0) > 0:
            # 先拼接
            cat_rsd = torch.cat([rsd_cov, rsd_esm_proj], dim=-1)  # [N, 2*hid_dim]

            # gate: [N, hid_dim]
            gate_rsd = torch.sigmoid(self.gate_linear(cat_rsd))

            # fused_rsd = gate_rsd * rsd_cov + (1 - gate_rsd) * rsd_esm_proj
            fused_rsd = gate_rsd * rsd_cov + (1.0 - gate_rsd) * rsd_esm_proj
        else:
            fused_rsd = torch.zeros_like(rsd_cov, device=bg.device)

        # cls-level
        cat_cls = torch.cat([cls_cov, cls_esm_proj], dim=-1)  # [B, 2*hid_dim]
        gate_cls = torch.sigmoid(self.gate_linear(cat_cls))   # [B, hid_dim]
        fused_cls = gate_cls * cls_cov + (1.0 - gate_cls) * cls_esm_proj

        return fused_rsd, fused_cls