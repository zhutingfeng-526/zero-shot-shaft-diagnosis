"""Final same-space structure model with six M classes, auxiliary M heads and fault-count head."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models_common import V4MinA0Model

class FaultCountHead(torch.nn.Module):

    def __init__(self, z_dim: int, hidden_dim: int=128, dropout: float=0.0, num_classes: int=3):
        super().__init__()
        self.fc1 = torch.nn.Linear(3 * int(z_dim), int(hidden_dim))
        self.drop = torch.nn.Dropout(float(dropout)) if float(dropout) > 0 else torch.nn.Identity()
        self.fc2 = torch.nn.Linear(int(hidden_dim), int(num_classes))

    def forward(self, z_c: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = torch.cat([z_c['U'], z_c['M'], z_c['R']], dim=-1)
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        return self.fc2(h)


class RealSemanticSameSpaceModel(torch.nn.Module):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        base_cfg = dict(cfg)
        base_cfg['K_U'] = 3
        base_cfg['K_M'] = 6
        base_cfg['K_R'] = 3
        self.base_model = V4MinA0Model(feat_len=int(base_cfg['feat_len']), gate_hidden=int(base_cfg['gate_hidden']), gate_dropout=float(base_cfg.get('gate_dropout', 0.0)), proj_hidden=int(base_cfg['proj_hidden']), z_dim=int(base_cfg['z_dim']), proj_dropout=float(base_cfg.get('proj_dropout', 0.2)), K_U=3, K_M=6, K_R=3, adapt_pool_mode=str(base_cfg.get('adapt_pool_mode', 'max')))
        z_dim = int(base_cfg['z_dim'])
        self.head_m_position = torch.nn.Linear(z_dim, 3)
        self.head_m_degree = torch.nn.Linear(z_dim, 2)
        self.count_head = FaultCountHead(z_dim=z_dim, hidden_dim=int(cfg.get('count_hidden', 128)), dropout=float(cfg.get('count_dropout', 0.0)), num_classes=int(cfg.get('count_num_classes', 3)))

    @property
    def p_null(self):
        return self.base_model.p_null

    def forward(self, x10: torch.Tensor) -> Dict[str, Any]:
        out = self.base_model(x10)
        z_m = out['z_c']['M']
        out['m_position_logits'] = self.head_m_position(z_m)
        out['m_degree_logits'] = self.head_m_degree(z_m)
        out['count_logits'] = self.count_head(out['z_c'])
        return out
