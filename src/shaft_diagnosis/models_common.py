"""Multimodal WDCNN and component gates. R state keys are retained for checkpoint compatibility."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
class WDCNNBackbone(nn.Module):

    def __init__(self, c_in: int, feat_len: int=10, dropout: float=0.0, adapt_pool_mode: str='max'):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, 16, kernel_size=64, stride=16, padding=24)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        adapt_pool_mode = str(adapt_pool_mode).strip().lower()
        if adapt_pool_mode == 'avg':
            self.adapt = nn.AdaptiveAvgPool1d(feat_len)
        elif adapt_pool_mode == 'max':
            self.adapt = nn.AdaptiveMaxPool1d(feat_len)
        else:
            raise ValueError(f'Unsupported adapt_pool_mode: {adapt_pool_mode}')
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.adapt(x)
        x = self.drop(x)
        return x.flatten(1)


class A0ComponentGate(nn.Module):

    def __init__(self, feat_dim: int, gate_hidden: int=256, dropout: float=0.0):
        super().__init__()
        self.mlps = nn.ModuleDict({c: nn.Sequential(nn.Linear(3 * feat_dim, gate_hidden), nn.ReLU(), nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(), nn.Linear(gate_hidden, 3)) for c in ('U', 'M', 'R')})

    def forward(self, h_vib: torch.Tensor, h_orb: torch.Tensor, h_acou: torch.Tensor):
        hcat = torch.cat([h_vib, h_orb, h_acou], dim=-1)
        out_h, out_w, out_logits = ({}, {}, {})
        for c in ('U', 'M', 'R'):
            logits = self.mlps[c](hcat)
            w = F.softmax(logits, dim=-1)
            hc = w[:, 0:1] * h_vib + w[:, 1:2] * h_orb + w[:, 2:3] * h_acou
            out_h[c] = hc
            out_w[c] = w
            out_logits[c] = logits
        return (out_h, out_w, out_logits)


class V4MinA0Model(nn.Module):

    def __init__(self, feat_len=10, gate_hidden=256, gate_dropout=0.0, proj_hidden=256, z_dim=128, proj_dropout=0.2, K_U=3, K_M=2, K_R=3, adapt_pool_mode='max'):
        super().__init__()
        self.enc_vib = WDCNNBackbone(c_in=6, feat_len=feat_len, dropout=0.0, adapt_pool_mode=adapt_pool_mode)
        self.enc_orb = WDCNNBackbone(c_in=2, feat_len=feat_len, dropout=0.0, adapt_pool_mode=adapt_pool_mode)
        self.enc_acou = WDCNNBackbone(c_in=2, feat_len=feat_len, dropout=0.0, adapt_pool_mode=adapt_pool_mode)
        feat_dim = 64 * feat_len
        self.gate = A0ComponentGate(feat_dim=feat_dim, gate_hidden=gate_hidden, dropout=gate_dropout)
        self.fc1 = nn.ModuleDict({c: nn.Linear(feat_dim, proj_hidden) for c in ('U', 'M', 'R')})
        self.fc2 = nn.ModuleDict({c: nn.Linear(proj_hidden, z_dim) for c in ('U', 'M', 'R')})
        self.proj_drop = nn.Dropout(proj_dropout) if proj_dropout and proj_dropout > 0 else nn.Identity()
        self.head_exist = nn.ModuleDict({c: nn.Linear(z_dim, 1) for c in ('U', 'M', 'R')})
        self.head_grade = nn.ModuleDict({'U': nn.Linear(z_dim, K_U), 'M': nn.Linear(z_dim, K_M), 'R': nn.Linear(z_dim, K_R)})
        self.p_null = nn.ParameterDict({c: nn.Parameter(torch.zeros(z_dim)) for c in ('U', 'M', 'R')})

    def forward(self, x10: torch.Tensor) -> Dict[str, Any]:
        x_vib = x10[:, 0:6, :]
        x_orb = x10[:, 6:8, :]
        x_acou = x10[:, 8:10, :]
        h_v = self.enc_vib(x_vib)
        h_o = self.enc_orb(x_orb)
        h_a = self.enc_acou(x_acou)
        h_c, gate_w, gate_logits = self.gate(h_v, h_o, h_a)
        z_c, exist_logits, grade_logits = ({}, {}, {})
        for c in ('U', 'M', 'R'):
            u = F.relu(self.fc1[c](h_c[c]))
            u = self.proj_drop(u)
            z = self.fc2[c](u)
            z_c[c] = z
            exist_logits[c] = self.head_exist[c](z).squeeze(-1)
            grade_logits[c] = self.head_grade[c](z)
        return {'gate_w': gate_w, 'z_c': z_c, 'exist_logits': exist_logits, 'grade_logits': grade_logits, 'w': gate_w, 'z': z_c, 'gate_logits': gate_logits, 'h_vib': h_v, 'h_orb': h_o, 'h_acou': h_a}
