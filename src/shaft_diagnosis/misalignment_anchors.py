"""Background attribute predictors and spatial reference inputs for M."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



POSITION_CLASSES = 3


DEGREE_CLASSES = 2


def transform(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float32) - center[None, :]) / scale[None, :]
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0).astype(np.float32)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1.0e-12)).astype(np.float32)


class DualAttributeNet(nn.Module):
    def __init__(self, d_position: int, d_degree: int, hidden: int, shared: bool) -> None:
        super().__init__()
        self.shared = bool(shared)
        if self.shared:
            if d_position != d_degree:
                raise ValueError("Shared model requires identical feature dimensions")
            self.shared_encoder = nn.Sequential(
                nn.Linear(d_position, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            )
            self.position_encoder = None
            self.degree_encoder = None
        else:
            self.shared_encoder = None
            self.position_encoder = nn.Sequential(
                nn.Linear(d_position, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            )
            self.degree_encoder = nn.Sequential(
                nn.Linear(d_degree, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            )
        self.position_head = nn.Linear(hidden, POSITION_CLASSES)
        self.degree_head = nn.Linear(hidden, DEGREE_CLASSES)

    def forward(self, x_position: torch.Tensor, x_degree: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.shared:
            z_position = self.shared_encoder(x_position)
            z_degree = z_position
        else:
            z_position = self.position_encoder(x_position)
            z_degree = self.degree_encoder(x_degree)
        return self.position_head(z_position), self.degree_head(z_degree), z_position, z_degree


def prepare_inputs(
    data: Mapping[str, np.ndarray], position_indices: np.ndarray, degree_indices: np.ndarray,
    position_center: np.ndarray, position_scale: np.ndarray,
    degree_center: np.ndarray, degree_scale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data["x"], dtype=np.float32)
    return (
        transform(x[:, position_indices], position_center, position_scale),
        transform(x[:, degree_indices], degree_center, degree_scale),
    )


@torch.no_grad()
def predict_net(
    model: DualAttributeNet, x_position: np.ndarray, x_degree: np.ndarray,
    device: torch.device, batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    position_logits: List[np.ndarray] = []
    degree_logits: List[np.ndarray] = []
    for start in range(0, len(x_position), batch_size):
        stop = min(len(x_position), start + batch_size)
        xp = torch.from_numpy(x_position[start:stop]).to(device)
        xd = torch.from_numpy(x_degree[start:stop]).to(device)
        lp, ld, _, _ = model(xp, xd)
        position_logits.append(lp.cpu().numpy())
        degree_logits.append(ld.cpu().numpy())
    return softmax_numpy(np.concatenate(position_logits)), softmax_numpy(np.concatenate(degree_logits))


def prepare_topology(x: np.ndarray, triplets: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)[:, triplets]
    z = (values - center[None, None, :]) / scale[None, None, :]
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
    return z.astype(np.float32)


class TopologyResidualPosition(nn.Module):
    def __init__(self, feature_count: int, hidden: int, topology_hidden: int) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_count, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, topology_hidden), nn.LayerNorm(topology_hidden), nn.SiLU(),
        )
        self.self_projection = nn.Linear(topology_hidden, topology_hidden, bias=False)
        self.neighbor_projection = nn.Linear(topology_hidden, topology_hidden, bias=False)
        self.coordinate_projection = nn.Linear(1, topology_hidden, bias=False)
        self.residual_head = nn.Linear(topology_hidden, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.register_buffer("coordinates", torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1))

    def forward(
        self, node_features: torch.Tensor, anchor_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relative = node_features - node_features.mean(dim=1, keepdim=True)
        encoded = self.node_encoder(relative)
        neighbor = torch.stack((encoded[:, 1], 0.5 * (encoded[:, 0] + encoded[:, 2]), encoded[:, 1]), dim=1)
        topology = F.silu(
            self.self_projection(encoded)
            + self.neighbor_projection(neighbor)
            + self.coordinate_projection(self.coordinates.expand(len(encoded), -1, -1))
        )
        residual = self.residual_head(topology).squeeze(-1)
        return anchor_logits + residual, topology, residual


def position_prob_from_model(
    model: TopologyResidualPosition, data: Mapping[str, np.ndarray], anchor_prob: np.ndarray,
    meta: Mapping[str, Any], device: torch.device, batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    nodes = prepare_topology(
        data["x"], np.asarray(meta["topology_triplets"], dtype=np.int64),
        np.asarray(meta["topology_center"], dtype=np.float32),
        np.asarray(meta["topology_scale"], dtype=np.float32),
    )
    anchor_logits = np.log(np.maximum(np.asarray(anchor_prob, dtype=np.float32), 1.0e-8))
    probabilities: List[np.ndarray] = []
    residuals: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(nodes), batch_size):
            stop = min(len(nodes), start + batch_size)
            logits, _, residual = model(
                torch.from_numpy(nodes[start:stop]).to(device),
                torch.from_numpy(anchor_logits[start:stop]).to(device),
            )
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            residuals.append(residual.cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32), np.concatenate(residuals).astype(np.float32)


def probability_logits(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float32), 1.0e-6, 1.0)
    logits = np.log(probability)
    return (logits - logits.mean(axis=1, keepdims=True)).astype(np.float32)
