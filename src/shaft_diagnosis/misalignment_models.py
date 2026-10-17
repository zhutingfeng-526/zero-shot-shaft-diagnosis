"""Neural modules for M spatial proposals and per-rotation residual readout."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



POSITION_CLASSES = 3


DEGREE_CLASSES = 2


NODE_NAMES = ("B1", "B2", "B3", "O12", "A1", "A3")


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden)), nn.LayerNorm(int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)), nn.LayerNorm(int(hidden)), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TypedTransportLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.relation = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(3)])
        self.dynamic_gate = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self, state: torch.Tensor, observed: torch.Tensor,
        edges: Sequence[Tuple[int, int, int]], relations: Set[int], mode: str,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        directed: List[Tuple[int, int, int]] = []
        for left, right, relation in edges:
            if int(relation) in relations:
                directed.append((int(left), int(right), int(relation)))
                directed.append((int(right), int(left), int(relation)))
        if not directed:
            return state, []
        aggregate = state.new_zeros(state.shape)
        normalizer = state.new_zeros(state.shape[:2])
        diagnostics: List[Dict[str, Any]] = []
        for source, target, relation in directed:
            source_state = state[:, source]
            source_obs, target_obs = observed[:, source], observed[:, target]
            if mode == "static":
                weight = source_state.new_ones((len(source_state), 1))
            elif mode == "dynamic":
                pair = torch.cat((source_obs, target_obs, torch.abs(source_obs - target_obs), source_obs * target_obs), dim=1)
                weight = torch.sigmoid(self.dynamic_gate(pair))
            else:
                raise ValueError("Unsupported transport mode: %s" % mode)
            message = self.relation[relation](source_state) * weight
            aggregate[:, target] = aggregate[:, target] + message
            normalizer[:, target] = normalizer[:, target] + weight.squeeze(1)
            diagnostics.append({
                "source": source, "target": target, "relation": relation,
                "weight_mean": weight.mean(),
            })
        active = (normalizer > 0.0).unsqueeze(2)
        aggregate = aggregate / torch.clamp(normalizer.unsqueeze(2), min=1.0)
        updated = self.norm(state + self.update(torch.cat((state, aggregate), dim=2)))
        # A node with no incoming edge in this relation phase must remain
        # unchanged.  Otherwise affine biases create evidence at disconnected
        # nodes and invalidate the candidate-source transport interpretation.
        return torch.where(active, updated, state), diagnostics


class HypothesisTransportCore(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.observation_refine = TypedTransportLayer(hidden)
        self.mechanical_1 = TypedTransportLayer(hidden)
        self.mechanical_2 = TypedTransportLayer(hidden)
        self.observation_predict = TypedTransportLayer(hidden)
        self.source_token = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.source_token, mean=0.0, std=0.02)
        self.reconstruct = nn.Linear(hidden, hidden)

    def forward(
        self, observed: torch.Tensor, edges: Sequence[Tuple[int, int, int]], mode: str,
        node_quality: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        refined, diagnostics = self.observation_refine(observed, observed, edges, {1, 2}, mode)
        energies: List[torch.Tensor] = []
        states: List[torch.Tensor] = []
        observed_unit = F.normalize(refined, dim=2, eps=1.0e-6)
        for candidate in range(POSITION_CLASSES):
            state = refined.new_zeros(refined.shape)
            state[:, candidate] = refined[:, candidate] + self.source_token.unsqueeze(0)
            state, rows = self.mechanical_1(state, refined, edges, {0}, mode)
            diagnostics.extend(rows)
            state, rows = self.mechanical_2(state, refined, edges, {0}, mode)
            diagnostics.extend(rows)
            state, rows = self.observation_predict(state, refined, edges, {1, 2}, mode)
            diagnostics.extend(rows)
            predicted = F.normalize(self.reconstruct(state), dim=2, eps=1.0e-6)
            distance = 1.0 - torch.sum(predicted * observed_unit, dim=2)
            mask = torch.ones_like(distance)
            mask[:, candidate] = 0.0
            quality = node_quality * mask
            quality = quality / torch.clamp(quality.sum(dim=1, keepdim=True), min=1.0e-6)
            energies.append(torch.sum(quality * distance, dim=1))
            states.append(state)
        energy = torch.stack(energies, dim=1)
        state_stack = torch.stack(states, dim=1)
        return refined, state_stack, energy, diagnostics


def entropy(probability: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probability * torch.log(torch.clamp(probability, min=1.0e-8)), dim=1, keepdim=True)


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    middle = 0.5 * (left + right)
    left_kl = torch.sum(left * (torch.log(torch.clamp(left, min=1.0e-8)) - torch.log(torch.clamp(middle, min=1.0e-8))), dim=1, keepdim=True)
    right_kl = torch.sum(right * (torch.log(torch.clamp(right, min=1.0e-8)) - torch.log(torch.clamp(middle, min=1.0e-8))), dim=1, keepdim=True)
    return 0.5 * (left_kl + right_kl)


class PositionTopologyAdapter(nn.Module):
    def __init__(
        self, dims: Mapping[str, int], hidden: int, spec: Mapping[str, Any],
        edges: Sequence[Tuple[int, int, int]],
    ) -> None:
        super().__init__()
        self.spec = dict(spec)
        self.edges = [(int(a), int(b), int(r)) for a, b, r in edges]
        self.hidden = int(hidden)
        self.station_encoder = FeatureEncoder(int(dims["station"]), hidden)
        self.orbit_encoder = FeatureEncoder(int(dims["orbit"]), hidden)
        self.acoustic_encoder = FeatureEncoder(int(dims["acoustic"]), hidden)
        self.station_raw_encoder = FeatureEncoder(int(dims["station_raw"]), hidden)
        self.orbit_raw_encoder = FeatureEncoder(int(dims["orbit_raw"]), hidden)
        self.acoustic_raw_encoder = FeatureEncoder(int(dims["acoustic_raw"]), hidden)
        self.global_position_encoder = FeatureEncoder(int(dims["global_position"]), hidden)
        self.event_encoder = FeatureEncoder(int(dims["event_context"]), hidden)
        self.node_type_embedding = nn.Parameter(torch.zeros(len(NODE_NAMES), hidden))
        nn.init.normal_(self.node_type_embedding, mean=0.0, std=0.02)
        self.node_quality = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.transport = HypothesisTransportCore(hidden)
        self.energy_log_scale = nn.Parameter(torch.tensor(math.log(5.0), dtype=torch.float32))
        self.diff_head = nn.Sequential(nn.Linear(hidden * 6, hidden), nn.SiLU(), nn.Linear(hidden, POSITION_CLASSES))
        residual_dim = hidden * 2 + POSITION_CLASSES * 2 + 5
        self.position_residual = nn.Sequential(
            nn.Linear(residual_dim, hidden), nn.SiLU(), nn.Linear(hidden, POSITION_CLASSES),
        )
        nn.init.zeros_(self.position_residual[-1].weight)
        nn.init.zeros_(self.position_residual[-1].bias)

    def encode_nodes(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        station = self.station_encoder(batch["station"])
        orbit = self.orbit_encoder(batch["orbit"]).unsqueeze(1)
        acoustic = self.acoustic_encoder(batch["acoustic"])
        station_raw = self.station_raw_encoder(batch["station_raw"])
        orbit_raw = self.orbit_raw_encoder(batch["orbit_raw"]).unsqueeze(1)
        acoustic_raw = self.acoustic_raw_encoder(batch["acoustic_raw"])
        return (
            torch.cat((station, orbit, acoustic), dim=1)
            + torch.cat((station_raw, orbit_raw, acoustic_raw), dim=1)
            + self.node_type_embedding.unsqueeze(0)
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
        observed = self.encode_nodes(batch)
        global_position = self.global_position_encoder(batch["global_position"])
        event = self.event_encoder(batch["event_context"])
        quality_logits = self.node_quality(observed).squeeze(2)
        node_quality = torch.softmax(quality_logits, dim=1)
        diagnostics: List[Dict[str, Any]] = []
        if str(self.spec["core_mode"]) == "none":
            refined = observed
            energy = observed.new_zeros((len(observed), POSITION_CLASSES))
        else:
            refined, _states, energy, diagnostics = self.transport(
                observed, self.edges, str(self.spec["edge_mode"]), node_quality,
            )
        stations = refined[:, :3]
        differential = torch.cat((
            stations[:, 0] - stations[:, 1],
            stations[:, 1] - stations[:, 2],
            stations[:, 0] - stations.mean(dim=1),
            stations[:, 1] - stations.mean(dim=1),
            stations[:, 2] - stations.mean(dim=1),
        ), dim=1)
        scale = torch.clamp(torch.exp(self.energy_log_scale), min=1.0, max=20.0)
        graph_position_logits = -scale * energy + self.diff_head(torch.cat((differential, global_position), dim=1))
        anchor_logits = batch["position_anchor_logits"]
        anchor_probability = torch.softmax(anchor_logits, dim=1)
        graph_probability = torch.softmax(graph_position_logits, dim=1)
        anchor_uncertainty = entropy(anchor_probability) / math.log(float(POSITION_CLASSES))
        graph_confidence = 1.0 - entropy(graph_probability) / math.log(float(POSITION_CLASSES))
        disagreement = js_divergence(anchor_probability, graph_probability) / math.log(2.0)
        evidence_gate = torch.clamp(graph_confidence * torch.clamp(anchor_uncertainty + disagreement, 0.0, 1.0), 0.0, 1.0)
        sorted_energy, _ = torch.sort(energy, dim=1)
        energy_min = sorted_energy[:, :1]
        energy_margin = sorted_energy[:, 1:2] - sorted_energy[:, :1]
        residual_input = torch.cat((
            global_position, event, anchor_logits, graph_position_logits,
            energy_min, energy_margin, anchor_uncertainty, graph_confidence,
            node_quality.max(dim=1, keepdim=True).values,
        ), dim=1)
        position_delta = torch.tanh(self.position_residual(residual_input))
        position_logits = anchor_logits + evidence_gate * position_delta
        degree_logits = batch["degree_anchor_logits"]
        return {
            "position_logits": position_logits,
            "degree_logits": degree_logits,
            "graph_position_logits": graph_position_logits,
            "position_delta": position_delta,
            "position_gate": evidence_gate,
            "candidate_energy": energy,
            "energy_scale": scale.expand(len(observed), 1),
            "node_quality": node_quality,
            "edge_diagnostics": diagnostics,
        }


OBSERVATIONS = ("V_B1", "V_B2", "V_B3", "O_12", "A_B1", "A_B3")


MODALITY_IDS = (0, 0, 0, 1, 2, 2)


FROZEN_COUNTERFACTUALS: Tuple[Dict[str, str], ...] = (
    {"id": "F0_Native", "mode": "native"},
    {"id": "F1_OrbitShiftB2B3", "mode": "orbit_shift"},
    {"id": "F2_AcousticEndpointSwap", "mode": "acoustic_swap"},
    {"id": "F3_DropOrbit", "mode": "drop_orbit"},
    {"id": "F4_DropAcoustic", "mode": "drop_acoustic"},
    {"id": "F5_RandomTargetPermutation", "mode": "permute_targets"},
    {"id": "F6_DropExtraObservers", "mode": "drop_extra"},
    {"id": "F7_NoNullAdmission", "mode": "no_null"},
    {"id": "F8_IdentityAllObservationsDropped", "mode": "identity"},
)


def allowed_mask(layout: str, mode: str = "native") -> np.ndarray:
    mask = np.zeros((len(OBSERVATIONS), POSITION_CLASSES + 1), dtype=np.bool_)
    if layout == "all_to_all":
        mask[:, :] = True
    else:
        mask[0, 0] = True; mask[1, 1] = True; mask[2, 2] = True
        mask[3, [0, 1, 3]] = True
        mask[4, :] = True; mask[5, :] = True
    if mode == "orbit_shift":
        mask[3, :] = False; mask[3, [1, 2, 3]] = True
    if mode == "no_null":
        mask[:, 3] = False
    if np.any(np.sum(mask, axis=1) == 0):
        raise RuntimeError("Every observation needs at least one admissible target")
    return mask


def relation_attributes(layout: str, mode: str = "native") -> np.ndarray:
    target_position = np.asarray([0.0, 0.5, 1.0, 0.5], dtype=np.float32)
    observation_position = np.asarray([0.0, 0.5, 1.0, 0.25, 0.0, 1.0], dtype=np.float32)
    modality = np.asarray([0, 0, 0, 1, 2, 2], dtype=np.int64)
    attrs = np.zeros((len(OBSERVATIONS), POSITION_CLASSES + 1, 9), dtype=np.float32)
    for observation in range(len(OBSERVATIONS)):
        for target in range(POSITION_CLASSES + 1):
            null = float(target == POSITION_CLASSES)
            distance = abs(float(observation_position[observation] - target_position[target]))
            local = float(observation < 3 and target == observation)
            span = float(observation == 3 and target in (0, 1))
            endpoint = -1.0 if observation == 4 else 1.0 if observation == 5 else 0.0
            attrs[observation, target] = np.asarray([
                distance, local, span, null,
                float(modality[observation] == 0), float(modality[observation] == 1),
                float(modality[observation] == 2), endpoint,
                float(layout == "all_to_all"),
            ], dtype=np.float32)
    if mode == "orbit_shift":
        attrs[3, :, 2] = 0.0; attrs[3, [1, 2], 2] = 1.0
    return attrs


class SafeInnovationRoutingAdapter(nn.Module):
    def __init__(self, a0: nn.Module, hidden: int, spec: Mapping[str, Any]) -> None:
        super().__init__()
        self.a0 = a0
        self.hidden = int(hidden)
        self.spec = dict(spec)
        if int(getattr(a0, "hidden", hidden)) != int(hidden):
            raise RuntimeError("P4J hidden dimension must equal frozen A0 hidden dimension")
        self.runtime_mode = "native"
        self._cached_observed: Optional[torch.Tensor] = None
        self._capture_handle = self.a0.node_quality.register_forward_pre_hook(self._capture_observed)
        self.modality_embedding = nn.Embedding(3, hidden)
        self.null_target = nn.Parameter(torch.zeros(hidden))
        self.predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(3)
        ])
        pair_dim = hidden * 4 + 9
        self.route_net = nn.Sequential(nn.Linear(pair_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.reliability_net = nn.Sequential(nn.Linear(pair_dim + 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.message_net = nn.Sequential(nn.Linear(pair_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        self.alpha_raw = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.null_target, mean=0.0, std=0.02)
        nn.init.normal_(self.score_head[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.score_head[-1].bias)
        self.register_buffer("modality_index", torch.tensor([0, 0, 0, 1, 2, 2], dtype=torch.long))
        self.register_buffer("native_mask", torch.from_numpy(allowed_mask(str(spec["layout"]))))
        self.register_buffer("native_attrs", torch.from_numpy(relation_attributes(str(spec["layout"]))))

    def _capture_observed(self, _module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        self._cached_observed = inputs[0].detach()

    def train(self, mode: bool = True) -> "SafeInnovationRoutingAdapter":
        super().train(mode)
        self.a0.eval()
        return self

    def close(self) -> None:
        if self._capture_handle is not None:
            self._capture_handle.remove()
            self._capture_handle = None
        self._cached_observed = None

    def set_counterfactual(self, mode: str) -> None:
        valid = {str(row["mode"]) for row in FROZEN_COUNTERFACTUALS}
        if mode not in valid:
            raise KeyError("Unknown P4J counterfactual: %s" % mode)
        self.runtime_mode = str(mode)

    def _runtime_contract(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        layout = str(self.spec["layout"])
        mask_mode = self.runtime_mode if self.runtime_mode in ("orbit_shift", "no_null") else "native"
        if mask_mode == "native":
            return self.native_mask.to(device=device), self.native_attrs.to(device=device)
        mask = torch.from_numpy(allowed_mask(layout, mask_mode)).to(device=device)
        attrs = torch.from_numpy(relation_attributes(layout, mask_mode)).to(device=device)
        return mask, attrs

    def _fixed_routes(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        route = mask.to(dtype=dtype)
        # A fixed physical route has no data-driven basis for assigning a
        # nonzero null prior.  Null admission is therefore reserved for the
        # learned route; fixed routing averages only admissible real bearings.
        route[:, POSITION_CLASSES] = 0.0
        route = route / torch.clamp(route.sum(dim=1, keepdim=True), min=1.0)
        return route

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
        self.a0.eval(); self._cached_observed = None
        with torch.no_grad():
            base = self.a0(batch)
        observed = self._cached_observed
        self._cached_observed = None
        if observed is None or observed.shape[1] != len(OBSERVATIONS):
            raise RuntimeError("Frozen A0 observation capture failed")
        if self.runtime_mode == "acoustic_swap":
            observed = observed[:, [0, 1, 2, 3, 5, 4], :]
        stations = observed[:, :POSITION_CLASSES, :]
        batch_size = int(observed.shape[0])
        null = self.null_target.view(1, 1, -1).expand(batch_size, 1, -1)
        targets = torch.cat((stations, null), dim=1)
        mask, attrs = self._runtime_contract(observed.device)
        obs = observed[:, :, None, :].expand(-1, -1, POSITION_CLASSES + 1, -1)
        target = targets[:, None, :, :].expand(-1, len(OBSERVATIONS), -1, -1)
        modality = self.modality_embedding(self.modality_index)
        obs_typed = obs + modality[None, :, None, :]
        predicted_parts: List[torch.Tensor] = []
        for observation in range(len(OBSERVATIONS)):
            predictor = self.predictors[MODALITY_IDS[observation]]
            predicted_parts.append(predictor(target[:, observation]))
        predicted = torch.stack(predicted_parts, dim=1)
        innovation = obs_typed - predicted
        if not bool(self.spec["innovation"]):
            innovation = torch.zeros_like(innovation)
        pair_attrs = attrs[None, :, :, :].expand(batch_size, -1, -1, -1)
        pair = torch.cat((obs_typed, target, innovation, torch.abs(innovation), pair_attrs), dim=3)
        route_logits = self.route_net(pair).squeeze(3)
        route_logits = route_logits.masked_fill(~mask[None, :, :], -1.0e4)
        if str(self.spec["routing"]) == "learned":
            route = torch.softmax(route_logits, dim=2)
        elif str(self.spec["routing"]) in ("fixed", "pooled"):
            if str(self.spec["routing"]) == "pooled":
                pooled_mask = torch.zeros_like(mask); pooled_mask[:, :POSITION_CLASSES] = True
                route = self._fixed_routes(pooled_mask, observed.dtype)[None].expand(batch_size, -1, -1)
            else:
                route = self._fixed_routes(mask, observed.dtype)[None].expand(batch_size, -1, -1)
        else:
            raise KeyError(str(self.spec["routing"]))
        if self.runtime_mode == "permute_targets":
            route = torch.cat((route[:, :, [2, 0, 1]], route[:, :, 3:4]), dim=2)
        innovation_norm = torch.mean(innovation.square(), dim=3, keepdim=True)
        reconstruction_quality = torch.exp(-torch.clamp(innovation_norm, min=0.0, max=20.0))
        quality_input = torch.cat((pair, innovation_norm, torch.sqrt(innovation_norm + 1.0e-8)), dim=3)
        learned_quality = torch.sigmoid(self.reliability_net(quality_input))
        if str(self.spec["reliability"]) == "uniform":
            reliability = torch.ones_like(learned_quality)
        else:
            reliability = torch.sqrt(torch.clamp(learned_quality * reconstruction_quality, min=1.0e-8))
        availability = observed.new_ones((batch_size, len(OBSERVATIONS), 1, 1))
        if self.runtime_mode in ("drop_orbit", "drop_extra"):
            availability[:, 3] = 0.0
        if self.runtime_mode in ("drop_acoustic", "drop_extra"):
            availability[:, 4:6] = 0.0
        message = self.message_net(pair) * availability
        weighted = route[:, :, :, None] * reliability * message
        bearing_sum = weighted[:, :, :POSITION_CLASSES, :].sum(dim=1)
        bearing_mass = (
            route[:, :, :POSITION_CLASSES, None]
            * reliability[:, :, :POSITION_CLASSES, :]
            * availability
        ).sum(dim=1)
        bearing_evidence = bearing_sum / torch.clamp(bearing_mass, min=1.0e-4)
        score_input = torch.cat((stations, bearing_evidence, stations - bearing_evidence, torch.abs(stations - bearing_evidence)), dim=2)
        raw_delta = self.score_head(score_input).squeeze(2)
        raw_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
        bounded_delta = torch.tanh(raw_delta)
        scale = torch.tanh(self.alpha_raw)
        if self.runtime_mode == "identity":
            scale = scale * 0.0
        correction = scale * bounded_delta
        position_logits = base["position_logits"] + correction
        proposal_logits = base["position_logits"] + bounded_delta
        reconstruction_error = torch.mean(innovation.square(), dim=3, keepdim=True)
        reconstruction_weight = route[:, :, :, None] * reliability * availability
        weighted_reconstruction = torch.sum(reconstruction_weight * reconstruction_error) / torch.clamp(
            reconstruction_weight.sum(), min=1.0e-4,
        )
        return {
            "position_logits": position_logits,
            "degree_logits": base["degree_logits"],
            "base_position_logits": base["position_logits"],
            "proposal_position_logits": proposal_logits,
            "position_delta": bounded_delta,
            "position_gate": torch.abs(scale).expand(batch_size, 1),
            "correction": correction,
            "route_weights": route,
            "reliability": reliability.squeeze(3),
            "null_mass": route[:, :, 3],
            "innovation_l2": torch.sqrt(torch.mean(innovation.square(), dim=3) + 1.0e-8),
            "reconstruction_loss": weighted_reconstruction,
            "alpha": scale.expand(batch_size, 1),
        }


def load_adapter_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(dict(state), strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [key for key in incompatible.missing_keys if not key.startswith("a0.")]
    if unexpected or invalid_missing:
        raise RuntimeError("Adapter state mismatch missing=%s unexpected=%s" % (invalid_missing, unexpected))


class TaskSetEncoder(nn.Module):
    def __init__(
        self, feature_dim: int, state_dim: int, hidden: int, mode: str,
        position_bound: float, degree_bound: float,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.direct = self.mode in ("J0", "J1")
        self.shared_route = self.mode == "J0"
        self.use_film = self.mode == "J3"
        self.position_bound = float(position_bound)
        self.degree_bound = float(degree_bound)
        self.feature = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.SiLU())
        self.state = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.position_score = nn.Linear(hidden, 1)
        self.degree_score = self.position_score if self.shared_route else nn.Linear(hidden, 1)
        self.position_film = nn.Linear(state_dim, hidden * 2)
        self.degree_film = nn.Linear(state_dim, hidden * 2)
        self.position_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, POSITION_CLASSES))
        self.degree_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, DEGREE_CLASSES))
        if not self.direct:
            nn.init.zeros_(self.position_head[-1].weight); nn.init.zeros_(self.position_head[-1].bias)
            nn.init.zeros_(self.degree_head[-1].weight); nn.init.zeros_(self.degree_head[-1].bias)
        if self.use_film:
            nn.init.zeros_(self.position_film.weight); nn.init.zeros_(self.position_film.bias)
            nn.init.zeros_(self.degree_film.weight); nn.init.zeros_(self.degree_film.bias)

    @staticmethod
    def masked_attention(hidden: torch.Tensor, score: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = score.squeeze(-1).masked_fill(~mask, -1.0e4)
        weight = torch.softmax(logits, dim=1) * mask.float()
        weight = weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=1.0e-8)
        pooled = torch.sum(weight.unsqueeze(-1) * hidden, dim=1)
        return pooled, weight

    def forward(
        self, tokens: torch.Tensor, states: torch.Tensor,
        position_valid: torch.Tensor, degree_valid: torch.Tensor,
        base_position: torch.Tensor, base_degree: torch.Tensor, gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.feature(tokens) + 0.25 * torch.tanh(self.state(states))
        position_hidden, position_attention = self.masked_attention(hidden, self.position_score(hidden), position_valid)
        degree_hidden, degree_attention = self.masked_attention(hidden, self.degree_score(hidden), degree_valid)
        valid = degree_valid.float()
        summary = torch.sum(states * valid.unsqueeze(-1), dim=1) / torch.clamp(valid.sum(dim=1, keepdim=True), min=1.0)
        if self.use_film:
            p_gamma, p_beta = torch.chunk(self.position_film(summary), 2, dim=1)
            d_gamma, d_beta = torch.chunk(self.degree_film(summary), 2, dim=1)
            position_hidden = position_hidden * (1.0 + 0.20 * torch.tanh(p_gamma)) + 0.20 * torch.tanh(p_beta)
            degree_hidden = degree_hidden * (1.0 + 0.20 * torch.tanh(d_gamma)) + 0.20 * torch.tanh(d_beta)
        position_raw = self.position_head(position_hidden)
        degree_raw = self.degree_head(degree_hidden)
        if self.direct:
            position_logits = position_raw
            degree_logits = degree_raw
            position_delta = position_logits - base_position
            degree_delta = degree_logits - base_degree
        else:
            position_delta = self.position_bound * torch.tanh(position_raw)
            degree_delta = self.degree_bound * torch.tanh(degree_raw)
            position_delta = position_delta - position_delta.mean(dim=1, keepdim=True)
            degree_delta = degree_delta - degree_delta.mean(dim=1, keepdim=True)
            position_delta = gate.unsqueeze(1) * position_delta
            degree_delta = gate.unsqueeze(1) * degree_delta
            position_logits = base_position + position_delta
            degree_logits = base_degree + degree_delta
        return {
            "position_logits": position_logits,
            "degree_logits": degree_logits,
            "position_delta": position_delta,
            "degree_delta": degree_delta,
            "position_attention": position_attention,
            "degree_attention": degree_attention,
            "state_summary": summary,
        }
