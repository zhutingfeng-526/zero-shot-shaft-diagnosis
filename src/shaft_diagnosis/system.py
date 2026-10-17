"""Complete frozen diagnosis on one preprocessed 9.6 s history."""
from pathlib import Path

import numpy as np

from . import integration
from ._routing import physics_feature_matrix
from .impact import ImpactExpert
from .misalignment import MisalignmentExpert
from .structure import StructurePredictor
from .unbalance import UnbalancePredictor


def history_windows(raw):
    """Return the original 281 overlapping, float16-rounded windows.

    Input is a synchronized, already record-filtered (10,48000) array at
    5 kHz. Storage rounding is retained; no filtering or rescaling is done.
    """
    value = np.asarray(raw, dtype=np.float32)
    if value.shape != (10, 48000) or not np.isfinite(value).all():
        raise ValueError("Expected finite preprocessed history (10,48000)")
    if np.max(np.abs(value)) > np.finfo(np.float16).max:
        raise ValueError("History exceeds the original float16 storage range")
    value = value.astype(np.float16).astype(np.float32)
    windows = np.stack([value[:, start:start + 6000] for start in range(0, 42001, 150)])
    return value, windows


class DiagnosisPredictor:
    """Run the final structure, U/M/I and temporal integration configuration.

    One call returns one endpoint. The input must lie within one continuous
    record. Labels and data from other records are not used for inference.
    """

    def __init__(self, repository_root=None, device="cpu"):
        root = Path(repository_root) if repository_root else Path(__file__).absolute().parents[2]
        self.structure = StructurePredictor(root / "models/structure", device)
        self.unbalance = UnbalancePredictor(root / "models/unbalance", device)
        self.misalignment = MisalignmentExpert(root / "models/misalignment", device,
                                               position_mode="final")
        self.impact = ImpactExpert(root / "models/impact", root / "configs/impact.json")
        self.presence, self.config = integration.load_presence_models(root)

    def predict(self, raw, rpm, *, batch_size=16, workers=1,
                condition_id=0, run_id=0, start_sample=0):
        """Return endpoint labels, availability, probabilities and window evidence.

        ``rpm`` is the supplied speed in r/min. ``start_sample`` locates the
        history in its record and is used only for event coordinates. The
        original experiments used 1200 r/min. Runtime and precision settings
        are documented in docs/ENVIRONMENT.md.
        """
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not np.isscalar(rpm) or not np.isfinite(rpm) or rpm <= 0:
            raise ValueError("rpm must be a positive finite scalar")
        history, windows = history_windows(raw)
        evidence = {key: [] for key in ("structure", "structure_index", "u", "mp", "md", "mq")}
        for left in range(0, len(windows), batch_size):
            current = windows[left:left + batch_size]
            speed = np.full(len(current), rpm, dtype=np.float32)
            structure = self.structure.predict(current)
            # U selection uses the base window decision, before temporal routing.
            u = self.unbalance.predict(current, speed, structure["bits"][:, 2].astype(bool))
            m = self.misalignment.predict(current, speed, workers=workers)
            for key, value in (("structure", structure["probabilities"]),
                               ("structure_index", structure["structure_index"]),
                               ("u", u["probabilities"]), ("mp", m["position_prob"]),
                               ("md", m["degree_prob"]), ("mq", m["degree_route_guard"])):
                evidence[key].append(np.asarray(value))
        evidence = {key: np.concatenate(value) for key, value in evidence.items()}
        # The frozen window interface stores consensus probabilities as float32.
        evidence["structure"] = evidence["structure"].astype(np.float32)
        segment, rank, endpoint = np.zeros(281, dtype=np.int64), np.arange(281), np.array([280])
        physical = np.zeros((281, 15), dtype=np.float32)
        selected = integration.context_indices(segment, rank, endpoint, stride=10)[0]
        physical[selected] = physics_feature_matrix(windows[selected], np.full(len(selected), rpm))
        route = integration.route_structure(evidence["structure"], physical, segment,
                                             rank, endpoint, self.presence, self.config)
        _, u_probability = integration.aggregate_u(evidence["u"], segment)
        m_position, m_degree = integration.aggregate_m_context(
            evidence["mp"][None], evidence["md"][None], evidence["mq"][None])
        impact = self.impact.predict(history, rpm, condition_id=condition_id,
                                     run_id=run_id, start_sample=start_sample)
        result = integration.assemble_diagnosis(
            route["route"], u_probability.argmax(1), m_position.argmax(1), m_degree.argmax(1),
            [impact["legacy_prediction"]], [True], [True], [impact["ready"]])
        result.update({
            "u_probability": u_probability,
            "m_position_probability": m_position, "m_degree_probability": m_degree,
            "i_probability": np.asarray(impact["probabilities"])[None],
            "i_event_count": np.array([impact["event_count"]], dtype=np.int32),
            "i_acoustic_check_applied": np.array([impact["c2_applied"]], dtype=bool),
            "structure_refinement_applied": route["applied"],
            "structure_mean_log_score": route["mean_log_score"],
            "standardized_presence": route["standardized_presence"],
            "window_structure_probability": evidence["structure"],
            "window_structure_index": evidence["structure_index"],
            "window_u_probability": evidence["u"],
            "window_m_position_probability": evidence["mp"],
            "window_m_degree_probability": evidence["md"],
            "window_m_degree_quality": evidence["mq"],
        })
        return result
