"""Numerical event detection functions."""
from __future__ import annotations
import argparse
import hashlib
import json
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple
import numpy as np
import torch

def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(current) for key, current in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(current) for current in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_hash(value: Any) -> str:
    payload = json.dumps(jsonable(value), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Fragment:
    cond_int: int
    run_id: int
    fragment_id: int
    start_sample: int
    raw: np.ndarray
    window_indices: np.ndarray
    window_starts: np.ndarray


@dataclass(frozen=True)
class Detection:
    center: int
    score: float
    width: int
    chunk_id: int
    left: int
    right: int


def chunk_starts(length: int, size: int, overlap: int) -> List[int]:
    if length <= size:
        return [0]
    hop = int(size) - int(overlap)
    starts = list(range(0, length - size + 1, hop))
    if starts[-1] != length - size:
        starts.append(length - size)
    return sorted(set(starts))


def detect_fragment(
    fragment: Fragment,
    detector: Mapping[str, Any],
    rpm: int,
    args: argparse.Namespace,
) -> List[Detection]:
    detections: List[Detection] = []
    starts = chunk_starts(
        fragment.raw.shape[1], int(args.detector_chunk_samples), int(args.detector_chunk_overlap)
    )
    for chunk_id, start in enumerate(starts):
        end = min(fragment.raw.shape[1], start + int(args.detector_chunk_samples))
        value = fragment.raw[:, start:end]
        if value.shape[1] < 3 * r0.p4zc.samples_per_rotation(int(rpm)):
            continue
        trace = r0.p4zc.detector_trace(
            value, int(rpm), np.asarray(detector["channel_scale"], dtype=np.float64), str(detector["mode"])
        )
        states = r0.p4zc.trace_states(trace, int(rpm), detector, args)
        for left, right in states["intervals"]:
            if int(right) <= int(left):
                continue
            local = int(left) + int(np.argmax(trace[int(left): int(right)]))
            detections.append(Detection(
                center=int(fragment.start_sample + start + local),
                score=float(trace[local]), width=int(right) - int(left), chunk_id=int(chunk_id),
                left=int(fragment.start_sample + start + int(left)),
                right=int(fragment.start_sample + start + int(right)),
            ))
    return detections


def cluster_unique_events(
    detections: Sequence[Detection], tolerance: int
) -> List[Detection]:
    clusters: List[List[Detection]] = []
    for row in sorted(detections, key=lambda current: (current.center, -current.score, current.chunk_id)):
        choices: List[Tuple[int, int]] = []
        for index, cluster in enumerate(clusters):
            if any(existing.chunk_id == row.chunk_id for existing in cluster):
                continue
            center = int(round(np.median([existing.center for existing in cluster])))
            distance = abs(int(row.center) - center)
            if distance <= int(tolerance):
                choices.append((distance, index))
        if choices:
            clusters[min(choices)[1]].append(row)
        else:
            clusters.append([row])
    representatives = [max(cluster, key=lambda current: (current.score, -current.width)) for cluster in clusters]
    return sorted(representatives, key=lambda row: row.center)

def _bind(modules):
    globals().update(modules)
