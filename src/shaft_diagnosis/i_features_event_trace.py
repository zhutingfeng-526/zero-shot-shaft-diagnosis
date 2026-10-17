"""Numerical event trace functions."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

NODE_CHANNELS = {
    "B1": (0, 1), "B2": (2, 3), "B3": (4, 5),
    "orbit": (6, 7), "acoustic_A": (8,), "acoustic_B": (9,),
}


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    width = max(1, min(int(width), x.shape[-1]))
    left, right = width // 2, width - 1 - width // 2
    padded = np.pad(x, ((0, 0), (left, right)), mode="reflect")
    csum = np.cumsum(padded, axis=1, dtype=np.float64)
    csum = np.concatenate((np.zeros((len(x), 1)), csum), axis=1)
    return (csum[:, width:] - csum[:, :-width]) / float(width)


def node_event_trace(raw: np.ndarray, channel_scale: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float64)
    z = (x - moving_average(x, 251)) / np.maximum(np.asarray(channel_scale)[:, None], 1.0e-8)
    traces = []
    for width in (5, 17, 49):
        energy = np.sqrt(np.maximum(moving_average(z * z, width), 0.0))
        node_values = np.stack([
            np.sqrt(np.mean(energy[np.asarray(ids)] ** 2, axis=0))
            for ids in NODE_CHANNELS.values()
        ], axis=0)
        # Median prevents one noisy sensor from defining all events; max keeps local impacts visible.
        traces.append(0.55 * np.max(node_values, axis=0) + 0.45 * np.median(node_values, axis=0))
    return np.max(np.stack(traces, axis=0), axis=0)


def close_binary(mask: np.ndarray, gap: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    true_ids = np.where(result)[0]
    if len(true_ids) < 2:
        return result
    for left, right in zip(true_ids[:-1], true_ids[1:]):
        if 0 < int(right - left - 1) <= int(gap):
            result[left:right + 1] = True
    return result


def connected_intervals(mask: np.ndarray, minimum: int) -> List[Tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    delta = np.diff(padded)
    starts, ends = np.where(delta == 1)[0], np.where(delta == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if int(b - a) >= int(minimum)]

def _bind(modules):
    globals().update(modules)
