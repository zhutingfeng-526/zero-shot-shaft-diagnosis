"""Exact normalization of U network outputs for median aggregation."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
def normalize_output(output: Mapping[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    raw_anchor = output['raw_anchor'].detach().cpu().numpy()
    corrected = output.get('corrected_anchor', output.get('evidence'))
    if corrected is None:
        raise RuntimeError('Transport output lacks corrected anchor')
    shift = output.get('predicted_shift', output.get('bounded_shift'))
    if shift is None:
        shift = output['raw_anchor'] - corrected
    confidence = output.get('confidence', torch.ones_like(output['raw_anchor']))
    uncertainty = output.get('uncertainty', 1.0 - confidence)
    return {'probabilities': output['probabilities'].detach().cpu().numpy(), 'evidence': corrected.detach().cpu().numpy(), 'anchor': raw_anchor, 'deep_evidence': shift.detach().cpu().numpy(), 'confidence': confidence.detach().cpu().numpy(), 'uncertainty': uncertainty.detach().cpu().numpy()}
