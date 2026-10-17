"""Shared encoder features and frozen attribute evidence for the M frontend."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import m_features_p0 as p0



@torch.no_grad()
def k5_batch_output(model: Any, model_cfg: Mapping[str, Any], raw: np.ndarray, device: torch.device) -> Tuple[np.ndarray, List[str]]:
    xb = torch.from_numpy(np.asarray(raw, dtype=np.float32)).to(device, non_blocking=True)
    xb = p0.apply_model_input_norm(xb, str(model_cfg.get("input_norm", "none")))
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        output = model(xb)
    values: List[torch.Tensor] = []
    names: List[str] = []
    for component in ("U", "M", "R"):
        latent = output["z_c"][component].float()
        values.append(latent)
        names.extend(["%s_z_%03d" % (component, index) for index in range(latent.shape[1])])
        values.append(torch.sigmoid(output["exist_logits"][component].float()).unsqueeze(1))
        names.append("%s_exist_prob" % component)
    probabilities = {
        "U_grade": torch.softmax(output["grade_logits"]["U"].float(), dim=1),
        "M_position": torch.softmax(output["m_position_logits"].float(), dim=1),
        "M_degree": torch.softmax(output["m_degree_logits"].float(), dim=1),
        "R_position": torch.softmax(output["grade_logits"]["R"].float(), dim=1),
    }
    for key, probability in probabilities.items():
        values.append(probability)
        names.extend(["%s_prob_%d" % (key, index) for index in range(probability.shape[1])])
    array = torch.cat(values, dim=1).detach().cpu().numpy().astype(np.float32)
    del xb, output, values
    return array, names
