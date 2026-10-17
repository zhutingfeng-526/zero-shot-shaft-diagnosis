"""P4J adapter objective on caller-supplied clean/stressed forward packets.

This preserves the final adapter loss. Dataset sampling, stress scheduling,
optimizer/SWA orchestration and source acceptance are not a training entry point.
"""
import torch
import torch.nn.functional as F
from .misalignment_training import weighted_cross_entropy, class_domain_centroid_loss, anchor_kl

LOSS_WEIGHTS = {
    'proposal_weight': 0.25,
    'identity_weight': 0.03,
    'anchor_preserve_weight': 0.08,
    'stress_consistency_weight': 0.06,
    'minimal_intervention_weight': 0.004,
    'reconstruction_weight': 0.02,
}


def routing_adapter_loss(result, position, sample_weight, domains, stressed=None):
    """Original P4J loss; pass stressed on the original every-second-batch schedule."""
    final_ce = weighted_cross_entropy(result['position_logits'], position, sample_weight)
    proposal_ce = weighted_cross_entropy(result['proposal_position_logits'], position, sample_weight)
    identity = class_domain_centroid_loss(result['position_logits'], position, domains, 3)
    source_mask = domains == 0
    preservation = anchor_kl(result['position_logits'], result['base_position_logits'], source_mask)
    intervention = torch.mean(result['correction'].square())
    stress_loss = result['position_logits'].new_zeros(())
    if stressed is not None:
        stressed_ce = weighted_cross_entropy(stressed['position_logits'], position, sample_weight)
        target = torch.softmax(result['position_logits'].detach(), dim=1)
        consistency = F.kl_div(torch.log_softmax(stressed['position_logits'], dim=1), target, reduction='batchmean')
        stress_loss = stressed_ce + consistency
    return (
        final_ce + float(LOSS_WEIGHTS['proposal_weight']) * proposal_ce
        + float(LOSS_WEIGHTS['identity_weight']) * identity
        + float(LOSS_WEIGHTS['anchor_preserve_weight']) * preservation
        + float(LOSS_WEIGHTS['stress_consistency_weight']) * stress_loss
        + float(LOSS_WEIGHTS['minimal_intervention_weight']) * intervention
        + float(LOSS_WEIGHTS['reconstruction_weight']) * result['reconstruction_loss']
    )
