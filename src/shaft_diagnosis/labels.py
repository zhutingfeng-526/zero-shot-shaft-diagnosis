"""Public U/M/I labels; historical R is retained solely in checkpoint keys."""
PUBLIC_COMPONENTS = ("U", "M", "I")
PUBLIC_STRUCTURES = ("normal", "U", "M", "I", "UM", "UI", "MI", "UMI")
LEGACY_COMPONENTS = ("U", "M", "R")


def public_label(legacy):
    return str(legacy).replace("R", "I")


def legacy_label(public):
    return str(public).replace("I", "R")
