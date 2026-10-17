# Frozen model assets

All required inference assets are included. Models contain numerical arrays only; no optimizer states or experimental measurements are bundled.

| Component | Assets |
| --- | --- |
| Structure | Five network checkpoints, five corresponding source-prototype sets, shared decoder calibration |
| U | Five transport predictors and fixed ordinal calibration |
| M | Background, per-turn and synchronous predictors, matching background anchors, source normalizations and acceptance settings |
| I | Spatial projectors, local-relation models and acoustic calibration |
| Integration | Physical-presence models and selected temporal/routing settings |

The structure prototype files are `models/structure/prototypes_epoch_056.npz` through `prototypes_epoch_060.npz`. Their original archives match the final-run receipt hashes exactly. Tensor values were exported without refitting or substitution. The structure loader checks archive hashes, tensor schemas and null-vector agreement with each corresponding checkpoint.

Model manifests record individual file checksums. `release_manifest.json` inventories the complete checkout. Run `python scripts/check_release.py --tests` after copying the release directory.
