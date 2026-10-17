# Frozen-result regression checks

The three modes below answer different questions. None fits a model, changes calibration or sets backend precision flags. Experimental inputs and predictions are not included in this checkout; access may be requested from the corresponding author, subject to the confidentiality restrictions described in [README](../README.md#data-and-reuse).

| Mode | Input | What a pass establishes |
| --- | --- | --- |
| `synthetic` | Bundled artificial numerical inputs | Original-function reference agreement for decoding, voting, temporal aggregation, structure switching and scoring |
| `frozen-endpoints` | Authorized original endpoint predictions, truth and grouping keys | Frozen decision identity and independent re-scoring against the historical result |
| `current-endpoints` | Caller-produced endpoint predictions, with the same authorized truth and keys | Exact endpoint-decision agreement and re-scoring against that reference |

The synthetic expectations were generated with isolated original frozen functions, not copied from a new run of the release implementation. They contain no experimental waveforms or private record keys. Endpoint references contain aggregate metrics and array hashes only. `current-endpoints` checks supplied outputs; it does not run neural inference or compare floating-point probabilities.

## Commands

Run from the release directory:

```sh
python scripts/check_frozen_results.py --mode synthetic
python scripts/check_frozen_results.py --mode frozen-endpoints --split final_pair --candidate E0_C3_T7 --bundle /path/to/authorized_endpoints.npz
python scripts/check_frozen_results.py --mode current-endpoints --split final_pair --candidate E0_C3_T7 --bundle /path/to/new_endpoints.npz
```

References cover the baseline (`B0_C0_T0`) and complete integration (`E0_C3_T7`), for `final_pair` (120,060 endpoints) and `final_triple` (24,012 endpoints). Each group binds all input/result arrays and retains 17 unrounded, condition/run-balanced metrics. Accuracies are proportions, not percentages.

Exit code 0 means the requested check passed. Missing authorized data produces `status: skipped`, `passed: false` and exit code 2. A mismatch or invalid bundle also exits with 2. Passing synthetic tests does not establish experimental reproducibility.

## Authorized endpoint bundle

Use a numeric NPZ with exactly these fields, aligned to the authorized frozen endpoint order:

| Fields | Shape |
| --- | --- |
| `truth_structure_index`, `truth_u_degree`, `truth_m_position`, `truth_m_degree`, `truth_i_position` | Each `(N,)` |
| `prediction_structure_index`, `prediction_u_prediction`, `prediction_m_position_prediction`, `prediction_m_degree_prediction`, `prediction_m_joint_prediction`, `prediction_i_prediction` | Each `(N,)` |
| `prediction_u_ready`, `prediction_m_ready`, `prediction_i_ready` | Each `(N,)` |
| `condition`, `run` | Each `(N,)` |
| `endpoint_key` | `(N,2)` ordered, unique record/window keys supplied with authorized data |

All arrays must be integer or Boolean; pickle/object arrays are rejected. Structures use `N/U/M/I/UM/UI/MI/UMI` indices `0` through `7`. Attribute class indices are zero-based as defined in [INPUTS.md](INPUTS.md). Absent truth attributes use `-1`. Predictions use `-1` for absent/unavailable attributes, and readiness is `0` or `1`. The M joint index is `2 * position + degree` when both are valid; otherwise it is `-1`.

The checker verifies complete array identities using canonical little-endian int64 values, including shapes. Reordering endpoints or changing predictions fails even if aggregate accuracy is unchanged. It independently computes metrics through `evaluation.evaluate_complete`; comparison uses an absolute tolerance of `1e-12` for score summation, with exact class/readiness/key identities. Model probabilities are not covered by that score tolerance.

For new waveform inference, retain its [runtime report](ENVIRONMENT.md), model/configuration identity, endpoint ordering and grouping labels. Assemble the endpoint bundle from the actual outputs without replacing changed decisions with historical values. A failed comparison is a diagnostic finding, not a reason to retune thresholds or replace the reference. The release does not claim that different CPU/GPU environments produce identical floating-point scores or endpoint decisions.
