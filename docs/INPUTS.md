# Input and output contract

## Waveforms

The channel order is:

| Indices | Signals |
| --- | --- |
| 0–5 | B1-x, B1-y, B2-x, B2-y, B3-x, B3-y vibration |
| 6–7 | Orthogonal shaft-orbit displacement channels |
| 8–9 | Acoustic A, acoustic B |

All channels are synchronized at 5,000 samples/s. Supply the same physical scaling as the recorded experiment. Per-window standardization or arbitrary amplitude normalization is not an input preprocessing step: U severity depends on amplitude. Branch-specific normalization is applied inside the experts.

The original pipeline filters complete records before window extraction. Filtering each cropped window separately changes boundary behavior. The original window store uses float16 quantization, converted to float32 for inference. Retain that quantization when reproducing the frozen outputs.

`preprocessing.preprocess_record` reproduces the selected preprocessing: take the final 300 s of a synchronized 10 kHz acquisition record, decimate to 5 kHz, then apply full-record modality filters. `preprocessing.window_at` selects a 6,000-sample window and preserves its float16 storage rounding. Do not filter already-preprocessed windows again. The record filters are zero-phase offline operations; causal inference histories do not make this preprocessing online-causal.

## NPZ fields for `scripts/predict.py`

| Expert | Required fields | Shape |
| --- | --- | --- |
| Complete system | `raw`, `rpm` | `(10,48000)`, one scalar |
| Structure | `raw` | `(N,10,6000)` |
| U | `raw`, `rpm`, `companion_i` | `(N,10,6000)`, `(N,)`, Boolean `(N,)` |
| M | `raw`, `rpm` | `(N,10,6000)`, `(N,)` |
| I | `raw`, `rpm` | `(10,48000)`, one scalar |

`rpm` is the supplied speed in r/min. The final experiment uses 1,200 r/min. M retains the original physical-phase validity range and nominal-speed fallback. A different supplied speed is not evidence of validation at that speed.

`companion_i` is obtained from the base window-level structure prediction. It selects the U readout path. The final structure is used later to determine which attributes are reported.

Complete inference uses `system.DiagnosisPredictor.predict`. It reproduces the original float16 waveform storage rounding, extracts 281 windows and evaluates the final endpoint. The default M configuration is the position rule used by the final integrated run, including its three frozen background predictors. Per-window structure, U and M probabilities are returned alongside the endpoint output for inspection.

I operates on a trailing 9.6 s buffer reconstructed from contiguous quantized DATA4 windows. Its `ready` flag must be checked: an empty event set returns unavailable output, not a valid I0 diagnosis. The CLI encodes unavailable position as `-1`.

## Minimal example

Run from the repository root after installation:

```console
python scripts/make_example.py
python scripts/predict.py --expert system --input data/example_history.npz --output data/example_prediction.npz --device cpu
```

Both scripts refuse to overwrite an existing output. To use a different input location, pass `--output PATH` to `make_example.py` and use that path with `predict.py`; the prediction output's parent directory must already exist.

The example contains deterministic sine waves, not experimental data. It exercises model loading and complete inference; its predicted labels do not measure diagnostic accuracy. The generated history is supplied directly at the preprocessed-input interface, without record preprocessing.

Inspect the saved result with `np.load("data/example_prediction.npz", allow_pickle=False)`. The complete system returns one endpoint and the 281 supporting windows.

## Complete-system output fields

| Fields | Shape | Meaning |
| --- | --- | --- |
| `structure_index`, `structure_label` | `(1,)` each | Final temporally integrated composition; see label order below. |
| `u_prediction` | `(1,)` | Reported U degree index. |
| `m_position_prediction`, `m_degree_prediction`, `m_joint_prediction` | `(1,)` each | Reported M attributes; joint index is `2 * position + degree`. |
| `i_prediction` | `(1,)` | Reported I position index. |
| `u_ready`, `m_ready`, `i_ready` | `(1,)` each | Branch availability. The complete-history interface sets U/M ready; I requires valid event evidence. |
| `u_probability`, `m_position_probability`, `m_degree_probability`, `i_probability` | `(1,3)`, `(1,3)`, `(1,2)`, `(1,3)` | Endpoint branch readouts before final composition gating. |
| `structure_refinement_applied` | `(1,)` | Whether physical evidence changed the temporal base decision. |
| `structure_mean_log_score`, `standardized_presence` | `(1,8)`, `(1,3)` | Temporal composition scores and signed U/M/I presence evidence; neither is a normalized probability. |
| `i_event_count`, `i_acoustic_check_applied` | `(1,)` each | Retained event count and acoustic-review application flag. |
| `window_structure_probability`, `window_structure_index` | `(281,8)`, `(281,)` | Base window composition votes and decisions, before temporal refinement. |
| `window_u_probability`, `window_m_position_probability`, `window_m_degree_probability` | `(281,3)`, `(281,3)`, `(281,2)` | Window-level branch readouts used in temporal integration. |
| `window_m_degree_quality` | `(281,)` | Synchronous degree acceptance flag, not a confidence score. |

Use the final `*_prediction` fields for reported attributes. A value of `-1` means the final composition excludes that component or its branch is unavailable; distinguish these cases using `structure_label` and `*_ready`. Branch probabilities may still be present for excluded components, and an unavailable I probability must not be interpreted as a valid diagnosis. The output does not contain a final composition probability vector: `window_structure_probability` is window evidence, while `structure_mean_log_score` is an unnormalized temporal score.

## Labels and temporal integration

- Structure indices: `0:N, 1:U, 2:M, 3:I, 4:UM, 5:UI, 6:MI, 7:UMI`.
- U degree indices: `0,1,2` correspond to U1,U2,U3.
- M position indices: `0,1,2` correspond to B1,B2,B3; degree indices `0,1` correspond to D0,D1. Joint index is `2 × position + degree`.
- I position indices: `0,1,2` correspond to I0,I1,I2. Historical `R` names inside numerical modules refer to I, not a fourth fault.

Consecutive 1.2 s windows advance by 150 samples (0.03 s). A complete 9.6 s context contains 281 windows. Histories must not cross records. U probabilities are averaged after path selection; M uses the frozen robust spatial/severity aggregation; I is evaluated on the entire trailing buffer. The final structure gates reported attributes. `-1` denotes an absent or unavailable attribute, distinguished by structure and readiness.
