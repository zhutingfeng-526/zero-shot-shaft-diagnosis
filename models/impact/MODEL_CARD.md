# Impact expert

This package exposes the final I expert (legacy numerical labels R0/R1/R2),
with an ordered 10-channel trailing buffer of 48,000 samples at 5,000 Hz.
The buffer is 9.6 s long. No future samples or target labels are inputs.
Supply the RPM for that buffer, plus optional condition/run identifiers and
the first sample's coordinate within the run. RPM is cast to integer exactly
as in the original detector. Channel order is recorded in `configs/impact.json`.

The model contains 34 numeric parameter arrays (33,040 bytes compressed),
including the source-Normal fallback normalizer, original covariance projector
prototypes, endpoint calibration, orbit relation classifier and background
operators, three source outer-fold acoustic prototype models, and detector
channel scales. It contains no raw signals, event feature banks, or prediction
caches. Array archive SHA-256 is checked when loading.

The baseline uses original-signal analytic covariances at 4, 12, 32 and 80 ms.
The detector uses phase-residual signals. Endpoint support permits an upgrade
to I2; orbit support permits I0 to I1. The acoustic residual acts only on an
I0 outcome with at least 12 eligible mature events. Its 12 ms margin threshold
is -0.5210351888090372. It exchanges the I0/I1 probabilities and preserves I2.

Events must be detector-closed and have complete 512-sample support on both
sides before use. The source's short/long spectral-validity filter also applies.
Normalization uses event-free samples from the current buffer; fewer than
1,024 valid samples triggers the frozen source-Normal fallback.

`ImpactExpert.predict` returns probabilities, readiness, class label, event
count and diagnostic metadata. Consumers must check `ready`. When no event
evidence is available, the original uniform probabilities and numerical argmax
0 are preserved, while the public `label` is `None`. Original mask failure
behavior is retained: if fallback robust mask computation raises, the causal
normalization helper uses an all-false exclusion mask. This release does not
silently redesign that behavior.

The parameters are assembled from the selected baseline, endpoint, relation
and acoustic archives. All 53 arrays/scalars in the model tree match the
original inference loader. Numeric NPZ loading does not execute pickled code.

Source model-building primitives are available as `fit_carrier_prototypes`
(group-balanced source projector means), `fit_orbit_classifier` (weighted
logistic fitting), and
`fit_background_operators` (source background
operators). These do not constitute a full training rerun: the original source
split construction, detector calibration, band selection, gain/cap calibration
and model certification are not recreated by a single public command. No
retraining was performed for this release.

Synthetic checks cover detector events, maturity, covariance features, H1/C2
scores and no-evidence handling. Four original-source AST reference cases
(zero, sparse impulses, dense impulses, noise) matched all 18 exported cell
arrays and final probabilities bit-for-bit on the audited CPU environment.
This is numerical extraction parity, not a new validation on measured data.
