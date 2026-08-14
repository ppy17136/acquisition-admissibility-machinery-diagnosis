# Frozen spectral-CNN confirmatory protocol

Status: frozen on 2026-08-11 before constructing the CNN input cache or fitting
any neural model. This is a cross-learner confirmation, not a model search.

## Input

For every audited record and each of `vibration_1`, `phase_current_1`, and
`phase_current_2`:

- interpolate to the frozen 64 kHz, 256,000-sample common grid;
- compute Welch PSD using the already frozen Hann 4,096-sample window and 50%
  overlap;
- store `log10(PSD + float32 epsilon)` at the 2,049 non-negative frequencies.

Modalities are vibration (one channel), current (two channels), and synchronous
fusion (three channels). Standardisation is fitted from training records only,
separately for every channel-frequency coordinate.

## Outer and validation splits

Run both predeclared five-fold outer designs:

- repetition-level outer folds;
- physical-bearing outer folds.

For outer fold `f`, validation fold is `(f mod 5) + 1` under the same split unit.
The outer test fold is never used for normalisation, optimisation, early stopping,
or model selection.

## Fixed architecture

One-dimensional spectral CNN:

1. Conv1d(input, 32, kernel 9, stride 2, padding 4), batch normalisation, ReLU;
2. max pool 2;
3. Conv1d(32, 64, kernel 7, stride 2, padding 3), batch normalisation, ReLU;
4. max pool 2;
5. Conv1d(64, 128, kernel 5, stride 2, padding 2), batch normalisation, ReLU;
6. adaptive average pool, dropout 0.2, linear three-class head.

## Fixed training

- seeds: 42, 1001, 1002;
- AdamW, learning rate 1e-3, weight decay 1e-4;
- cross-entropy loss;
- batch size 64;
- maximum 100 epochs;
- early stopping patience 12 on validation cross-entropy;
- deterministic CuDNN settings where supported.

All 90 fits (3 seeds x 3 modalities x 2 split designs x 5 folds) must be reported.
The confirmatory question is whether the split-unit gap persists across a
nonlinear learner; no superiority over the logistic baseline is claimed.
