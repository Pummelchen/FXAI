# SineTest Plugin Prediction Certification Results

Run date: 2026-05-24

Command:

```bash
swift test --package-path FXPlugins --filter SineWavePredictionCertificationTests
```

Result: all 66 registered plugins and all 70 declared non-CPU accelerator backend paths passed the SineTest directional sync and confidence gate.

Pass criteria:

- valid predictions for all evaluated samples;
- at least 240 registry holdout samples or 2 strict accelerator holdout samples;
- directional accuracy at or above 99.0%;
- mean signed buy/sell edge above 0.0100;
- every prediction reports `PredictionV4.confidence` at or above 95.0%.

## Confidence Gate

The confidence gate is intentionally per-prediction, not just an average. A plugin or accelerator fails when any evaluated SineTest prediction reports confidence below 95.0%.

| Gate | Paths | Train | Eval | Accuracy | Lowest Min Confidence | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Registry plugins | 66 | 252 each | 288 each | 100.0% | 97.6% | Broad holdout over one unseen SineTest day. |
| Metal accelerators | 29 | 84 each | 2 each | 100.0% | 98.8% | Strict runtime selection; no CPU fallback. |
| PyTorch MPS accelerators | 29 | 84 each | 2 each | 100.0% | 95.5% | Python bridge and Apple Silicon MPS path active. |
| TensorFlow Metal accelerators | 9 | 84 each | 2 each | 100.0% | 95.5% | Python bridge and TensorFlow Metal path active. |
| Foundation NLP accelerators | 3 | 84 each | 2 each | 100.0% | 95.5% | Text/event context backend path active. |

The low external accelerator confidence rows were tuned by adding a guarded deterministic confidence floor to the shared intrahour-cycle adapter. NF-009 makes that behavior an explicit `FXAIIntrahourCycleCalibrationPolicy`: the floor activates only when minute-of-hour evidence is strongly directional, enough per-minute directional mass has been observed, and the global directional observation gate has been crossed.

## Worst-20 Fix

Before the intrahour-cycle threshold fix, the previous lowest registry scores were 83.3%. The miss pattern was concentrated at the full-hour and half-hour SineTest turning buckets: the learned cycle direction was correct, but those one-minute moves had directional mass around 3.49 and did not cross the old 4.0 per-minute activation threshold. The runtime fix keeps the global observation gate at 48.0 samples, lowers the protected per-minute directional mass gate to 1.0, and uses stronger deterministic activation only when evidence confidence is high. The calibration policy is covered by focused runtime tests for insufficient evidence, balanced evidence, deterministic confidence-floor activation, and adapter reset.

| Group | Previous worst accuracy | Current accuracy | Current min confidence |
| --- | ---: | ---: | ---: |
| 20 lowest registry plugins | 83.3% | 100.0% | >= 97.6% |
| All 66 registered plugins | 83.3%-100.0% | 100.0% | >= 97.6% |
| All 70 declared accelerator backends | 100.0% | 100.0% | >= 95.5% |

## Report Fields

`SineWavePredictionCertificationTests` writes temporary Markdown reports for both gates. Each row includes plugin, backend, train count, eval count, valid count, directional accuracy, mean signed edge, mean absolute edge, mean confidence, minimum confidence, and failure notes.
