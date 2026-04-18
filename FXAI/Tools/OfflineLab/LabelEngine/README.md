# Label Engine

The Multi-Horizon Label Engine + Meta-Labeling subsystem builds reproducible training and evaluation targets on top of exported `dataset_bars`.

Phase 1 produces:

- direction labels
- move-magnitude labels
- MFE / MAE path labels
- time-to-favorable / time-to-adverse labels
- cost-aware tradeability labels
- meta-labels answering whether a raw candidate signal should have been traded at all

## Commands

Validate config:

```bash
python3 Tools/fxai_offline_lab.py label-engine-validate
```

Build artifacts for existing datasets:

```bash
python3 Tools/fxai_offline_lab.py label-engine-build \
  --profile continuous \
  --dataset-keys EURUSD_m1_3m_... \
  --execution-profile default
```

Build using deterministic baseline candidates:

```bash
python3 Tools/fxai_offline_lab.py label-engine-build \
  --profile continuous \
  --symbol EURUSD \
  --months-list 3 \
  --candidate-mode BASELINE_MOMENTUM
```

Build using external candidate signals:

```bash
python3 Tools/fxai_offline_lab.py label-engine-build \
  --profile continuous \
  --dataset-keys EURUSD_m1_3m_... \
  --candidate-path /absolute/path/to/candidates.ndjson \
  --candidate-mode EXTERNAL_FILE
```

Refresh the aggregated report:

```bash
python3 Tools/fxai_offline_lab.py label-engine-report --profile continuous
```

## Artifacts

- Config: `<FXAI_ROOT>/Tools/OfflineLab/LabelEngine/label_engine_config.json`
- Status: `<FXAI_ROOT>/Tools/OfflineLab/LabelEngine/label_engine_status.json`
- Report: `<FXAI_ROOT>/Tools/OfflineLab/LabelEngine/Reports/label_engine_report.json`
- Per-dataset artifacts: `<FXAI_ROOT>/Tools/OfflineLab/LabelEngine/Artifacts/...`

Each artifact bundle contains:

- `labels.ndjson`
- `meta_labels.ndjson`
- `summary.json`
- `bundle.json`
- `config_snapshot.json`

## Notes

- Label generation is deterministic for a fixed dataset and config snapshot.
- Path-aware logic is based on M1 bar geometry, so it is a bar-path approximation, not tick-perfect execution replay.
- If a symbol needs an exact point size, add it under `symbol_point_overrides` in the config.
