# Drift Retraining

The Drift Retraining subsystem turns significant drift-governance evidence into
auditable, research-only retraining requests.

It does not detect drift itself. `drift-governance-run` remains the detector of
record and writes the governance state that this queue consumes.

The root [FXAI Governance](../../../../GOVERNANCE.md) contract remains
authoritative for promotion, release evidence, demo execution, and live
execution gates.

## Commands

Run drift governance first:

```bash
python3 FXDataEngine/Tools/fxai_offline_lab.py drift-governance-run --profile continuous
```

Queue retraining requests from degraded governance state:

```bash
python3 FXDataEngine/Tools/fxai_offline_lab.py drift-retraining-queue \
  --profile continuous \
  --data-days 90 \
  --wf-train-years 1,2,3,5
```

Inspect queued requests:

```bash
python3 FXDataEngine/Tools/fxai_offline_lab.py drift-retraining-report --profile continuous
```

Preview the research campaign without running it:

```bash
python3 FXDataEngine/Tools/fxai_offline_lab.py drift-retraining-execute \
  --profile continuous \
  --dry-run
```

Run one queued research campaign explicitly:

```bash
python3 FXDataEngine/Tools/fxai_offline_lab.py drift-retraining-execute \
  --profile continuous \
  --limit 1
```

## Safety Contract

- Drift retraining requests are idempotent for the same profile, symbol, plugin,
  policy version, action, state, reason codes, and drift windows.
- Low-support drift is skipped unless `--include-low-support` is set.
- The execution command runs research export/audit/tune work only.
- The execution command does not call `best-params`.
- Auto-promotion is false in every queue/report/execute artifact.
- Any challenger or champion change still requires walk-forward, calibration,
  financial-utility, lineage, promotion-review, risk, and kill-switch gates.

## Artifacts

- Status: `FXDataEngine/Tools/OfflineLab/DriftRetraining/drift_retraining_status.json`
- Report: `FXDataEngine/Tools/OfflineLab/DriftRetraining/Reports/drift_retraining_report.json`
- History: `FXDataEngine/Tools/OfflineLab/DriftRetraining/drift_retraining_history.ndjson`
- Alerts: `FXDataEngine/Tools/OfflineLab/DriftRetraining/drift_retraining_alerts.jsonl`

## Alerts

Every newly queued request writes a local JSONL alert. To mirror alerts into an
existing operational alert file, set:

```bash
export FXAI_ALERT_JSONL_PATH=/absolute/path/to/alerts.jsonl
```

Webhook delivery is disabled by default. Enable it explicitly:

```bash
export FXAI_DRIFT_RETRAINING_ENABLE_WEBHOOKS=1
```

For a generic or Discord-compatible webhook:

```bash
export FXAI_DRIFT_RETRAINING_WEBHOOK_URL=https://example.invalid/webhook
```

For Telegram:

```bash
export FXAI_DRIFT_RETRAINING_TELEGRAM_BOT_TOKEN=...
export FXAI_DRIFT_RETRAINING_TELEGRAM_CHAT_ID=...
```

Webhook delivery failures are recorded in the queue result but do not stop local
queue persistence.
