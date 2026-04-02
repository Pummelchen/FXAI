# FXAI Offline Lab

`fxai_offline_lab.py` is the SQLite-backed offline control loop for FXAI.

It does not replace the MT5 model engine. MT5 and MQL5 still execute the real plugins. The offline lab adds:
- exact-window `M1 OHLC + spread` export from MT5
- SQLite storage for exported bars, tuning runs, scenario metrics, and promoted configs
- repeated model-zoo tuning on 3/6/12-month windows
- automatic promotion of best parameter packs per symbol and plugin
- champion/challenger governance, parameter lineage, and family scorecards
- distillation artifacts for lighter student targets and learned red-team plans for future hostile-market runs
- ready-to-use MT5 `.set` files so no parameter copy/paste is needed

Main commands from the repo root:

```bash
python3 FXAI/Tools/fxai_offline_lab.py init-db
python3 FXAI/Tools/fxai_offline_lab.py export-dataset --symbol-pack majors --months-list 3,6,12
python3 FXAI/Tools/fxai_offline_lab.py tune-zoo --profile continuous --auto-export --symbol-pack majors --months-list 3,6,12
python3 FXAI/Tools/fxai_offline_lab.py best-params --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py control-loop --profile continuous --symbol-pack majors --months-list 3,6,12 --cycles 0 --sleep-seconds 1800
```

Notes:
- `best-params` promotes all symbols under the selected profile by default; use `--symbol`, `--symbol-list`, or `--symbol-pack` only when you want to narrow the scope.
- Exact-window datasets store the effective exported first and last bar range, so later tuning and promotion stay aligned to the data that was actually ingested.
- SQLite access is opened with a bounded retry and busy timeout so overlapping admin and control-loop calls do not fail on transient lock contention.

Generated promotion artifacts land in:
- `FXAI/Tools/OfflineLab/Profiles/`
- `MQL5/Profiles/Tester/`
- `FILE_COMMON/FXAI/Offline/Promotions/`

Research and governance artifacts land in:
- `FXAI/Tools/OfflineLab/ResearchOS/`
- `FXAI/Tools/OfflineLab/Distillation/`
