# Dynamic Ensemble

Dynamic Ensemble is the phase-1 post-inference meta-router for the FXAI plugin zoo.

It does not replace Adaptive Router. Adaptive Router stays the regime-aware prior layer. Dynamic Ensemble consumes live plugin outputs plus current context, then:

- reduces trust in miscalibrated or context-bad plugins
- suppresses clearly weak participants
- reweights the surviving plugins
- raises `SKIP` / abstain bias when ensemble quality is weak

Run:

- `python3 Tools/fxai_offline_lab.py dynamic-ensemble-validate`
- `python3 Tools/fxai_offline_lab.py dynamic-ensemble-replay-report --symbol EURUSD --hours-back 72`

Runtime artifacts:

- `FILE_COMMON/FXAI/Runtime/dynamic_ensemble_config.tsv`
- `FILE_COMMON/FXAI/Runtime/fxai_dynamic_ensemble_<SYMBOL>.tsv`
- `FILE_COMMON/FXAI/Runtime/fxai_dynamic_ensemble_history_<SYMBOL>.ndjson`
