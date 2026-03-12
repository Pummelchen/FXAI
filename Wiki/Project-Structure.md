# Project Structure

## Root
- `FXAI.mq5` main EA entry point
- `FXAI.ex5` compiled EA

## API
- `API/api.mqh` registry and API helpers
- `API/plugin_base.mqh` plugin base contract and shared services

## Engine
- `Engine/core.mqh` shared types, enums, manifest helpers, schema helpers, common math
- `Engine/data_pipeline.mqh` feature generation, normalization, and aligned context loading
- `Engine/*.mqh` runtime, training, warmup, lifecycle, sample, and meta orchestration

## Plugins
- `Plugins/*.mqh` individual model plugins

## Tests
- `Tests/FXAI_AuditRunner.mq5` MT5-side audit runner
- `Tests/audit_core.mqh` synthetic and market replay audit logic

## Tools
- `Tools/fxai_testlab.py` external drill-sergeant tool
- `Tools/plugin_oracles.json` plugin grading rules
- `Tools/Baselines/` saved regression baselines

## Operating Rule
The live MT5 Experts folder is the runtime source of truth. GitHub receives synchronized source after a clean compile.
