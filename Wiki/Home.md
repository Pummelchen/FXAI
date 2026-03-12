# FXAI Wiki

FXAI is a native MetaTrader 5 framework for building, testing, and operating cost-aware FX prediction models inside MT5 and MQL5. The framework combines one execution engine, one data pipeline, one plugin API, and a dedicated Audit Lab so research, validation, and deployment stay inside the same environment.

## Start Here
- [Getting Started](Getting-Started)
- [FXAI Framework](FXAI-Framework)
- [Audit Lab](Audit-Lab)
- [Project Structure](Project-Structure)

## What FXAI Gives You
- One framework to compare many model families under the same rules
- Cost-aware `BUY / SELL / SKIP` prediction flow
- Shared feature generation, normalization, warmup, calibration, and risk controls
- MT5-native backtesting and deployment without external DLLs or inference services
- A drill-sergeant Audit Lab for plugin certification, regression checks, and release gating

## Typical Workflow
1. Compile and work in the live MT5 Experts folder.
2. Run targeted backtests or audits for the plugin or feature change you are making.
3. Use the Audit Lab to catch weak behavior before expensive cloud optimization.
4. Sync the clean MT5 project into Git only after the live code compiles cleanly.

## Who This Is For
- MT5/MQL5 researchers who want a serious plugin-based FX framework
- Strategy developers who want repeatable testing and release discipline
- Engineers who need one codebase for research, backtesting, auditing, and live execution
