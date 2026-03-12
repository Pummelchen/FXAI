# Getting Started

## What You Need
- MetaTrader 5 installed and working
- The live project in the MT5 Experts folder
- A clean compile before syncing anything to GitHub

## Source Of Truth
The MT5 Experts folder is the runnable source of truth. The GitHub repo is the synchronized, versioned copy.

Live project path:
`MQL5/Experts/FXAI`

## First Compile
Compile these first:
- `FXAI.mq5`
- `Tests/FXAI_AuditRunner.mq5`

This confirms the framework and the Audit Lab are both buildable before you start research work.

## First Backtest
1. Choose one plugin.
2. Keep warmup, context, and ensemble settings simple.
3. Use realistic spread and commission settings.
4. Validate behavior in Strategy Tester before scaling to cloud runs.

## First Audit
Run the Audit Lab when you want to verify plugin correctness, stability, or regression quality before a large test campaign.

Main tool entry point:
`FXAI/Tools/fxai_testlab.py`

Common commands:
- `compile-main`
- `compile-audit`
- `run-audit`
- `analyze`
- `baseline-save`
- `baseline-compare`
- `release-gate`

## Recommended Learning Path
1. Read [FXAI Framework](FXAI-Framework)
2. Read [Project Structure](Project-Structure)
3. Read [Audit Lab](Audit-Lab)
4. Start with one baseline plugin and one market symbol before scaling up
