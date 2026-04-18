from __future__ import annotations

import argparse
import sys

import fxai_testlab as testlab

from .cli_commands import *
from .common import DEFAULT_DB, OfflineLabError
from .mode import RUNTIME_MODES
from .promotion import SERIOUS_SCENARIOS

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="FXAI Turso-backed offline tuning and control lab")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    init_db = sub.add_parser("init-db", help="Initialize the Turso offline lab schema")
    init_db.set_defaults(func=cmd_init_db)

    val = sub.add_parser("validate-env", help="Validate Python, MT5, FILE_COMMON, and Offline Lab path assumptions")
    val.set_defaults(func=cmd_validate_env)

    doctor = sub.add_parser("doctor", help="Run a profile-aware FXAI toolchain and environment self-check")
    doctor.set_defaults(func=cmd_doctor)

    npv = sub.add_parser("newspulse-validate", help="Validate NewsPulse config, whitelist, and query scaffolding")
    npv.set_defaults(func=cmd_newspulse_validate)

    rev = sub.add_parser("rates-engine-validate", help="Validate the rates engine config, provider inputs, and proxy scaffolding")
    rev.set_defaults(func=cmd_rates_engine_validate)

    arv = sub.add_parser("adaptive-router-validate", help="Validate the adaptive router config and regime/plugin routing priors")
    arv.set_defaults(func=cmd_adaptive_router_validate)

    dev = sub.add_parser("dynamic-ensemble-validate", help="Validate the dynamic ensemble config and runtime weighting thresholds")
    dev.set_defaults(func=cmd_dynamic_ensemble_validate)

    der = sub.add_parser("dynamic-ensemble-replay-report", help="Rebuild a dynamic-ensemble replay report from append-only runtime history")
    der.add_argument("--symbol", default="")
    der.add_argument("--hours-back", type=int, default=72)
    der.set_defaults(func=cmd_dynamic_ensemble_replay_report)

    pcv = sub.add_parser("prob-calibration-validate", help="Validate the probabilistic calibration config, fallback memory, and runtime exports")
    pcv.set_defaults(func=cmd_prob_calibration_validate)

    pcr = sub.add_parser("prob-calibration-replay-report", help="Rebuild a probabilistic-calibration replay report from append-only runtime history")
    pcr.add_argument("--symbol", default="")
    pcr.add_argument("--hours-back", type=int, default=72)
    pcr.set_defaults(func=cmd_prob_calibration_replay_report)

    eqv = sub.add_parser("execution-quality-validate", help="Validate the execution-quality config, tier memory, and runtime forecast scaffolding")
    eqv.set_defaults(func=cmd_execution_quality_validate)

    eqr = sub.add_parser("execution-quality-replay-report", help="Rebuild an execution-quality replay report from append-only runtime history")
    eqr.add_argument("--symbol", default="")
    eqr.add_argument("--hours-back", type=int, default=72)
    eqr.set_defaults(func=cmd_execution_quality_replay_report)

    pnv = sub.add_parser("pair-network-validate", help="Validate the pair-network config, exported runtime policy, and factor mapping scaffolding")
    pnv.set_defaults(func=cmd_pair_network_validate)

    pnb = sub.add_parser("pair-network-build", help="Build the pair-network/factor-graph report and export the runtime portfolio-conflict policy")
    pnb.add_argument("--profile", default="continuous")
    pnb.set_defaults(func=cmd_pair_network_build)

    pnr = sub.add_parser("pair-network-report", help="Rebuild the pair-network report from the current offline-lab state")
    pnr.add_argument("--profile", default="continuous")
    pnr.set_defaults(func=cmd_pair_network_report)

    dgv = sub.add_parser("drift-governance-validate", help="Validate the drift-governance config, thresholds, and challenger policy")
    dgv.set_defaults(func=cmd_drift_governance_validate)

    dgr = sub.add_parser("drift-governance-run", help="Run one drift-governance cycle, update router artifacts, and refresh the dashboard")
    dgr.add_argument("--profile", default="continuous")
    dgr.add_argument("--group-key", default="")
    dgr.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    dgr.set_defaults(func=cmd_drift_governance_run)

    dgrep = sub.add_parser("drift-governance-report", help="Rebuild the drift-governance report from stored DB state")
    dgrep.add_argument("--profile", default="continuous")
    dgrep.set_defaults(func=cmd_drift_governance_report)

    lev = sub.add_parser("label-engine-validate", help="Validate the multi-horizon label-engine config, horizons, and artifact paths")
    lev.set_defaults(func=cmd_label_engine_validate)

    leb = sub.add_parser("label-engine-build", help="Build deterministic multi-horizon labels and meta-labels for existing exported datasets")
    leb.add_argument("--profile", default="continuous")
    leb.add_argument("--dataset-keys", default="")
    leb.add_argument("--group-key", default="")
    leb.add_argument("--symbol", default="EURUSD")
    leb.add_argument("--symbol-list", default="")
    leb.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    leb.add_argument("--months-list", default="3")
    leb.add_argument("--start-unix", type=int, default=0)
    leb.add_argument("--end-unix", type=int, default=0)
    leb.add_argument("--execution-profile", default="", choices=[""] + sorted(testlab.EXECUTION_PROFILES.keys()))
    leb.add_argument("--candidate-path", default="")
    leb.add_argument("--candidate-mode", default="")
    leb.add_argument("--limit-datasets", type=int, default=0)
    leb.set_defaults(func=cmd_label_engine_build)

    ler = sub.add_parser("label-engine-report", help="Rebuild the aggregated label-engine report from stored artifact metadata")
    ler.add_argument("--profile", default="")
    ler.set_defaults(func=cmd_label_engine_report)

    miv = sub.add_parser("microstructure-validate", help="Validate the microstructure proxy config, runtime contract, and service scaffolding")
    miv.set_defaults(func=cmd_microstructure_validate)

    mis = sub.add_parser("microstructure-install-service", help="Install the MT5 microstructure probe service into MQL5/Services")
    mis.add_argument("--skip-compile", action="store_true")
    mis.set_defaults(func=cmd_microstructure_install_service)

    mih = sub.add_parser("microstructure-health", help="Show current microstructure runtime health and the latest shared snapshot status")
    mih.set_defaults(func=cmd_microstructure_health)

    mir = sub.add_parser("microstructure-replay-report", help="Rebuild a microstructure replay report from append-only runtime history")
    mir.add_argument("--symbol", default="")
    mir.add_argument("--hours-back", type=int, default=24)
    mir.set_defaults(func=cmd_microstructure_replay_report)

    cav = sub.add_parser("cross-asset-validate", help="Validate the cross-asset config, runtime contracts, and probe scaffolding")
    cav.set_defaults(func=cmd_cross_asset_validate)

    cas = sub.add_parser("cross-asset-install-service", help="Install the MT5 cross-asset probe service into MQL5/Services")
    cas.add_argument("--skip-compile", action="store_true")
    cas.set_defaults(func=cmd_cross_asset_install_service)

    cao = sub.add_parser("cross-asset-once", help="Run one cross-asset engine cycle and refresh the shared snapshot")
    cao.set_defaults(func=cmd_cross_asset_once)

    cad = sub.add_parser("cross-asset-daemon", help="Continuously refresh the cross-asset shared snapshot")
    cad.add_argument("--interval-seconds", type=int, default=0)
    cad.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    cad.set_defaults(func=cmd_cross_asset_daemon)

    cah = sub.add_parser("cross-asset-health", help="Show current cross-asset engine and probe health, source state, and status artifacts")
    cah.set_defaults(func=cmd_cross_asset_health)

    car = sub.add_parser("cross-asset-replay-report", help="Rebuild a cross-asset replay report from append-only runtime history")
    car.add_argument("--symbol", default="")
    car.add_argument("--hours-back", type=int, default=72)
    car.set_defaults(func=cmd_cross_asset_replay_report)

    npi = sub.add_parser("newspulse-install-service", help="Install the MT5 NewsPulse calendar service into MQL5/Services")
    npi.add_argument("--skip-compile", action="store_true")
    npi.set_defaults(func=cmd_newspulse_install_service)

    mus = sub.add_parser("market-universe-show", help="Show the configured tradable FX universe and indicator-only symbol universe")
    mus.add_argument("--summary-only", action="store_true")
    mus.set_defaults(func=cmd_market_universe_show)

    mue = sub.add_parser("market-universe-export", help="Export the market-universe configuration from Turso metadata to JSON")
    mue.add_argument("--output-path", default="")
    mue.set_defaults(func=cmd_market_universe_export)

    mui = sub.add_parser("market-universe-import", help="Import a market-universe configuration JSON into Turso metadata")
    mui.add_argument("--input-path", required=True)
    mui.set_defaults(func=cmd_market_universe_import)

    mur = sub.add_parser("market-universe-reset-defaults", help="Reset the market-universe configuration back to the FXAI defaults")
    mur.set_defaults(func=cmd_market_universe_reset_defaults)

    npo = sub.add_parser("newspulse-once", help="Run one NewsPulse fusion cycle and refresh the shared snapshot")
    npo.set_defaults(func=cmd_newspulse_once)

    npd = sub.add_parser("newspulse-daemon", help="Continuously refresh NewsPulse from MT5 calendar exports and GDELT")
    npd.add_argument("--interval-seconds", type=int, default=0)
    npd.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    npd.set_defaults(func=cmd_newspulse_daemon)

    nph = sub.add_parser("newspulse-health", help="Show current NewsPulse daemon health, source state, and status artifacts")
    nph.set_defaults(func=cmd_newspulse_health)

    npr = sub.add_parser("newspulse-replay-report", help="Rebuild a NewsPulse replay report from append-only history")
    npr.add_argument("--pair", default="")
    npr.add_argument("--hours-back", type=int, default=48)
    npr.set_defaults(func=cmd_newspulse_replay_report)

    reo = sub.add_parser("rates-engine-once", help="Run one rates-engine cycle and refresh the shared rates snapshot")
    reo.set_defaults(func=cmd_rates_engine_once)

    red = sub.add_parser("rates-engine-daemon", help="Continuously refresh the rates-engine shared snapshot")
    red.add_argument("--interval-seconds", type=int, default=0)
    red.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    red.set_defaults(func=cmd_rates_engine_daemon)

    reh = sub.add_parser("rates-engine-health", help="Show current rates-engine daemon health, source state, and status artifacts")
    reh.set_defaults(func=cmd_rates_engine_health)

    rer = sub.add_parser("rates-engine-replay-report", help="Rebuild a rates-engine replay report from append-only history")
    rer.add_argument("--symbol", default="")
    rer.add_argument("--hours-back", type=int, default=72)
    rer.set_defaults(func=cmd_rates_engine_replay_report)

    boot = sub.add_parser("bootstrap", help="Create required lab folders, validate the environment, and initialize Turso")
    boot.add_argument("--report", default="")
    boot.add_argument("--no-init-db", action="store_true")
    boot.add_argument("--seed-demo", action="store_true")
    boot.set_defaults(func=cmd_bootstrap)

    comp = sub.add_parser("compile-export", help="Compile the MT5 offline export runner")
    comp.set_defaults(func=cmd_compile_export)

    exp = sub.add_parser("export-dataset", help="Export exact-window M1 OHLC+spread history from MT5 into Turso")
    exp.add_argument("--symbol", default="EURUSD")
    exp.add_argument("--symbol-list", default="")
    exp.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    exp.add_argument("--months-list", default="3,6,12")
    exp.add_argument("--start-unix", type=int, default=0)
    exp.add_argument("--end-unix", type=int, default=0)
    exp.add_argument("--max-bars", type=int, default=600000)
    exp.add_argument("--group-key", default="")
    exp.add_argument("--notes", default="")
    exp.add_argument("--replace", action="store_true")
    exp.add_argument("--skip-compile", action="store_true")
    exp.add_argument("--login")
    exp.add_argument("--server")
    exp.add_argument("--password")
    exp.add_argument("--timeout", type=int, default=300)
    exp.set_defaults(func=cmd_export_dataset)

    tune = sub.add_parser("tune-zoo", help="Run the full MT5 model-zoo tuning campaign on exact exported windows")
    tune.add_argument("--profile", default="continuous")
    tune.add_argument("--dataset-keys", default="")
    tune.add_argument("--group-key", default="")
    tune.add_argument("--auto-export", action="store_true")
    tune.add_argument("--symbol", default="EURUSD")
    tune.add_argument("--symbol-list", default="")
    tune.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    tune.add_argument("--months-list", default="3,6,12")
    tune.add_argument("--start-unix", type=int, default=0)
    tune.add_argument("--end-unix", type=int, default=0)
    tune.add_argument("--replace", action="store_true")
    tune.add_argument("--skip-compile", action="store_true")
    tune.add_argument("--top-plugins", type=int, default=0)
    tune.add_argument("--limit-experiments", type=int, default=0)
    tune.add_argument("--limit-runs", type=int, default=0)
    tune.add_argument("--scenario-list", default=SERIOUS_SCENARIOS)
    tune.add_argument("--bars", type=int, default=0)
    tune.add_argument("--horizon", type=int, default=5)
    tune.add_argument("--m1sync-bars", type=int, default=3)
    tune.add_argument("--normalization", type=int, default=0)
    tune.add_argument("--sequence-bars", type=int, default=0)
    tune.add_argument("--schema-id", type=int, default=0)
    tune.add_argument("--feature-mask", type=int, default=0)
    tune.add_argument("--commission-per-lot-side", type=float, default=None)
    tune.add_argument("--cost-buffer-points", type=float, default=None)
    tune.add_argument("--slippage-points", type=float, default=None)
    tune.add_argument("--fill-penalty-points", type=float, default=None)
    tune.add_argument("--wf-train-bars", type=int, default=256)
    tune.add_argument("--wf-test-bars", type=int, default=64)
    tune.add_argument("--wf-purge-bars", type=int, default=32)
    tune.add_argument("--wf-embargo-bars", type=int, default=24)
    tune.add_argument("--wf-folds", type=int, default=6)
    tune.add_argument("--seed", type=int, default=42)
    tune.add_argument("--strategy-profile", default="default")
    tune.add_argument("--broker-profile", default="")
    tune.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    tune.add_argument("--execution-profile", default="default", choices=sorted(testlab.EXECUTION_PROFILES.keys()))
    tune.add_argument("--login")
    tune.add_argument("--server")
    tune.add_argument("--password")
    tune.add_argument("--timeout", type=int, default=300)
    tune.set_defaults(func=cmd_tune_zoo)

    best = sub.add_parser("best-params", help="Promote the strongest parameter packs and emit MT5-ready presets")
    best.add_argument("--profile", default="continuous")
    best.add_argument("--dataset-keys", default="")
    best.add_argument("--group-key", default="")
    best.add_argument("--symbol", default="")
    best.add_argument("--symbol-list", default="")
    best.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    best.set_defaults(func=cmd_best_params)

    shadow = sub.add_parser("shadow-sync", help="Ingest live shadow-fleet ledgers from FILE_COMMON into Turso")
    shadow.add_argument("--profile", default="continuous")
    shadow.set_defaults(func=cmd_shadow_sync)

    tbranch = sub.add_parser("turso-branch-create", help="Create a Turso branch database and emit a branch env artifact")
    tbranch.add_argument("--profile", default="continuous")
    tbranch.add_argument("--source-database", default="")
    tbranch.add_argument("--target-database", default="")
    tbranch.add_argument("--timestamp", default="")
    tbranch.add_argument("--group-name", default="")
    tbranch.add_argument("--location-name", default="")
    tbranch.add_argument("--token-expiration", default="7d")
    tbranch.add_argument("--read-only-token", action="store_true")
    tbranch.set_defaults(func=cmd_turso_branch_create)

    tpitr = sub.add_parser("turso-pitr-restore", help="Create a point-in-time restore branch from Turso")
    tpitr.add_argument("--profile", default="continuous")
    tpitr.add_argument("--source-database", default="")
    tpitr.add_argument("--target-database", default="")
    tpitr.add_argument("--timestamp", required=True)
    tpitr.add_argument("--group-name", default="")
    tpitr.add_argument("--location-name", default="")
    tpitr.add_argument("--token-expiration", default="7d")
    tpitr.add_argument("--read-only-token", action="store_true")
    tpitr.set_defaults(func=cmd_turso_pitr_restore)

    tinv = sub.add_parser("turso-branch-inventory", help="List Turso branch inventory from the Platform API")
    tinv.add_argument("--profile", default="continuous")
    tinv.add_argument("--source-database", default="")
    tinv.set_defaults(func=cmd_turso_branch_inventory)

    tkill = sub.add_parser("turso-branch-destroy", help="Destroy a Turso branch database and mark it destroyed locally")
    tkill.add_argument("--target-database", required=True)
    tkill.set_defaults(func=cmd_turso_branch_destroy)

    taudit = sub.add_parser("turso-audit-sync", help="Ingest Turso organization audit logs into the Offline Lab")
    taudit.add_argument("--limit", type=int, default=50)
    taudit.add_argument("--pages", type=int, default=1)
    taudit.set_defaults(func=cmd_turso_audit_sync)

    tvec = sub.add_parser("turso-vector-reindex", help="Refresh Turso native vectors for analog-state retrieval")
    tvec.add_argument("--profile", default="continuous")
    tvec.add_argument("--symbol", default="")
    tvec.set_defaults(func=cmd_turso_vector_reindex)

    tnn = sub.add_parser("turso-vector-neighbors", help="Show nearest analog-state neighbors from Turso vectors")
    tnn.add_argument("--profile", default="continuous")
    tnn.add_argument("--symbol", required=True)
    tnn.add_argument("--limit", type=int, default=5)
    tnn.set_defaults(func=cmd_turso_vector_neighbors)

    demo = sub.add_parser("seed-demo", help="Seed a deterministic smoke profile and emit MT5 runtime artifacts without broker data")
    demo.add_argument("--profile", default="smoke")
    demo.add_argument("--symbol", default="EURUSD")
    demo.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    demo.set_defaults(func=cmd_seed_demo)

    deploy = sub.add_parser("deploy-profiles", help="Build live deployment profiles for MT5 runtime control plane")
    deploy.add_argument("--profile", default="continuous")
    deploy.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    deploy.set_defaults(func=cmd_deploy_profiles)

    adap = sub.add_parser("adaptive-router-profiles", help="Build adaptive regime-router profiles for live runtime orchestration")
    adap.add_argument("--profile", default="continuous")
    adap.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    adap.set_defaults(func=cmd_adaptive_router_profiles)

    arrep = sub.add_parser("adaptive-router-replay-report", help="Rebuild adaptive router replay summaries from runtime history")
    arrep.add_argument("--symbol", default="")
    arrep.add_argument("--hours-back", type=int, default=72)
    arrep.set_defaults(func=cmd_adaptive_router_replay_report)

    gov = sub.add_parser("autonomous-governance", help="Build portfolio supervisor and world-plan artifacts from live research telemetry")
    gov.add_argument("--profile", default="continuous")
    gov.add_argument("--group-key", default="")
    gov.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    gov.set_defaults(func=cmd_autonomous_governance)

    sup = sub.add_parser("supervisor-sync", help="Build central supervisor-service artifacts from live control-plane snapshots")
    sup.add_argument("--profile", default="continuous")
    sup.set_defaults(func=cmd_supervisor_sync)

    supd = sub.add_parser("supervisor-daemon", help="Continuously refresh supervisor-service artifacts from live control-plane snapshots")
    supd.add_argument("--profile", default="continuous")
    supd.add_argument("--interval-seconds", type=int, default=30)
    supd.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    supd.set_defaults(func=cmd_supervisor_daemon)

    attr = sub.add_parser("attribution-prune", help="Build attribution and live student-router pruning profiles")
    attr.add_argument("--profile", default="continuous")
    attr.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    attr.set_defaults(func=cmd_attribution_prune)

    dash = sub.add_parser("dashboard", help="Render the operator dashboard for a profile")
    dash.add_argument("--profile", default="continuous")
    dash.set_defaults(func=cmd_dashboard)

    live = sub.add_parser("live-state", help="Show the current file-backed runtime state for one symbol")
    live.add_argument("--profile", default="continuous")
    live.add_argument("--symbol", default="EURUSD")
    live.set_defaults(func=cmd_live_state)

    lineage = sub.add_parser("lineage-report", help="Render config lineage and live deployment provenance for a profile")
    lineage.add_argument("--profile", default="continuous")
    lineage.add_argument("--symbol", default="")
    lineage.set_defaults(func=cmd_lineage_report)

    bundle = sub.add_parser("minimal-bundle", help="Build a minimal live artifact bundle for promoted symbols")
    bundle.add_argument("--profile", default="continuous")
    bundle.add_argument("--output-dir", default="")
    bundle.set_defaults(func=cmd_minimal_bundle)

    det = sub.add_parser("verify-deterministic", help="Run seeded deterministic artifact checks against golden fixtures")
    det.add_argument("--refresh-golden", action="store_true")
    det.set_defaults(func=cmd_verify_deterministic)

    rec = sub.add_parser("recover-artifacts", help="Rebuild generated promotions, dashboards, and summaries from Turso state")
    rec.add_argument("--profile", default="continuous")
    rec.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    rec.set_defaults(func=cmd_recover_artifacts)

    loop = sub.add_parser("control-loop", help="Run the full export -> tune -> promote cycle continuously")
    loop.add_argument("--profile", default="continuous")
    loop.add_argument("--symbol", default="EURUSD")
    loop.add_argument("--symbol-list", default="")
    loop.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    loop.add_argument("--months-list", default="3,6,12")
    loop.add_argument("--start-unix", type=int, default=0)
    loop.add_argument("--end-unix", type=int, default=0)
    loop.add_argument("--replace", action="store_true")
    loop.add_argument("--skip-compile", action="store_true")
    loop.add_argument("--top-plugins", type=int, default=0)
    loop.add_argument("--limit-experiments", type=int, default=0)
    loop.add_argument("--limit-runs", type=int, default=0)
    loop.add_argument("--scenario-list", default=SERIOUS_SCENARIOS)
    loop.add_argument("--bars", type=int, default=0)
    loop.add_argument("--horizon", type=int, default=5)
    loop.add_argument("--m1sync-bars", type=int, default=3)
    loop.add_argument("--normalization", type=int, default=0)
    loop.add_argument("--sequence-bars", type=int, default=0)
    loop.add_argument("--schema-id", type=int, default=0)
    loop.add_argument("--feature-mask", type=int, default=0)
    loop.add_argument("--commission-per-lot-side", type=float, default=None)
    loop.add_argument("--cost-buffer-points", type=float, default=None)
    loop.add_argument("--slippage-points", type=float, default=None)
    loop.add_argument("--fill-penalty-points", type=float, default=None)
    loop.add_argument("--wf-train-bars", type=int, default=256)
    loop.add_argument("--wf-test-bars", type=int, default=64)
    loop.add_argument("--wf-purge-bars", type=int, default=32)
    loop.add_argument("--wf-embargo-bars", type=int, default=24)
    loop.add_argument("--wf-folds", type=int, default=6)
    loop.add_argument("--seed", type=int, default=42)
    loop.add_argument("--strategy-profile", default="default")
    loop.add_argument("--broker-profile", default="")
    loop.add_argument("--execution-profile", default="default", choices=sorted(testlab.EXECUTION_PROFILES.keys()))
    loop.add_argument("--cycles", type=int, default=1, help="0 means run forever")
    loop.add_argument("--sleep-seconds", type=int, default=0)
    loop.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    loop.add_argument("--login")
    loop.add_argument("--server")
    loop.add_argument("--password")
    loop.add_argument("--timeout", type=int, default=300)
    loop.set_defaults(func=cmd_control_loop)

    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    try:
        return int(args.func(args))
    except OfflineLabError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
