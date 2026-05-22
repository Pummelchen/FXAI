import XCTest
@testable import FXDataEngine

final class ControlPlaneTests: XCTestCase {
    func testSafeTokensAndProfilePathsMatchMQLShape() {
        XCTAssertEqual(ControlPlanePaths.safeToken("EUR/USD live"), "EUR_USD_live")
        XCTAssertEqual(ControlPlanePaths.safeToken(""), "default")
        XCTAssertEqual(
            ControlPlanePaths.liveDeploymentProfileFile(symbol: "EUR/USD live"),
            "FXAI/Offline/Promotions/fxai_live_deploy_EUR_USD_live.tsv"
        )
        XCTAssertEqual(
            ControlPlanePaths.snapshotFile(symbol: "EURUSD", login: 42, magic: 99, chartID: 2_147_483_648),
            "FXAI/ControlPlane/cp_42_99_EURUSD_1.tsv"
        )
    }

    func testKeyValueAndCSVUtilitiesFollowMQLParsingRules() {
        let doc = ControlPlaneKeyValueDocument(tsv: """
        teacher_weight\t2.5
        ignored
        teacher_weight\tbad
        tabbed\tvalue\textra
        enabled\t1
        """)

        XCTAssertEqual(doc.double("teacher_weight", default: 0.58), 0.0)
        XCTAssertEqual(doc.double("missing", default: 0.58), 0.58)
        XCTAssertEqual(doc.string("tabbed"), "value")
        XCTAssertTrue(doc.bool("enabled"))
        XCTAssertTrue(ControlPlaneCSV.containsToken("FX7, FXStupid, Mythos", token: "FXStupid"))
        XCTAssertFalse(ControlPlaneCSV.containsToken("FX7, FXStupid, Mythos", token: "Unknown"))
        XCTAssertEqual(ControlPlaneCSV.mapWeight("FX7=1.4, Mythos = 0.5", key: "Mythos"), 0.5)
        XCTAssertEqual(ControlPlaneCSV.mapWeight("FX7=1.4", key: "Missing", default: 0.8), 0.8)
    }

    func testLiveDeploymentProfileDefaultsAndClampsMatchLegacyLoader() {
        let profile = LiveDeploymentProfile.parse(symbol: "EURUSD", tsv: """
        profile_name\tstress
        teacher_weight\t2.5
        student_weight\t0.0
        policy_trade_floor\t0.1
        policy_size_bias\t9.0
        soft_timeout_bars\t10
        hard_timeout_bars\t3
        runtime_mode\tinvalid
        telemetry_level\tverbose
        snapshot_detail\tdeep
        performance_budget_ms\t500
        max_runtime_models\t999
        promotion_tier\tbad
        shadow_enabled\t0
        """, loadedAtUTC: 1_704_067_200)

        XCTAssertTrue(profile.ready)
        XCTAssertEqual(profile.symbol, "EURUSD")
        XCTAssertEqual(profile.teacherWeight, 0.95)
        XCTAssertEqual(profile.studentWeight, 0.05)
        XCTAssertEqual(profile.policyTradeFloor, 0.20)
        XCTAssertEqual(profile.policySizeBias, 1.60)
        XCTAssertEqual(profile.softTimeoutBars, 10)
        XCTAssertEqual(profile.hardTimeoutBars, 11)
        XCTAssertEqual(profile.runtimeMode, "research")
        XCTAssertEqual(profile.telemetryLevel, "full")
        XCTAssertEqual(profile.snapshotDetail, "full")
        XCTAssertEqual(profile.performanceBudgetMS, 100.0)
        XCTAssertEqual(profile.maxRuntimeModels, FXDataEngineConstants.aiCount)
        XCTAssertEqual(profile.promotionTier, "experimental")
        XCTAssertFalse(profile.shadowEnabled)
        XCTAssertEqual(profile.loadedAtUTC, 1_704_067_200)
    }

    func testStudentRouterProfileWeightsAndChampionFiltering() {
        let profile = StudentRouterProfile.parse(symbol: "EURUSD", tsv: """
        champion_only\t1
        max_active_models\t999
        min_meta_weight\t0.8
        allow_plugins_csv\tChampion, Backup
        plugin_weights_csv\tChampion=1.7, Backup=0.4, Blocked=0.0
        family_weight_linear\t1.2
        family_weight_tree\t0.01
        """)

        XCTAssertTrue(profile.ready)
        XCTAssertEqual(profile.maxActiveModels, FXDataEngineConstants.aiCount)
        XCTAssertEqual(profile.minMetaWeight, 0.25)
        XCTAssertEqual(profile.familyWeight(.linear), 1.2)
        XCTAssertEqual(profile.familyWeight(.tree), 0.05)
        XCTAssertEqual(profile.pluginWeight(pluginName: "Champion", family: .linear), 1.6)
        XCTAssertTrue(profile.allowsPlugin(pluginName: "Champion", family: .linear))
        XCTAssertFalse(profile.allowsPlugin(pluginName: "Other", family: .linear))
        XCTAssertFalse(profile.allowsPlugin(pluginName: "Champion", family: .tree))
        XCTAssertFalse(profile.allowsPlugin(pluginName: "Blocked", family: .linear))
    }

    func testAdaptiveRouterProfileParsesWeightsAndClampsThresholds() {
        let profile = AdaptiveRouterProfile.parse(symbol: "EURUSD", tsv: """
        enabled\t1
        router_mode\tunsupported
        caution_threshold\t0.07
        abstain_threshold\t0.80
        block_threshold\t0.50
        min_plugin_weight\t0.20
        max_plugin_weight\t2.00
        max_active_weight_share\t3.0
        plugin_global_weights_csv\tFX7=9.0, FXStupid=0.01
        plugin_news_compatibility_csv\tFX7=3.0
        plugin_liquidity_robustness_csv\tFX7=0.01
        plugin_regime_TREND_PERSISTENT_csv\tFX7=1.8
        plugin_session_LONDON_csv\tFX7=2.8
        """)

        XCTAssertTrue(profile.ready)
        XCTAssertTrue(profile.enabled)
        XCTAssertEqual(profile.routerMode, "WEIGHTED_ENSEMBLE")
        XCTAssertEqual(profile.cautionThreshold, 0.10)
        XCTAssertEqual(profile.abstainThreshold, 0.10)
        XCTAssertEqual(profile.blockThreshold, 0.10)
        XCTAssertEqual(profile.maxActiveWeightShare, 0.99)
        XCTAssertEqual(profile.globalWeight(pluginName: "FX7"), 2.0)
        XCTAssertEqual(profile.globalWeight(pluginName: "FXStupid"), 0.20)
        XCTAssertEqual(profile.newsCompatibility(pluginName: "FX7"), 2.50)
        XCTAssertEqual(profile.liquidityRobustness(pluginName: "FX7"), 0.05)
        XCTAssertEqual(profile.regimeWeight(pluginName: "FX7", regimeLabel: "TREND_PERSISTENT"), 1.8)
        XCTAssertEqual(profile.sessionWeight(pluginName: "FX7", sessionLabel: "LONDON"), 2.50)
        XCTAssertEqual(profile.sessionWeight(pluginName: "FX7", sessionLabel: "UNKNOWN"), 1.0)
    }

    func testControlPlaneSnapshotClampsAndValidatesLikeLegacyReader() {
        let snapshot = ControlPlaneSnapshot.parse(tsv: """
        login\t42
        magic\t99
        chart_id\t7
        symbol\tEURUSD
        bar_time\t1704067200
        direction\t1
        signal_intensity\t8
        confidence\t2
        reliability\t-1
        trade_gate\t0.7
        hierarchy_score\t0.8
        macro_quality\t0.9
        trade_edge_norm\t-4
        expected_move_norm\t8
        policy_trade_prob\t2
        policy_no_trade_prob\t-1
        policy_lifecycle_action\t999
        policy_size_mult\t5
        gross_exposure_lots\t2000
        correlated_exposure_lots\t2000
        directional_cluster_lots\t2000
        capital_risk_pct\t200
        portfolio_pressure\t5
        """)

        XCTAssertTrue(snapshot.valid)
        XCTAssertEqual(snapshot.signalIntensity, 4.0)
        XCTAssertEqual(snapshot.confidence, 1.0)
        XCTAssertEqual(snapshot.reliability, 0.0)
        XCTAssertEqual(snapshot.tradeEdgeNorm, -1.0)
        XCTAssertEqual(snapshot.expectedMoveNorm, 4.0)
        XCTAssertEqual(snapshot.policyTradeProb, 1.0)
        XCTAssertEqual(snapshot.policyNoTradeProb, 0.0)
        XCTAssertEqual(snapshot.policyLifecycleAction, .timeout)
        XCTAssertEqual(snapshot.policySizeMultiplier, 2.0)
        XCTAssertEqual(snapshot.grossExposureLots, 1_000.0)
        XCTAssertEqual(snapshot.capitalRiskPct, 100.0)
        XCTAssertEqual(snapshot.portfolioPressure, 2.0)

        let invalid = ControlPlaneSnapshot.parse(tsv: "login\t42\nsymbol\tEURUSD\n")
        XCTAssertFalse(invalid.valid)
    }
}
