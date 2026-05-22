import XCTest
@testable import FXDataEngine

final class ControlPlaneTests: XCTestCase {
    private func temporaryRepository() throws -> (URL, ControlPlaneFileRepository) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("FXDataEngineTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (root, ControlPlaneFileRepository(rootURL: root))
    }

    private func write(_ text: String, relativePath: String, root: URL) throws {
        let url = root.appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

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

    func testPortfolioSupervisorProfileClampsLegacyDefaults() {
        let profile = PortfolioSupervisorProfile.parse(tsv: """
        profile_name\tportfolio
        gross_budget_bias\t0.1
        correlated_budget_bias\t9.0
        directional_budget_bias\t0.1
        capital_risk_cap_pct\t20
        macro_overlap_cap\t0
        concentration_cap\t9
        supervisor_weight\t2
        hard_block_score\t0.1
        policy_enter_floor\t2
        policy_no_trade_ceiling\t0
        """, loadedAtUTC: 1_704_067_200)

        XCTAssertTrue(profile.ready)
        XCTAssertEqual(profile.profileName, "portfolio")
        XCTAssertEqual(profile.grossBudgetBias, 0.40)
        XCTAssertEqual(profile.correlatedBudgetBias, 1.60)
        XCTAssertEqual(profile.directionalBudgetBias, 0.40)
        XCTAssertEqual(profile.capitalRiskCapPct, 10.0)
        XCTAssertEqual(profile.macroOverlapCap, 0.10)
        XCTAssertEqual(profile.concentrationCap, 2.0)
        XCTAssertEqual(profile.supervisorWeight, 1.0)
        XCTAssertEqual(profile.hardBlockScore, 0.20)
        XCTAssertEqual(profile.policyEnterFloor, 0.95)
        XCTAssertEqual(profile.policyNoTradeCeiling, 0.10)
        XCTAssertEqual(profile.loadedAtUTC, 1_704_067_200)
    }

    func testSupervisorServiceStateClampsBudgetFallbackAndFreshness() {
        let state = SupervisorServiceState.parse(tsv: """
        symbol\t__GLOBAL__
        generated_at\t1000
        expires_at\t0
        snapshot_count\t20000
        gross_pressure\t3
        directional_long_pressure\t3
        directional_short_pressure\t3
        macro_pressure\t3
        concentration_pressure\t3
        freshness_penalty\t3
        pressure_velocity\t-3
        gross_velocity\t3
        budget_multiplier\t0.5
        add_multiplier\t2
        reduce_bias\t2
        exit_bias\t2
        entry_floor\t2
        block_score\t0.1
        supervisor_score\t4
        """, nowUTC: 1_200)

        XCTAssertTrue(state.ready)
        XCTAssertEqual(state.resolvedSymbol("EURUSD").symbol, "EURUSD")
        XCTAssertEqual(state.snapshotCount, 10_000)
        XCTAssertEqual(state.grossPressure, 2.0)
        XCTAssertEqual(state.macroPressure, 1.5)
        XCTAssertEqual(state.concentrationPressure, 1.0)
        XCTAssertEqual(state.freshnessPenalty, 1.0)
        XCTAssertEqual(state.pressureVelocity, -1.0)
        XCTAssertEqual(state.grossVelocity, 1.0)
        XCTAssertEqual(state.longEntryBudgetMultiplier, 0.5)
        XCTAssertEqual(state.shortEntryBudgetMultiplier, 0.5)
        XCTAssertEqual(state.addMultiplier, 1.40)
        XCTAssertEqual(state.reduceBias, 1.0)
        XCTAssertEqual(state.exitBias, 1.0)
        XCTAssertEqual(state.entryFloor, 0.95)
        XCTAssertEqual(state.blockScore, 0.20)
        XCTAssertEqual(state.supervisorScore, 3.0)

        let stale = SupervisorServiceState.parse(tsv: "symbol\tEURUSD\ngenerated_at\t1000\n", nowUTC: 1_300)
        XCTAssertFalse(stale.ready)
    }

    func testSupervisorCommandStateDirectionHelpersAndFreshness() {
        let state = SupervisorCommandState.parse(symbol: "EURUSD", tsv: """
        symbol\t__GLOBAL__
        generated_at\t1000
        entry_budget_mult\t0.5
        hold_budget_mult\t9
        add_cap_mult\t0
        reduce_bias\t2
        exit_bias\t2
        tighten_bias\t2
        timeout_bias\t2
        long_block\t1
        short_block\t0
        block_score\t9
        max_active_models\t999
        champion_only\t1
        """, nowUTC: 1_200)

        XCTAssertTrue(state.ready)
        XCTAssertEqual(state.resolvedSymbol("EURUSD").symbol, "EURUSD")
        XCTAssertEqual(state.longEntryBudgetMultiplier, 0.5)
        XCTAssertEqual(state.shortEntryBudgetMultiplier, 0.5)
        XCTAssertEqual(state.entryBudgetMultiplier(for: 1), 0.5)
        XCTAssertEqual(state.entryBudgetMultiplier(for: 0), 0.5)
        XCTAssertEqual(state.holdBudgetMultiplier, 1.20)
        XCTAssertEqual(state.addCapMultiplier, 0.05)
        XCTAssertEqual(state.reduceBias, 1.0)
        XCTAssertEqual(state.exitBias, 1.0)
        XCTAssertEqual(state.tightenBias, 1.0)
        XCTAssertEqual(state.timeoutBias, 1.0)
        XCTAssertEqual(state.blockScore, 3.0)
        XCTAssertEqual(state.maxActiveModels, FXDataEngineConstants.aiCount)
        XCTAssertTrue(state.championOnly)
        XCTAssertTrue(state.blocksDirection(1))
        XCTAssertFalse(state.blocksDirection(0))
        XCTAssertFalse(state.blocksDirection(-1))

        let expired = SupervisorCommandState.parse(
            symbol: "EURUSD",
            tsv: "generated_at\t1000\nexpires_at\t1100\n",
            nowUTC: 1_101
        )
        XCTAssertFalse(expired.ready)
    }

    func testControlPlaneScoringAggregatesPeerSnapshots() {
        let identity = ControlPlanePeerIdentity(login: 42, magic: 99, chartID: 7)
        let peerA = ControlPlaneSnapshot.parse(tsv: """
        login\t42
        magic\t99
        chart_id\t8
        symbol\tUSDJPY
        bar_time\t10000
        direction\t1
        signal_intensity\t1.2
        confidence\t0.8
        reliability\t0.6
        macro_quality\t0.5
        policy_trade_prob\t0.7
        policy_no_trade_prob\t0.2
        policy_capital_efficiency\t0.9
        policy_portfolio_fit\t0.8
        capital_risk_pct\t1.4
        """)
        let peerB = ControlPlaneSnapshot.parse(tsv: """
        login\t42
        magic\t99
        chart_id\t9
        symbol\tEURJPY
        bar_time\t10000
        direction\t0
        signal_intensity\t0.8
        confidence\t0.5
        reliability\t0.5
        macro_quality\t0.2
        policy_trade_prob\t0.5
        policy_no_trade_prob\t0.4
        policy_capital_efficiency\t0.6
        policy_portfolio_fit\t0.4
        capital_risk_pct\t0.6
        """)
        let stale = ControlPlaneSnapshot.parse(tsv: """
        login\t42
        magic\t99
        chart_id\t10
        symbol\tGBPUSD
        bar_time\t1
        signal_intensity\t4
        """)
        let selfSnapshot = ControlPlaneSnapshot.parse(tsv: """
        login\t42
        magic\t99
        chart_id\t7
        symbol\tEURUSD
        bar_time\t10000
        signal_intensity\t4
        """)

        let aggregate = ControlPlaneScoring.aggregate(
            anchorSymbol: "EURUSD",
            direction: 1,
            identity: identity,
            nowUTC: 10_100,
            snapshots: [peerA, peerB, stale, selfSnapshot],
            correlationWeight: { _, _ in 1.0 },
            directionalAlignment: { _, direction, _, otherDirection in direction == otherDirection ? 1.0 : 0.0 }
        )

        XCTAssertEqual(aggregate.peerCount, 2)
        XCTAssertEqual(aggregate.grossIntensity, 2.0, accuracy: 1e-9)
        XCTAssertEqual(aggregate.directionalIntensity, 1.2, accuracy: 1e-9)
        XCTAssertEqual(aggregate.meanTradeProb, 0.6, accuracy: 1e-9)
        XCTAssertEqual(aggregate.meanNoTradeProb, 0.3, accuracy: 1e-9)
        XCTAssertEqual(aggregate.meanCapitalEfficiency, 0.75, accuracy: 1e-9)
        XCTAssertEqual(aggregate.maxCapitalRiskPct, 1.4, accuracy: 1e-9)
        XCTAssertGreaterThan(aggregate.score, 0.0)

        let profile = PortfolioSupervisorProfile()
        XCTAssertGreaterThan(ControlPlaneScoring.portfolioSupervisorScore(direction: 1, aggregate: aggregate, profile: profile), 0.0)
    }

    func testSupervisorServiceScoreMatchesLegacyBlend() {
        let state = SupervisorServiceState.parse(tsv: """
        symbol\tEURUSD
        generated_at\t1000
        gross_pressure\t1.0
        directional_long_pressure\t0.5
        directional_short_pressure\t0.2
        macro_pressure\t0.4
        concentration_pressure\t0.3
        reduce_bias\t0.2
        exit_bias\t0.1
        supervisor_score\t0.8
        """, nowUTC: 1_100)

        XCTAssertEqual(ControlPlaneScoring.supervisorServiceDirectionalPressure(state, direction: 1), 0.5)
        XCTAssertEqual(ControlPlaneScoring.supervisorServiceDirectionalPressure(state, direction: 0), 0.2)
        XCTAssertGreaterThan(ControlPlaneScoring.supervisorServiceScore(direction: 1, state: state), 0.0)
        XCTAssertEqual(ControlPlaneScoring.supervisorServiceScore(direction: 1, state: SupervisorServiceState()), 0.0)
    }

    func testFileRepositoryLoadsProfilesAndSnapshots() throws {
        let (root, repository) = try temporaryRepository()
        defer { try? FileManager.default.removeItem(at: root) }

        try write("profile_name\tlive\n", relativePath: ControlPlanePaths.liveDeploymentProfileFile(symbol: "EURUSD"), root: root)
        try write("profile_name\tportfolio\n", relativePath: ControlPlaneConstants.portfolioSupervisorFile, root: root)
        try write("champion_only\t1\n", relativePath: ControlPlanePaths.studentRouterProfileFile(symbol: "EURUSD"), root: root)
        try write("enabled\t1\n", relativePath: ControlPlanePaths.adaptiveRouterProfileFile(symbol: "EURUSD"), root: root)
        try write("""
        symbol\t__GLOBAL__
        generated_at\t1000
        gross_pressure\t0.5
        """, relativePath: ControlPlaneConstants.supervisorServiceGlobalFile, root: root)
        try write("""
        symbol\t__GLOBAL__
        generated_at\t1000
        long_block\t1
        """, relativePath: ControlPlaneConstants.supervisorCommandGlobalFile, root: root)
        try write("""
        login\t42
        magic\t99
        chart_id\t8
        symbol\tUSDJPY
        bar_time\t9900
        signal_intensity\t1
        """, relativePath: "\(ControlPlaneConstants.directory)/cp_peer.tsv", root: root)
        try write("""
        login\t42
        magic\t99
        chart_id\t9
        symbol\tGBPUSD
        bar_time\t1
        signal_intensity\t4
        """, relativePath: "\(ControlPlaneConstants.directory)/cp_stale.tsv", root: root)

        XCTAssertEqual(try repository.loadLiveDeploymentProfile(symbol: "EURUSD")?.profileName, "live")
        XCTAssertEqual(try repository.loadPortfolioSupervisorProfile()?.profileName, "portfolio")
        XCTAssertTrue(try XCTUnwrap(repository.loadStudentRouterProfile(symbol: "EURUSD")).championOnly)
        XCTAssertTrue(try XCTUnwrap(repository.loadAdaptiveRouterProfile(symbol: "EURUSD")).enabled)
        XCTAssertEqual(try repository.loadSupervisorServiceState(symbol: "EURUSD", nowUTC: 1_100)?.symbol, "EURUSD")
        XCTAssertTrue(try XCTUnwrap(repository.loadSupervisorCommandState(symbol: "EURUSD", nowUTC: 1_100)).longBlock)
        XCTAssertEqual(try repository.loadControlPlaneSnapshots().count, 2)

        let aggregate = try repository.loadControlPlaneAggregate(
            anchorSymbol: "EURUSD",
            direction: 1,
            identity: ControlPlanePeerIdentity(login: 42, magic: 99, chartID: 7),
            nowUTC: 10_000,
            correlationWeight: { _, _ in 1.0 },
            directionalAlignment: { _, _, _, _ in 1.0 }
        )
        XCTAssertEqual(aggregate.peerCount, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("\(ControlPlaneConstants.directory)/cp_stale.tsv").path))
    }
}
