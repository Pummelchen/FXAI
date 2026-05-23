import FXBacktestCore
import Foundation

public struct FX7: FXBacktestPluginV1 {
    public let descriptor = FXBacktestPluginDescriptor(
        id: "com.fxbacktest.plugins.fx7.v1",
        displayName: "FX7",
        version: "1.0.0",
        summary: "OHLC-only backtest port of the FX7 signal-timeframe momentum, regime, panic, correlation, novelty, and risk-control core.",
        author: "Pummelchen",
        supportsCPU: true,
        supportsMetal: true
    )

    public let parameterDefinitions: [ParameterDefinition]

    public init() {
        self.parameterDefinitions = Self.makeParameterDefinitions()
    }

    public var metalKernel: MetalKernelV1? {
        MetalKernelV1(source: Self.metalSource, entryPoint: "fx7_core_v1", maxPassesPerCommandBuffer: 4_096)
    }

    public var accelerationDescriptor: PluginAccelerationDescriptor {
        PluginAccelerationDescriptor(
            pluginIdentifier: descriptor.id,
            supportedBackends: [.swiftScalar, .swiftSIMD, .metal],
            metalEntryPoint: "fx7_core_v1",
            ir: PluginAccelerationIR(
                requiredColumns: [
                    PluginAccelerationInputColumn(field: "open"),
                    PluginAccelerationInputColumn(field: "high"),
                    PluginAccelerationInputColumn(field: "low"),
                    PluginAccelerationInputColumn(field: "close")
                ],
                operations: [
                    PluginAccelerationIROperation(
                        opcode: "fx7_closed_bar_core",
                        inputs: ["open", "high", "low", "close"],
                        outputs: ["position_signal", "trade_ledger", "pass_metrics"]
                    )
                ]
            )
        )
    }

    public func runPass(
        market: OhlcDataSeries,
        parameters: ParameterVector,
        context: BacktestContext
    ) throws -> BacktestPassResult {
        try runPass(marketUniverse: market.universe, parameters: parameters, context: context)
    }

    public func runPass(
        marketUniverse: OhlcMarketUniverse,
        parameters: ParameterVector,
        context: BacktestContext
    ) throws -> BacktestPassResult {
        var runtime = FX7Runtime(
            plugin: self,
            marketUniverse: marketUniverse,
            parameters: parameters,
            context: context
        )
        try runtime.run()
        return runtime.result()
    }
}

private extension FX7 {
    static func makeParameterDefinitions() -> [ParameterDefinition] {
        [
            parameter("signal_stride_bars", "Signal Timeframe Minutes", 15, 1, 1, 60, .integer),
            parameter("h1", "H1", 8, 2, 1, 60, .integer),
            parameter("h2", "H2", 24, 4, 1, 180, .integer),
            parameter("h3", "H3", 72, 8, 1, 360, .integer),
            parameter("vol_short_half_life", "Vol Short Half Life", 8, 2, 1, 64, .integer),
            parameter("vol_long_half_life", "Vol Long Half Life", 32, 8, 1, 180, .integer),
            parameter("atr_window", "ATR Window", 14, 4, 1, 80, .integer),
            parameter("base_entry_threshold", "Entry Threshold", 0.02, 0.005, 0.005, 0.12, .decimal),
            parameter("base_exit_threshold", "Exit Threshold", 0.01, 0.0, 0.005, 0.08, .decimal),
            parameter("reversal_threshold", "Reversal Threshold", 0.04, 0.01, 0.005, 0.16, .decimal),
            parameter("theta0", "Confidence Center", 0.05, 0.0, 0.01, 0.20, .decimal),
            parameter("confidence_slope", "Confidence Slope", 6.0, 1.0, 0.5, 16.0, .decimal),
            parameter("alpha_smooth", "Signal Smoothing", 0.55, 0.05, 0.05, 1.0, .decimal),
            parameter("tanh_scale", "Trend Tanh Scale", 2.0, 0.5, 0.25, 5.0, .decimal),
            parameter("gamma_a", "Alignment Gamma", 0.8, 0.0, 0.1, 3.0, .decimal),
            parameter("gamma_er", "Efficiency Gamma", 0.8, 0.0, 0.1, 3.0, .decimal),
            parameter("gamma_v", "Vol Penalty Gamma", 0.6, 0.0, 0.1, 3.0, .decimal),
            parameter("gamma_d", "Reversal Penalty Gamma", 0.4, 0.0, 0.1, 3.0, .decimal),
            parameter("v0", "Vol Ratio Neutral", 1.50, 0.50, 0.10, 3.0, .decimal),
            parameter("gamma_b", "Breakout Gamma", 0.5, 0.1, 0.1, 3.0, .decimal),
            parameter("gamma_p", "Panic Gamma", 0.4, 0.0, 0.1, 3.0, .decimal),
            parameter("v_panic", "Panic Vol Ratio", 1.80, 0.50, 0.10, 4.0, .decimal),
            parameter("eta_cost", "Cost Threshold Penalty", 0.05, 0.0, 0.01, 0.50, .decimal),
            parameter("eta_vol", "Vol Threshold Penalty", 0.02, 0.0, 0.01, 0.50, .decimal),
            parameter("eta_breakout", "Breakout Threshold Penalty", 0.02, 0.0, 0.01, 0.50, .decimal),
            parameter("gamma_cost", "Cost Gate Gamma", 0.50, 0.0, 0.10, 3.0, .decimal),
            parameter("round_trip_cost_pct", "Round Trip Cost Fraction", 0.0, 0.0, 0.00005, 0.005, .decimal),
            parameter("min_confidence", "Min Confidence", 0.30, 0.05, 0.05, 0.90, .decimal),
            parameter("min_regime_gate", "Min Regime Gate", 0.02, 0.0, 0.01, 0.50, .decimal),
            parameter("hard_min_regime_gate", "Hard Min Regime Gate", 0.01, 0.0, 0.01, 0.50, .decimal),
            parameter("min_exec_gate", "Min Execution Gate", 0.02, 0.0, 0.01, 1.0, .decimal),
            parameter("persistence_bars", "Persistence Bars", 1, 1, 1, 10, .integer),
            parameter("max_accepted_signals", "Max Accepted Signals", 10, 1, 1, 30, .integer),
            parameter("max_account_orders", "Max Account Orders", 10, 1, 1, 30, .integer),
            parameter("reference_lots", "Classic Reference Lots", 0.01, 0.01, 0.01, 1.0, .decimal),
            parameter("risk_per_trade_pct", "Risk Per Trade %", 0.35, 0.05, 0.05, 2.0, .decimal),
            parameter("max_portfolio_risk_pct", "Max Portfolio Risk %", 1.75, 0.25, 0.25, 10.0, .decimal),
            parameter("catastrophic_stop_atr", "Catastrophic Stop ATR", 3.0, 0.5, 0.25, 8.0, .decimal),
            parameter("single_position_take_profit_usd", "Single Position TP USD", 5.0, 0.0, 1.0, 100.0, .decimal),
            parameter("session_reset_profit_usd", "Session Reset Profit USD", 10.0, 0.0, 5.0, 250.0, .decimal),
            parameter("trailing_enabled", "Classic Trailing Enabled", 1, 0, 1, 1, .boolean),
            parameter("trail_start_pct", "Trail Start %", 50, 10, 10, 100, .integer),
            parameter("trail_spacing_pct", "Trail Spacing %", 20, 10, 10, 100, .integer),
            parameter("corr_lookback", "Correlation Lookback", 40, 10, 5, 160, .integer),
            parameter("shrinkage_lambda", "Correlation Shrinkage", 0.25, 0.0, 0.05, 1.0, .decimal),
            parameter("novelty_floor_weight", "Novelty Floor Weight", 0.50, 0.0, 0.05, 1.0, .decimal),
            parameter("novelty_cap", "Novelty Cap", 2.0, 0.25, 0.25, 5.0, .decimal),
            parameter("min_candidates_for_ortho", "Min Candidates For Ortho", 2, 2, 1, 10, .integer),
            parameter("uniqueness_min", "Uniqueness Min", 0.15, 0.0, 0.05, 1.0, .decimal),
            parameter("crowding_max", "Crowding Max", 0.90, 0.0, 0.05, 1.0, .decimal),
            parameter("fx_overlap_floor", "FX Overlap Floor", 0.35, -1.0, 0.05, 1.0, .decimal),
            parameter("class_overlap_floor", "Class Overlap Floor", 0.20, -1.0, 0.05, 1.0, .decimal),
            parameter("allow_long", "Allow Long", 1, 0, 1, 1, .boolean),
            parameter("allow_short", "Allow Short", 1, 0, 1, 1, .boolean),
            parameter("trend_weight_1", "Trend Weight 1", 0.45, 0.0, 0.05, 1.0, .decimal),
            parameter("trend_weight_2", "Trend Weight 2", 0.35, 0.0, 0.05, 1.0, .decimal),
            parameter("trend_weight_3", "Trend Weight 3", 0.20, 0.0, 0.05, 1.0, .decimal),
            parameter("er_window", "Efficiency Ratio Window", 12, 2, 1, 80, .integer),
            parameter("breakout_window", "Breakout Window", 12, 2, 1, 80, .integer),
            parameter("short_reversal_window", "Short Reversal Window", 4, 2, 1, 40, .integer)
        ]
    }

    static func parameter(
        _ key: String,
        _ displayName: String,
        _ value: Double,
        _ min: Double,
        _ step: Double,
        _ max: Double,
        _ kind: ParameterValueKind
    ) -> ParameterDefinition {
        do {
            return try ParameterDefinition(
                key: key,
                displayName: displayName,
                defaultValue: value,
                defaultMinimum: min,
                defaultStep: step,
                defaultMaximum: max,
                valueKind: kind
            )
        } catch {
            preconditionFailure("Invalid FX7 parameter definition '\(key)': \(error)")
        }
    }
}

private struct FX7Runtime {
    let plugin: FX7
    let marketUniverse: OhlcMarketUniverse
    let parameters: ParameterVector
    let context: BacktestContext
    let inputs: FX7Inputs
    let signalSeriesBySymbol: [String: FX7SignalSeries]
    var states: [FX7SymbolState]
    var broker: FX7PortfolioBroker
    var barsProcessed = 0
    var flags: UInt32 = 0

    init(
        plugin: FX7,
        marketUniverse: OhlcMarketUniverse,
        parameters: ParameterVector,
        context: BacktestContext
    ) {
        self.plugin = plugin
        self.marketUniverse = marketUniverse
        self.parameters = parameters
        self.context = context
        let resolvedInputs = FX7Inputs(parameters: parameters)
        self.inputs = resolvedInputs
        var compressedSeries: [String: FX7SignalSeries] = [:]
        var initialStates: [FX7SymbolState] = marketUniverse.symbols.compactMap { symbol -> FX7SymbolState? in
            guard let series = marketUniverse[symbol] else { return nil }
            compressedSeries[symbol.uppercased()] = FX7SignalSeries(
                symbol: symbol,
                source: series,
                signalStrideBars: resolvedInputs.signalStrideBars
            )
            return FX7SymbolState(symbol: symbol, series: series, retHistoryLength: FX7Math.retHistoryLength(resolvedInputs))
        }
        self.signalSeriesBySymbol = compressedSeries
        let symbolCount = initialStates.count
        for index in initialStates.indices {
            initialStates[index].correlations = Array(repeating: 0.0, count: symbolCount)
            initialStates[index].effectiveCorrelations = Array(repeating: 0.0, count: symbolCount)
        }
        self.states = initialStates
        self.broker = FX7PortfolioBroker(context: context)
    }

    mutating func run() throws {
        guard !states.isEmpty else {
            flags |= FX7ResultFlag.noTradableSymbols.rawValue
            return
        }
        for index in 0..<marketUniverse.count {
            barsProcessed = index + 1
            try markToMarket(index: index)
            try applyClassicOverlays()
            if shouldEvaluateSignals(at: index) {
                let targets = refreshSignalsAndBuildTargets(index: index)
                try closeExitedPositions(targets: targets, index: index)
                try openAcceptedTargets(targets: targets, index: index)
            }
        }
        try finish()
    }

    private func shouldEvaluateSignals(at index: Int) -> Bool {
        guard let primary = signalSeriesBySymbol[marketUniverse.primarySymbol],
              let signalIndex = primary.signalIndexForExecution(index) else {
            return false
        }
        return signalIndex + 1 >= inputs.requiredBars
    }

    mutating func markToMarket(index: Int) throws {
        for state in states {
            guard let series = marketUniverse[state.symbol], index < series.count else { continue }
            try broker.markToMarket(symbol: state.symbol, price: series.close[index], digits: state.digits)
        }
    }

    mutating func refreshSignalsAndBuildTargets(index: Int) -> [Int: Int] {
        for stateIndex in states.indices {
            refreshFeatures(stateIndex: stateIndex, executionIndex: index)
        }
        buildCorrelationMatrices()
        updatePanicGateAndScores()
        computeNoveltyOverlay()
        return buildTradeTargets()
    }

    mutating func refreshFeatures(stateIndex: Int, executionIndex: Int) {
        guard var state = states[safe: stateIndex] else { return }
        guard let series = signalSeriesBySymbol[state.symbol],
              let decisionIndex = series.signalIndexForExecution(executionIndex) else {
            state.dataOK = false
            states[stateIndex] = state
            return
        }
        guard decisionIndex + 1 >= inputs.requiredBars,
              decisionIndex < series.count else {
            state.dataOK = false
            states[stateIndex] = state
            return
        }

        let retCount = min(state.stdReturns.count, decisionIndex)
        let sigmaShort = FX7Math.ewmaStd(series: series, decisionIndex: decisionIndex, returnsCount: retCount, halfLife: inputs.volShortHalfLife)
        let sigmaLong = FX7Math.ewmaStd(series: series, decisionIndex: decisionIndex, returnsCount: retCount, halfLife: inputs.volLongHalfLife)
        guard sigmaLong > FX7Math.eps else {
            state.dataOK = false
            states[stateIndex] = state
            return
        }

        let closeNow = FX7Math.price(series.close[decisionIndex])
        let z1 = FX7Math.clip(log(closeNow / FX7Math.price(series.close[decisionIndex - inputs.h1])) / (sigmaLong * sqrt(Double(inputs.h1)) + FX7Math.eps), -6, 6)
        let z2 = FX7Math.clip(log(closeNow / FX7Math.price(series.close[decisionIndex - inputs.h2])) / (sigmaLong * sqrt(Double(inputs.h2)) + FX7Math.eps), -6, 6)
        let z3 = FX7Math.clip(log(closeNow / FX7Math.price(series.close[decisionIndex - inputs.h3])) / (sigmaLong * sqrt(Double(inputs.h3)) + FX7Math.eps), -6, 6)
        let trendWeights = inputs.normalizedTrendWeights

        state.sigmaShort = sigmaShort
        state.sigmaLong = sigmaLong
        state.atrPct = FX7Math.atrPct(series: series, decisionIndex: decisionIndex, window: inputs.atrWindow)
        state.momentum = trendWeights.0 * tanh(z1 / inputs.tanhScale)
            + trendWeights.1 * tanh(z2 / inputs.tanhScale)
            + trendWeights.2 * tanh(z3 / inputs.tanhScale)
        state.alignment = abs(trendWeights.0 * Double(FX7Math.sign(z1))
            + trendWeights.1 * Double(FX7Math.sign(z2))
            + trendWeights.2 * Double(FX7Math.sign(z3)))

        var pathSum = 0.0
        for lookback in 0..<inputs.erWindow {
            let newer = FX7Math.price(series.close[decisionIndex - lookback])
            let older = FX7Math.price(series.close[decisionIndex - lookback - 1])
            pathSum += abs(log(newer / older))
        }
        let netMove = abs(log(closeNow / FX7Math.price(series.close[decisionIndex - inputs.erWindow])))
        state.efficiency = netMove / (pathSum + FX7Math.eps)
        state.volRatio = sigmaShort / (sigmaLong + FX7Math.eps)

        let zrev = FX7Math.clip(log(closeNow / FX7Math.price(series.close[decisionIndex - inputs.shortReversalWindow])) / (sigmaLong * sqrt(Double(inputs.shortReversalWindow)) + FX7Math.eps), -6, 6)
        state.reversalPenalty = max(0, -Double(FX7Math.sign(state.momentum)) * zrev)

        let breakoutCloses = (1...inputs.breakoutWindow).map { FX7Math.price(series.close[decisionIndex - $0]) }
        let highest = breakoutCloses.max() ?? closeNow
        let lowest = breakoutCloses.min() ?? closeNow
        let mid = 0.5 * (highest + lowest)
        let halfRange = 0.5 * max(highest - lowest, FX7Math.eps)
        state.breakout = 0.5 * (1 + tanh(Double(FX7Math.sign(state.momentum)) * (closeNow - mid) / halfRange))

        state.regimeGate = pow(max(state.alignment, 0), inputs.gammaA)
            * pow(max(state.efficiency, 0), inputs.gammaER)
            * exp(-inputs.gammaV * max(0, state.volRatio - inputs.v0))
            * exp(-inputs.gammaD * state.reversalPenalty * max(0, state.volRatio - inputs.v0))

        let k = inputs.roundTripCostPct / (state.atrPct + FX7Math.eps)
        state.costLong = k
        state.costShort = k
        state.cost = k
        state.execLong = exp(-inputs.gammaCost * k)
        state.execShort = exp(-inputs.gammaCost * k)
        state.execGate = min(state.execLong, state.execShort)
        state.composite = state.momentum
        for lag in 0..<state.stdReturns.count {
            let newer = FX7Math.price(series.close[decisionIndex - lag])
            let older = FX7Math.price(series.close[decisionIndex - lag - 1])
            state.stdReturns[lag] = log(newer / older) / (sigmaLong + FX7Math.eps)
        }
        state.dataOK = true
        states[stateIndex] = state
    }

    mutating func buildCorrelationMatrices() {
        for i in states.indices {
            for j in states.indices {
                let rho: Double
                if i == j {
                    rho = 1
                } else if states[i].dataOK && states[j].dataOK {
                    rho = FX7Math.pearson(states[i].stdReturns, states[j].stdReturns, count: inputs.corrLookback)
                } else {
                    rho = 0
                }
                states[i].correlations[j] = rho
                var effective = rho
                if i == j {
                    effective = 1
                } else if states[i].dataOK && states[j].dataOK {
                    if states[i].sharesCurrency(with: states[j]) {
                        effective = max(effective, inputs.fxOverlapFloor)
                    } else if states[i].sameClassOverlap(with: states[j]) {
                        effective = max(effective, inputs.classOverlapFloor)
                    }
                }
                states[i].effectiveCorrelations[j] = FX7Math.clip(effective, -1, 1)
            }
        }
    }

    mutating func updatePanicGateAndScores() {
        var universeReturns = Array(repeating: 0.0, count: FX7Math.retHistoryLength(inputs))
        for lag in universeReturns.indices {
            var sum = 0.0
            var count = 0
            for state in states where state.dataOK {
                sum += state.stdReturns[lag]
                count += 1
            }
            universeReturns[lag] = count > 0 ? sum / Double(count) : 0
        }

        let panicCount = min(5, universeReturns.count)
        let zu5 = universeReturns.prefix(panicCount).reduce(0, +) / sqrt(Double(max(1, panicCount)))
        let su = FX7Math.ewmaStdNewestFirst(universeReturns, count: min(20, universeReturns.count), halfLife: 20)
        let lu = FX7Math.ewmaStdNewestFirst(universeReturns, count: min(100, universeReturns.count), halfLife: 100)
        let vu = su / (lu + FX7Math.eps)

        for index in states.indices {
            guard states[index].dataOK else {
                states[index].resetSignalState()
                continue
            }
            let alphaDirection = FX7Math.sign(states[index].composite) == 0
                ? FX7Math.sign(states[index].momentum)
                : FX7Math.sign(states[index].composite)
            states[index].panicGate = exp(-inputs.gammaP * max(0, vu - inputs.vPanic) * max(0, -Double(alphaDirection) * zu5))
            let breakoutWeight = 0.5 + 0.5 * pow(FX7Math.clip(states[index].breakout, 0, 1), inputs.gammaB)
            states[index].coreScore = states[index].composite * states[index].panicGate * breakoutWeight
            states[index].smoothedScore = inputs.alphaSmooth * states[index].coreScore
                + (1 - inputs.alphaSmooth) * states[index].smoothedScore
            let signalMagnitude = max(abs(states[index].smoothedScore), abs(states[index].coreScore))
            states[index].confidence = FX7Math.sigmoid(inputs.confidenceSlope * (signalMagnitude - inputs.theta0))
            let nextDirection = signalDirection(index)
            if nextDirection == 0 {
                states[index].entryDirection = 0
                states[index].persistenceCount = 0
            } else if states[index].entryDirection == nextDirection {
                states[index].persistenceCount += 1
            } else {
                states[index].entryDirection = nextDirection
                states[index].persistenceCount = 1
            }
        }
    }

    mutating func computeNoveltyOverlay() {
        for index in states.indices {
            states[index].omega = 1
            states[index].rank = abs(states[index].smoothedScore) * states[index].confidence
        }
        let candidates = states.indices.filter { candidateMeetsMinimumGates($0, states[$0].entryDirection) }
        guard candidates.count >= inputs.minCandidatesForOrtho else { return }

        var matrix = Array(repeating: Array(repeating: 0.0, count: candidates.count), count: candidates.count)
        var rhs = Array(repeating: 0.0, count: candidates.count)
        for row in candidates.indices {
            let i = candidates[row]
            rhs[row] = states[i].smoothedScore
            for col in candidates.indices {
                let j = candidates[col]
                let raw = states[i].correlations[j]
                matrix[row][col] = row == col
                    ? (1 - inputs.shrinkageLambda) * raw + inputs.shrinkageLambda
                    : (1 - inputs.shrinkageLambda) * raw
            }
        }
        guard let solution = FX7Math.solveLinearSystem(matrix, rhs) else { return }
        for row in candidates.indices {
            let index = candidates[row]
            let psi = Double(FX7Math.sign(states[index].smoothedScore)) * solution[row]
            let omega = FX7Math.clip(psi / (abs(states[index].smoothedScore) + FX7Math.eps), 0, inputs.noveltyCap)
            states[index].omega = omega
            states[index].rank = abs(states[index].smoothedScore)
                * states[index].confidence
                * (inputs.noveltyFloorWeight + (1 - inputs.noveltyFloorWeight) * omega)
        }
    }

    func buildTradeTargets() -> [Int: Int] {
        var records: [FX7Candidate] = []
        for index in states.indices {
            guard candidateMeetsMinimumGates(index, states[index].entryDirection),
                  states[index].persistenceCount >= inputs.persistenceBars else { continue }
            let priority = candidatePriority(index, dir: states[index].entryDirection)
            guard priority > FX7Math.eps else { continue }
            records.append(FX7Candidate(index: index, direction: states[index].entryDirection, priority: priority))
        }
        records.sort { $0.priority > $1.priority }

        var accepted: [Int] = []
        var targets: [Int: Int] = [:]
        for record in records {
            guard accepted.count < inputs.maxAcceptedSignals else { break }
            guard candidatePassesReversalThreshold(record.index, targetDirection: record.direction) else { continue }
            let uniqueness = candidateUniqueness(record.index, dir: record.direction, accepted: accepted)
            guard uniqueness + FX7Math.eps >= inputs.uniquenessMin else { continue }
            let crowding = portfolioCrowdingIfAdded(record.index, dir: record.direction, accepted: accepted)
            guard crowding - FX7Math.eps <= inputs.crowdingMax else { continue }
            targets[record.index] = record.direction
            accepted.append(record.index)
        }
        return targets
    }

    mutating func closeExitedPositions(targets: [Int: Int], index: Int) throws {
        for stateIndex in states.indices {
            let state = states[stateIndex]
            guard let position = broker.positions[state.symbol],
                  let series = marketUniverse[state.symbol] else { continue }
            let currentDirection = position.direction.rawValue
            let targetDirection = targets[stateIndex] ?? 0
            if shouldExitManagedDirection(stateIndex, currentDirection: currentDirection)
                || (targetDirection != 0 && targetDirection != currentDirection && candidatePassesReversalThreshold(stateIndex, targetDirection: targetDirection)) {
                try broker.close(symbol: state.symbol, price: series.close[index], digits: state.digits)
            }
        }
    }

    mutating func openAcceptedTargets(targets: [Int: Int], index: Int) throws {
        let orderedTargets = targets.keys.sorted {
            candidatePriority($0, dir: targets[$0] ?? 0) > candidatePriority($1, dir: targets[$1] ?? 0)
        }
        for stateIndex in orderedTargets {
            guard broker.positions.count < inputs.maxAccountOrders,
                  let direction = targets[stateIndex],
                  direction != 0,
                  let series = marketUniverse[states[stateIndex].symbol] else { continue }
            let tradeDirection: TradeDirection = direction > 0 ? .long : .short
            if broker.positions[states[stateIndex].symbol]?.direction == tradeDirection { continue }
            let lots = targetLots(for: stateIndex, price: series.close[index])
            guard lots > 0 else { continue }
            try broker.open(
                symbol: states[stateIndex].symbol,
                direction: tradeDirection,
                lots: lots,
                price: series.close[index],
                digits: states[stateIndex].digits,
                stopRiskCash: stopRiskCash(stateIndex: stateIndex, price: series.close[index], lots: lots)
            )
        }
    }

    mutating func applyClassicOverlays() throws {
        let symbols = Array(broker.positions.keys)
        for symbol in symbols {
            guard var position = broker.positions[symbol],
                  let state = states.first(where: { $0.symbol == symbol }),
                  let price = broker.currentPrices[symbol] else { continue }
            let profit = broker.profit(position: position, price: price)
            position.peakProfit = max(position.peakProfit, profit)
            broker.positions[symbol] = position
            if inputs.trailingEnabled && inputs.singlePositionTakeProfitUSD > 0 {
                let start = inputs.singlePositionTakeProfitUSD * Double(inputs.trailStartPct) / 100
                let giveback = Double(inputs.trailSpacingPct) / 100
                if position.peakProfit >= start && profit <= position.peakProfit * (1 - giveback) {
                    try broker.close(symbol: symbol, price: price, digits: state.digits)
                }
            } else if inputs.singlePositionTakeProfitUSD > 0 && profit >= inputs.singlePositionTakeProfitUSD {
                try broker.close(symbol: symbol, price: price, digits: state.digits)
            }
        }
        if inputs.sessionResetProfitUSD > 0 && broker.equity - broker.sessionStartEquity >= inputs.sessionResetProfitUSD {
            try broker.closeAll(digitsBySymbol: digitsBySymbol())
            broker.sessionStartEquity = broker.equity
        }
    }

    mutating func finish() throws {
        try markToMarket(index: max(0, marketUniverse.count - 1))
        try broker.closeAll(digitsBySymbol: digitsBySymbol())
    }

    func result() -> BacktestPassResult {
        BacktestPassResult(
            passIndex: parameters.combinationIndex,
            pluginIdentifier: plugin.descriptor.id,
            engine: context.settings.target,
            parameters: parameters.snapshots,
            netProfit: broker.netProfit,
            grossProfit: broker.grossProfit,
            grossLoss: broker.grossLoss,
            maxDrawdown: broker.maxDrawdown,
            totalTrades: broker.totalTrades,
            winningTrades: broker.winningTrades,
            losingTrades: broker.losingTrades,
            winRate: broker.winRate,
            profitFactor: broker.profitFactor,
            barsProcessed: barsProcessed,
            flags: flags
        )
    }

    func signalDirection(_ index: Int) -> Int {
        let longThreshold = max(entryThresholdDirectional(index, dir: 1), inputs.baseEntryThreshold)
        let shortThreshold = max(entryThresholdDirectional(index, dir: -1), inputs.baseEntryThreshold)
        if inputs.allowLong, states[index].smoothedScore >= longThreshold { return 1 }
        if inputs.allowShort, states[index].smoothedScore <= -shortThreshold { return -1 }
        let alphaDirection = FX7Math.sign(states[index].composite)
        if inputs.allowLong, alphaDirection > 0, states[index].smoothedScore >= 0.60 * longThreshold, states[index].coreScore > 0 { return 1 }
        if inputs.allowShort, alphaDirection < 0, states[index].smoothedScore <= -0.60 * shortThreshold, states[index].coreScore < 0 { return -1 }
        return 0
    }

    func candidateMeetsMinimumGates(_ index: Int, _ dir: Int) -> Bool {
        guard states[index].dataOK, dir != 0 else { return false }
        return states[index].confidence >= inputs.minConfidence
            && states[index].regimeGate >= inputs.minRegimeGate
            && directionalExecGate(index, dir: dir) >= inputs.minExecGate
    }

    func shouldExitManagedDirection(_ index: Int, currentDirection: Int) -> Bool {
        guard states[index].dataOK else { return true }
        if states[index].regimeGate < inputs.hardMinRegimeGate || directionalExecGate(index, dir: currentDirection) < inputs.minExecGate {
            return true
        }
        let threshold = exitThresholdDirectional(index, dir: currentDirection)
        return currentDirection > 0
            ? states[index].smoothedScore <= threshold
            : states[index].smoothedScore >= -threshold
    }

    func candidatePassesReversalThreshold(_ index: Int, targetDirection: Int) -> Bool {
        guard let position = broker.positions[states[index].symbol] else { return true }
        if position.direction.rawValue == targetDirection { return true }
        return abs(states[index].smoothedScore) + FX7Math.eps >= inputs.reversalThreshold
    }

    func candidatePriority(_ index: Int, dir: Int) -> Double {
        let rawRank = abs(states[index].smoothedScore) * FX7Math.clip(states[index].confidence, 0, 1)
        let noveltyWeight = inputs.noveltyFloorWeight + (1 - inputs.noveltyFloorWeight) * states[index].omega
        let noveltyRank = rawRank * max(noveltyWeight, 0)
        let gateWeight = 0.50 + 0.25 * FX7Math.clip(states[index].regimeGate, 0, 1) + 0.25 * FX7Math.clip(directionalExecGate(index, dir: dir), 0, 1)
        let momentumWeight = 0.60 + 0.40 * min(abs(states[index].momentum), 1)
        return noveltyRank * gateWeight * momentumWeight
    }

    func entryThresholdDirectional(_ index: Int, dir: Int) -> Double {
        (inputs.baseEntryThreshold
            + 0.20 * inputs.etaCost * max(0, directionalCost(index, dir: dir) - 1)
            + 0.20 * inputs.etaVol * max(0, states[index].volRatio - 1)
            + 0.10 * inputs.etaBreakout * (1 - FX7Math.clip(states[index].breakout, 0, 1)))
    }

    func exitThresholdDirectional(_ index: Int, dir: Int) -> Double {
        inputs.baseExitThreshold + 0.10 * inputs.etaCost * max(0, directionalCost(index, dir: dir) - 1)
    }

    func directionalCost(_ index: Int, dir: Int) -> Double {
        if dir > 0 { return states[index].costLong }
        if dir < 0 { return states[index].costShort }
        return states[index].cost
    }

    func directionalExecGate(_ index: Int, dir: Int) -> Double {
        if dir > 0 { return states[index].execLong }
        if dir < 0 { return states[index].execShort }
        return states[index].execGate
    }

    func candidateUniqueness(_ index: Int, dir: Int, accepted: [Int]) -> Double {
        var sumPositive = 0.0
        for other in accepted {
            let sameWay = Double(dir * states[other].entryDirection) * states[index].effectiveCorrelations[other]
            if sameWay > 0 { sumPositive += sameWay }
        }
        return 1 / (1 + sumPositive)
    }

    func portfolioCrowdingIfAdded(_ index: Int, dir: Int, accepted: [Int]) -> Double {
        guard !accepted.isEmpty else { return 0 }
        var sum = 0.0
        var pairs = 0
        for aIndex in accepted.indices {
            let a = accepted[aIndex]
            for b in accepted.dropFirst(aIndex + 1) {
                let sameWay = Double(states[a].entryDirection * states[b].entryDirection) * states[a].effectiveCorrelations[b]
                if sameWay > 0 { sum += sameWay }
                pairs += 1
            }
        }
        for other in accepted {
            let sameWay = Double(dir * states[other].entryDirection) * states[index].effectiveCorrelations[other]
            if sameWay > 0 { sum += sameWay }
            pairs += 1
        }
        return pairs > 0 ? sum / Double(pairs) : 0
    }

    func targetLots(for stateIndex: Int, price: Int64) -> Double {
        let riskPerLot = stopRiskCash(stateIndex: stateIndex, price: price, lots: 1)
        guard riskPerLot > FX7Math.eps else { return 0 }
        let perTradeRoom = broker.equity * inputs.riskPerTradePct / 100 / riskPerLot
        let portfolioRoom = max(0, broker.equity * inputs.maxPortfolioRiskPct / 100 - broker.openRiskCash) / riskPerLot
        let lots = min(inputs.referenceLots, perTradeRoom, portfolioRoom)
        return FX7Math.normalizeLots(lots)
    }

    func stopRiskCash(stateIndex: Int, price: Int64, lots: Double) -> Double {
        let priceScale = pow(10.0, Double(states[stateIndex].digits))
        let rawPrice = Double(price) / priceScale
        let stopDistance = rawPrice * max(states[stateIndex].atrPct, FX7Math.eps) * inputs.catastrophicStopATR
        return stopDistance * context.settings.contractSize * lots
    }

    func digitsBySymbol() -> [String: Int] {
        Dictionary(uniqueKeysWithValues: states.map { ($0.symbol, $0.digits) })
    }
}

private struct FX7Inputs {
    let signalStrideBars: Int
    let h1: Int
    let h2: Int
    let h3: Int
    let volShortHalfLife: Int
    let volLongHalfLife: Int
    let atrWindow: Int
    let baseEntryThreshold: Double
    let baseExitThreshold: Double
    let reversalThreshold: Double
    let theta0: Double
    let confidenceSlope: Double
    let alphaSmooth: Double
    let tanhScale: Double
    let gammaA: Double
    let gammaER: Double
    let gammaV: Double
    let gammaD: Double
    let v0: Double
    let gammaB: Double
    let gammaP: Double
    let vPanic: Double
    let etaCost: Double
    let etaVol: Double
    let etaBreakout: Double
    let gammaCost: Double
    let roundTripCostPct: Double
    let minConfidence: Double
    let minRegimeGate: Double
    let hardMinRegimeGate: Double
    let minExecGate: Double
    let persistenceBars: Int
    let maxAcceptedSignals: Int
    let maxAccountOrders: Int
    let referenceLots: Double
    let riskPerTradePct: Double
    let maxPortfolioRiskPct: Double
    let catastrophicStopATR: Double
    let singlePositionTakeProfitUSD: Double
    let sessionResetProfitUSD: Double
    let trailingEnabled: Bool
    let trailStartPct: Int
    let trailSpacingPct: Int
    let corrLookback: Int
    let shrinkageLambda: Double
    let noveltyFloorWeight: Double
    let noveltyCap: Double
    let minCandidatesForOrtho: Int
    let uniquenessMin: Double
    let crowdingMax: Double
    let fxOverlapFloor: Double
    let classOverlapFloor: Double
    let allowLong: Bool
    let allowShort: Bool
    let trendWeight1: Double
    let trendWeight2: Double
    let trendWeight3: Double
    let erWindow: Int
    let breakoutWindow: Int
    let shortReversalWindow: Int

    init(parameters: ParameterVector) {
        self.signalStrideBars = max(1, Int(Self.value("signal_stride_bars", parameters).rounded()))
        self.h1 = max(1, Int(Self.value("h1", parameters).rounded()))
        self.h2 = max(h1 + 1, Int(Self.value("h2", parameters).rounded()))
        self.h3 = max(h2 + 1, Int(Self.value("h3", parameters).rounded()))
        self.volShortHalfLife = max(1, Int(Self.value("vol_short_half_life", parameters).rounded()))
        self.volLongHalfLife = max(volShortHalfLife, Int(Self.value("vol_long_half_life", parameters).rounded()))
        self.atrWindow = max(2, Int(Self.value("atr_window", parameters).rounded()))
        self.baseEntryThreshold = max(0, Self.value("base_entry_threshold", parameters))
        self.baseExitThreshold = min(baseEntryThreshold, max(0, Self.value("base_exit_threshold", parameters)))
        self.reversalThreshold = max(0, Self.value("reversal_threshold", parameters))
        self.theta0 = max(0, Self.value("theta0", parameters))
        self.confidenceSlope = max(FX7Math.eps, Self.value("confidence_slope", parameters))
        self.alphaSmooth = FX7Math.clip(Self.value("alpha_smooth", parameters), 0, 1)
        self.tanhScale = max(FX7Math.eps, Self.value("tanh_scale", parameters))
        self.gammaA = max(0, Self.value("gamma_a", parameters))
        self.gammaER = max(0, Self.value("gamma_er", parameters))
        self.gammaV = max(0, Self.value("gamma_v", parameters))
        self.gammaD = max(0, Self.value("gamma_d", parameters))
        self.v0 = max(FX7Math.eps, Self.value("v0", parameters))
        self.gammaB = max(FX7Math.eps, Self.value("gamma_b", parameters))
        self.gammaP = max(0, Self.value("gamma_p", parameters))
        self.vPanic = max(FX7Math.eps, Self.value("v_panic", parameters))
        self.etaCost = max(0, Self.value("eta_cost", parameters))
        self.etaVol = max(0, Self.value("eta_vol", parameters))
        self.etaBreakout = max(0, Self.value("eta_breakout", parameters))
        self.gammaCost = max(0, Self.value("gamma_cost", parameters))
        self.roundTripCostPct = max(0, Self.value("round_trip_cost_pct", parameters))
        self.minConfidence = FX7Math.clip(Self.value("min_confidence", parameters), 0, 1)
        self.minRegimeGate = FX7Math.clip(Self.value("min_regime_gate", parameters), 0, 1)
        self.hardMinRegimeGate = FX7Math.clip(Self.value("hard_min_regime_gate", parameters), 0, 1)
        self.minExecGate = FX7Math.clip(Self.value("min_exec_gate", parameters), 0, 1)
        self.persistenceBars = max(1, Int(Self.value("persistence_bars", parameters).rounded()))
        self.maxAcceptedSignals = max(1, Int(Self.value("max_accepted_signals", parameters).rounded()))
        self.maxAccountOrders = max(1, Int(Self.value("max_account_orders", parameters).rounded()))
        self.referenceLots = max(0.01, Self.value("reference_lots", parameters))
        self.riskPerTradePct = max(FX7Math.eps, Self.value("risk_per_trade_pct", parameters))
        self.maxPortfolioRiskPct = max(riskPerTradePct, Self.value("max_portfolio_risk_pct", parameters))
        self.catastrophicStopATR = max(FX7Math.eps, Self.value("catastrophic_stop_atr", parameters))
        self.singlePositionTakeProfitUSD = max(0, Self.value("single_position_take_profit_usd", parameters))
        self.sessionResetProfitUSD = max(0, Self.value("session_reset_profit_usd", parameters))
        self.trailingEnabled = Self.value("trailing_enabled", parameters) >= 0.5
        self.trailStartPct = Int(FX7Math.clip(Self.value("trail_start_pct", parameters), 10, 100).rounded())
        self.trailSpacingPct = Int(FX7Math.clip(Self.value("trail_spacing_pct", parameters), 10, 100).rounded())
        self.corrLookback = max(10, Int(Self.value("corr_lookback", parameters).rounded()))
        self.shrinkageLambda = FX7Math.clip(Self.value("shrinkage_lambda", parameters), 0, 1)
        self.noveltyFloorWeight = FX7Math.clip(Self.value("novelty_floor_weight", parameters), 0, 1)
        self.noveltyCap = max(FX7Math.eps, Self.value("novelty_cap", parameters))
        self.minCandidatesForOrtho = max(2, Int(Self.value("min_candidates_for_ortho", parameters).rounded()))
        self.uniquenessMin = FX7Math.clip(Self.value("uniqueness_min", parameters), 0, 1)
        self.crowdingMax = FX7Math.clip(Self.value("crowding_max", parameters), 0, 1)
        self.fxOverlapFloor = FX7Math.clip(Self.value("fx_overlap_floor", parameters), -1, 1)
        self.classOverlapFloor = FX7Math.clip(Self.value("class_overlap_floor", parameters), -1, 1)
        self.allowLong = Self.value("allow_long", parameters) >= 0.5
        self.allowShort = Self.value("allow_short", parameters) >= 0.5
        self.trendWeight1 = max(0, Self.value("trend_weight_1", parameters))
        self.trendWeight2 = max(0, Self.value("trend_weight_2", parameters))
        self.trendWeight3 = max(0, Self.value("trend_weight_3", parameters))
        self.erWindow = max(2, Int(Self.value("er_window", parameters).rounded()))
        self.breakoutWindow = max(2, Int(Self.value("breakout_window", parameters).rounded()))
        self.shortReversalWindow = max(2, Int(Self.value("short_reversal_window", parameters).rounded()))
    }

    var normalizedTrendWeights: (Double, Double, Double) {
        let sum = trendWeight1 + trendWeight2 + trendWeight3
        guard sum > FX7Math.eps else { return (0.45, 0.35, 0.20) }
        return (trendWeight1 / sum, trendWeight2 / sum, trendWeight3 / sum)
    }

    var requiredBars: Int {
        let trendMax = max(h3, max(erWindow, max(breakoutWindow, shortReversalWindow)))
        return max(trendMax + 3, FX7Math.retHistoryLength(self) + 3) + 100
    }

    private static func value(_ key: String, _ parameters: ParameterVector) -> Double {
        parameters[key] ?? 0
    }
}

private struct FX7SignalSeries {
    let symbol: String
    let digits: Int
    let utcTimestamps: ContiguousArray<Int64>
    let open: ContiguousArray<Int64>
    let high: ContiguousArray<Int64>
    let low: ContiguousArray<Int64>
    let close: ContiguousArray<Int64>
    private let signalIndexByExecutionIndex: [Int]

    var count: Int { close.count }

    init(symbol: String, source: OhlcDataSeries, signalStrideBars: Int) {
        self.symbol = symbol.uppercased()
        self.digits = source.metadata.digits
        let strideBars = max(1, signalStrideBars)
        let bucketSeconds = Int64(strideBars * 60)
        var signalIndexByExecutionIndex = Array(repeating: -1, count: source.count)
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()

        var bucketStart: Int64?
        var bucketOpen: Int64 = 0
        var bucketHigh: Int64 = 0
        var bucketLow: Int64 = 0
        var bucketClose: Int64 = 0
        var bucketCount = 0

        func appendCompleteBucket(executionIndex: Int) {
            guard bucketCount == strideBars,
                  let bucketStart,
                  executionIndex < signalIndexByExecutionIndex.count else { return }
            let signalIndex = close.count
            utc.append(bucketStart)
            open.append(bucketOpen)
            high.append(bucketHigh)
            low.append(bucketLow)
            close.append(bucketClose)
            signalIndexByExecutionIndex[executionIndex] = signalIndex
        }

        func startBucket(at sourceIndex: Int, bucket: Int64) {
            bucketStart = bucket
            bucketOpen = source.open[sourceIndex]
            bucketHigh = source.high[sourceIndex]
            bucketLow = source.low[sourceIndex]
            bucketClose = source.close[sourceIndex]
            bucketCount = 1
        }

        for sourceIndex in 0..<source.count {
            let timestamp = source.utcTimestamps[sourceIndex]
            let bucket = timestamp - (timestamp % bucketSeconds)
            guard let currentBucket = bucketStart else {
                startBucket(at: sourceIndex, bucket: bucket)
                continue
            }
            if bucket != currentBucket {
                appendCompleteBucket(executionIndex: sourceIndex)
                startBucket(at: sourceIndex, bucket: bucket)
                continue
            }
            bucketHigh = max(bucketHigh, source.high[sourceIndex])
            bucketLow = min(bucketLow, source.low[sourceIndex])
            bucketClose = source.close[sourceIndex]
            bucketCount += 1
        }

        self.utcTimestamps = utc
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.signalIndexByExecutionIndex = signalIndexByExecutionIndex
    }

    func signalIndexForExecution(_ executionIndex: Int) -> Int? {
        guard executionIndex >= 0,
              executionIndex < signalIndexByExecutionIndex.count else { return nil }
        let signalIndex = signalIndexByExecutionIndex[executionIndex]
        return signalIndex >= 0 ? signalIndex : nil
    }
}

private struct FX7SymbolState {
    let symbol: String
    let base: String
    let quote: String
    let digits: Int
    var dataOK = false
    var sigmaShort = 0.0
    var sigmaLong = 0.0
    var atrPct = 0.0
    var momentum = 0.0
    var alignment = 0.0
    var efficiency = 0.0
    var volRatio = 0.0
    var reversalPenalty = 0.0
    var breakout = 0.0
    var regimeGate = 0.0
    var cost = 0.0
    var costLong = 0.0
    var costShort = 0.0
    var execGate = 1.0
    var execLong = 1.0
    var execShort = 1.0
    var panicGate = 1.0
    var composite = 0.0
    var coreScore = 0.0
    var smoothedScore = 0.0
    var confidence = 0.0
    var omega = 1.0
    var rank = 0.0
    var entryDirection = 0
    var persistenceCount = 0
    var stdReturns: [Double]
    var correlations: [Double]
    var effectiveCorrelations: [Double]

    init(symbol: String, series: OhlcDataSeries, retHistoryLength: Int) {
        self.symbol = symbol.uppercased()
        self.base = String(symbol.uppercased().prefix(3))
        self.quote = String(symbol.uppercased().suffix(3))
        self.digits = series.metadata.digits
        self.stdReturns = Array(repeating: 0.0, count: retHistoryLength)
        self.correlations = []
        self.effectiveCorrelations = []
    }

    mutating func resetSignalState() {
        dataOK = false
        panicGate = 1
        coreScore = 0
        confidence = 0
        omega = 1
        rank = 0
        entryDirection = 0
        persistenceCount = 0
    }

    func sharesCurrency(with other: FX7SymbolState) -> Bool {
        base == other.base || base == other.quote || quote == other.base || quote == other.quote
    }

    func sameClassOverlap(with other: FX7SymbolState) -> Bool {
        guard !sharesCurrency(with: other) else { return false }
        for bloc in [FX7CurrencyBloc.commodity, .european, .funding] {
            if bloc.touches(self) && bloc.touches(other) { return true }
        }
        return false
    }
}

private enum FX7CurrencyBloc {
    case commodity
    case european
    case funding

    func touches(_ state: FX7SymbolState) -> Bool {
        contains(state.base) || contains(state.quote)
    }

    private func contains(_ currency: String) -> Bool {
        switch self {
        case .commodity:
            return ["AUD", "NZD", "CAD", "NOK"].contains(currency)
        case .european:
            return ["EUR", "GBP", "CHF", "SEK", "NOK"].contains(currency)
        case .funding:
            return ["JPY", "CHF", "EUR"].contains(currency)
        }
    }
}

private struct FX7Candidate {
    let index: Int
    let direction: Int
    let priority: Double
}

private struct FX7PortfolioBroker {
    let context: BacktestContext
    let initialDeposit: Double
    var sessionStartEquity: Double
    private(set) var balance: Double
    private(set) var equity: Double
    private(set) var equityPeak: Double
    private(set) var maxDrawdown: Double
    private(set) var grossProfit = 0.0
    private(set) var grossLoss = 0.0
    private(set) var totalTrades = 0
    private(set) var winningTrades = 0
    private(set) var losingTrades = 0
    var positions: [String: FX7Position] = [:]
    var currentPrices: [String: Int64] = [:]

    init(context: BacktestContext) {
        self.context = context
        self.initialDeposit = context.settings.initialDeposit
        self.sessionStartEquity = context.settings.initialDeposit
        self.balance = context.settings.initialDeposit
        self.equity = context.settings.initialDeposit
        self.equityPeak = context.settings.initialDeposit
        self.maxDrawdown = 0
    }

    var netProfit: Double { balance - initialDeposit }
    var openRiskCash: Double { positions.values.reduce(0) { $0 + $1.stopRiskCash } }
    var winRate: Double { totalTrades == 0 ? 0 : Double(winningTrades) / Double(totalTrades) }
    var profitFactor: Double { grossLoss == 0 ? (grossProfit > 0 ? Double.infinity : 0) : grossProfit / abs(grossLoss) }

    mutating func open(symbol: String, direction: TradeDirection, lots: Double, price: Int64, digits: Int, stopRiskCash: Double) throws {
        let normalized = symbol.uppercased()
        if positions[normalized]?.direction == direction { return }
        if positions[normalized] != nil {
            try close(symbol: normalized, price: price, digits: digits)
        }
        positions[normalized] = FX7Position(direction: direction, entryPrice: price, lots: lots, digits: digits, stopRiskCash: stopRiskCash)
        try markToMarket(symbol: normalized, price: price, digits: digits)
    }

    mutating func close(symbol: String, price: Int64, digits: Int) throws {
        let normalized = symbol.uppercased()
        guard let position = positions.removeValue(forKey: normalized) else { return }
        let pnl = profit(position: position, price: price)
        balance += pnl
        grossProfit += max(0, pnl)
        grossLoss += min(0, pnl)
        totalTrades += 1
        if pnl >= 0 {
            winningTrades += 1
        } else {
            losingTrades += 1
        }
        try markToMarket(symbol: normalized, price: price, digits: digits)
    }

    mutating func closeAll(digitsBySymbol: [String: Int]) throws {
        for symbol in Array(positions.keys) {
            let position = positions[symbol]
            let price = currentPrices[symbol] ?? position?.entryPrice ?? 0
            try close(symbol: symbol, price: price, digits: digitsBySymbol[symbol] ?? position?.digits ?? context.digits)
        }
        try recomputeEquity()
    }

    mutating func markToMarket(symbol: String, price: Int64, digits _: Int) throws {
        currentPrices[symbol.uppercased()] = price
        try recomputeEquity()
    }

    func profit(position: FX7Position, price: Int64) -> Double {
        let direction = Double(position.direction.rawValue)
        let priceScale = pow(10.0, Double(position.digits))
        let priceDelta = Double(price - position.entryPrice) / priceScale
        return direction * priceDelta * context.settings.contractSize * position.lots
    }

    private mutating func recomputeEquity() throws {
        var floating = 0.0
        for (symbol, position) in positions {
            let price = currentPrices[symbol] ?? position.entryPrice
            floating += profit(position: position, price: price)
        }
        equity = balance + floating
        equityPeak = max(equityPeak, equity)
        maxDrawdown = max(maxDrawdown, equityPeak - equity)
    }
}

private struct FX7Position: Sendable, Hashable {
    let direction: TradeDirection
    let entryPrice: Int64
    let lots: Double
    let digits: Int
    let stopRiskCash: Double
    var peakProfit: Double = 0
}

private enum FX7ResultFlag: UInt32 {
    case noTradableSymbols = 1
}

private enum FX7Math {
    static let eps = 1.0e-12

    static func price(_ value: Int64) -> Double {
        max(Double(value), eps)
    }

    static func retHistoryLength(_ inputs: FX7Inputs) -> Int {
        max(inputs.corrLookback, max(inputs.volLongHalfLife + 10, 110))
    }

    static func sign(_ value: Double) -> Int {
        if value > 0 { return 1 }
        if value < 0 { return -1 }
        return 0
    }

    static func clip(_ value: Double, _ lower: Double, _ upper: Double) -> Double {
        min(max(value, lower), upper)
    }

    static func sigmoid(_ value: Double) -> Double {
        if value >= 0 {
            let e = exp(-value)
            return 1 / (1 + e)
        }
        let e = exp(value)
        return e / (1 + e)
    }

    static func normalizeLots(_ lots: Double) -> Double {
        guard lots.isFinite, lots > 0 else { return 0 }
        return (lots * 100).rounded(.down) / 100
    }

    static func ewmaStd(series: OhlcDataSeries, decisionIndex: Int, returnsCount: Int, halfLife: Int) -> Double {
        let count = min(returnsCount, decisionIndex)
        guard count > 1 else { return 0 }
        let lambda = exp(-log(2.0) / max(1.0, Double(halfLife)))
        var variance = 0.0
        var seeded = false
        for lag in stride(from: count - 1, through: 0, by: -1) {
            let newer = price(series.close[decisionIndex - lag])
            let older = price(series.close[decisionIndex - lag - 1])
            let ret = log(newer / older)
            variance = seeded ? lambda * variance + (1 - lambda) * ret * ret : ret * ret
            seeded = true
        }
        return sqrt(max(variance, eps))
    }

    static func ewmaStd(series: FX7SignalSeries, decisionIndex: Int, returnsCount: Int, halfLife: Int) -> Double {
        let count = min(returnsCount, decisionIndex)
        guard count > 1 else { return 0 }
        let lambda = exp(-log(2.0) / max(1.0, Double(halfLife)))
        var variance = 0.0
        var seeded = false
        for lag in stride(from: count - 1, through: 0, by: -1) {
            let newer = price(series.close[decisionIndex - lag])
            let older = price(series.close[decisionIndex - lag - 1])
            let ret = log(newer / older)
            variance = seeded ? lambda * variance + (1 - lambda) * ret * ret : ret * ret
            seeded = true
        }
        return sqrt(max(variance, eps))
    }

    static func ewmaStdNewestFirst(_ values: [Double], count: Int, halfLife: Int) -> Double {
        let safeCount = min(count, values.count)
        guard safeCount > 1 else { return sqrt(eps) }
        let lambda = exp(-log(2.0) / max(1.0, Double(halfLife)))
        var variance = 0.0
        var seeded = false
        for index in stride(from: safeCount - 1, through: 0, by: -1) {
            let ret = values[index]
            variance = seeded ? lambda * variance + (1 - lambda) * ret * ret : ret * ret
            seeded = true
        }
        return sqrt(max(variance, eps))
    }

    static func atrPct(series: OhlcDataSeries, decisionIndex: Int, window: Int) -> Double {
        guard decisionIndex >= window else { return eps }
        var sum = 0.0
        var count = 0
        for lag in stride(from: window - 1, through: 0, by: -1) {
            let index = decisionIndex - lag
            let previous = decisionIndex - lag - 1
            let high = price(series.high[index])
            let low = price(series.low[index])
            let previousClose = price(series.close[previous])
            let tr = max(high - low, max(abs(high - previousClose), abs(low - previousClose)))
            sum += tr
            count += 1
        }
        let close = price(series.close[decisionIndex])
        return max((sum / Double(max(1, count))) / close, eps)
    }

    static func atrPct(series: FX7SignalSeries, decisionIndex: Int, window: Int) -> Double {
        guard decisionIndex >= window else { return eps }
        var sum = 0.0
        var count = 0
        for lag in stride(from: window - 1, through: 0, by: -1) {
            let index = decisionIndex - lag
            let previous = decisionIndex - lag - 1
            let high = price(series.high[index])
            let low = price(series.low[index])
            let previousClose = price(series.close[previous])
            let tr = max(high - low, max(abs(high - previousClose), abs(low - previousClose)))
            sum += tr
            count += 1
        }
        let close = price(series.close[decisionIndex])
        return max((sum / Double(max(1, count))) / close, eps)
    }

    static func pearson(_ left: [Double], _ right: [Double], count requested: Int) -> Double {
        let count = min(requested, left.count, right.count)
        guard count > 1 else { return 0 }
        var sumX = 0.0
        var sumY = 0.0
        var sumX2 = 0.0
        var sumY2 = 0.0
        var sumXY = 0.0
        for index in 0..<count {
            let x = left[index]
            let y = right[index]
            sumX += x
            sumY += y
            sumX2 += x * x
            sumY2 += y * y
            sumXY += x * y
        }
        let n = Double(count)
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt(max(n * sumX2 - sumX * sumX, 0) * max(n * sumY2 - sumY * sumY, 0))
        guard denominator > eps else { return 0 }
        return clip(numerator / denominator, -1, 1)
    }

    static func solveLinearSystem(_ matrix: [[Double]], _ rhs: [Double]) -> [Double]? {
        let n = rhs.count
        guard matrix.count == n, matrix.allSatisfy({ $0.count == n }) else { return nil }
        var a = matrix
        var b = rhs
        var x = Array(repeating: 0.0, count: n)
        for col in 0..<n {
            var pivot = col
            var maxAbs = abs(a[col][col])
            for row in (col + 1)..<n where abs(a[row][col]) > maxAbs {
                pivot = row
                maxAbs = abs(a[row][col])
            }
            guard maxAbs > eps else { return nil }
            if pivot != col {
                a.swapAt(pivot, col)
                b.swapAt(pivot, col)
            }
            let diagonal = a[col][col]
            for row in (col + 1)..<n {
                let factor = a[row][col] / diagonal
                if factor == 0 { continue }
                a[row][col] = 0
                for j in (col + 1)..<n {
                    a[row][j] -= factor * a[col][j]
                }
                b[row] -= factor * b[col]
            }
        }
        for row in stride(from: n - 1, through: 0, by: -1) {
            var value = b[row]
            for col in (row + 1)..<n {
                value -= a[row][col] * x[col]
            }
            guard abs(a[row][row]) > eps else { return nil }
            x[row] = value / a[row][row]
        }
        return x
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

private extension FX7 {
    static let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct FXBTMetalJob {
        ulong combinationIndex;
        uint parameterOffset;
        uint parameterCount;
    };

    struct FXBTMetalResult {
        ulong combinationIndex;
        float netProfit;
        float grossProfit;
        float grossLoss;
        float maxDrawdown;
        uint totalTrades;
        uint winningTrades;
        uint losingTrades;
        float winRate;
        float profitFactor;
        uint barsProcessed;
        uint flags;
    };

    struct FXBTMetalRunConfig {
        float initialDeposit;
        float contractLots;
        float priceScale;
        uint digits;
        float contractSize;
        float lotSize;
    };

    inline float fx7_clip(float x, float lo, float hi) {
        return min(max(x, lo), hi);
    }

    inline int fx7_sign(float x) {
        return x > 0.0 ? 1 : (x < 0.0 ? -1 : 0);
    }

    inline float fx7_sigmoid(float x) {
        if (x >= 0.0) {
            float e = exp(-x);
            return 1.0 / (1.0 + e);
        }
        float e = exp(x);
        return e / (1.0 + e);
    }

    inline bool fx7_is_complete_signal_execution(const device long *utc, uint index, int signalStride) {
        if (index == 0 || index < uint(signalStride)) {
            return false;
        }
        long bucketSeconds = long(signalStride) * 60;
        long currentBucket = utc[index] / bucketSeconds;
        long previousBucket = utc[index - 1] / bucketSeconds;
        if (currentBucket == previousBucket) {
            return false;
        }
        uint previousStart = index - uint(signalStride);
        long expectedStart = utc[index - 1] - long(signalStride - 1) * 60;
        long previousStartBucket = utc[previousStart] / bucketSeconds;
        return utc[previousStart] == expectedStart && previousStartBucket == previousBucket;
    }

    inline float fx7_signal_return(const device long *close, uint decision, int signalStride, int lag) {
        uint newerIndex = decision - uint(lag * signalStride);
        uint olderIndex = decision - uint((lag + 1) * signalStride);
        float newer = max(float(close[newerIndex]), 1.0e-12);
        float older = max(float(close[olderIndex]), 1.0e-12);
        return log(newer / older);
    }

    inline float fx7_signal_ewma_std(const device long *close, uint decision, int signalStride, int count, int halfLife) {
        float lambda = exp(-log(2.0) / max(1.0, float(halfLife)));
        float variance = 0.0;
        bool seeded = false;
        for (int lag = count - 1; lag >= 0; --lag) {
            float r = fx7_signal_return(close, decision, signalStride, lag);
            variance = seeded ? lambda * variance + (1.0 - lambda) * r * r : r * r;
            seeded = true;
        }
        return sqrt(max(variance, 1.0e-12));
    }

    inline float fx7_signal_ewma_stdret(const device long *close, uint decision, int signalStride, int count, int halfLife, float sigmaLong) {
        float lambda = exp(-log(2.0) / max(1.0, float(halfLife)));
        float variance = 0.0;
        bool seeded = false;
        for (int lag = count - 1; lag >= 0; --lag) {
            float r = fx7_signal_return(close, decision, signalStride, lag) / max(sigmaLong, 1.0e-12);
            variance = seeded ? lambda * variance + (1.0 - lambda) * r * r : r * r;
            seeded = true;
        }
        return sqrt(max(variance, 1.0e-12));
    }

    inline float fx7_signal_high(const device long *high, uint signalEnd, int signalStride) {
        float value = -1.0e30;
        for (int offset = 0; offset < signalStride; ++offset) {
            value = max(value, float(high[signalEnd - uint(offset)]));
        }
        return value;
    }

    inline float fx7_signal_low(const device long *low, uint signalEnd, int signalStride) {
        float value = 1.0e30;
        for (int offset = 0; offset < signalStride; ++offset) {
            value = min(value, float(low[signalEnd - uint(offset)]));
        }
        return value;
    }

    inline float fx7_signal_atr_pct(const device long *high, const device long *low, const device long *close, uint decision, int signalStride, int window) {
        float sum = 0.0;
        for (int lag = window - 1; lag >= 0; --lag) {
            uint signalEnd = decision - uint(lag * signalStride);
            uint previousSignalEnd = decision - uint((lag + 1) * signalStride);
            float h = fx7_signal_high(high, signalEnd, signalStride);
            float l = fx7_signal_low(low, signalEnd, signalStride);
            float pc = float(close[previousSignalEnd]);
            sum += max(h - l, max(abs(h - pc), abs(l - pc)));
        }
        return max((sum / float(window)) / max(float(close[decision]), 1.0e-12), 1.0e-12);
    }

    kernel void fx7_core_v1(
        const device long *utc [[buffer(0)]],
        const device long *open [[buffer(1)]],
        const device long *high [[buffer(2)]],
        const device long *low [[buffer(3)]],
        const device long *close [[buffer(4)]],
        constant uint &barCount [[buffer(5)]],
        const device FXBTMetalJob *jobs [[buffer(6)]],
        const device float *parameters [[buffer(7)]],
        device FXBTMetalResult *results [[buffer(8)]],
        constant FXBTMetalRunConfig &config [[buffer(9)]],
        uint id [[thread_position_in_grid]]
    ) {
        FXBTMetalJob job = jobs[id];
        uint offset = job.parameterOffset;
        int signalStride = max(1, int(parameters[offset + 0] + 0.5));
        int h1 = max(1, int(parameters[offset + 1] + 0.5));
        int h2 = max(h1 + 1, int(parameters[offset + 2] + 0.5));
        int h3 = max(h2 + 1, int(parameters[offset + 3] + 0.5));
        int volShort = max(1, int(parameters[offset + 4] + 0.5));
        int volLong = max(volShort, int(parameters[offset + 5] + 0.5));
        int atrWindow = max(2, int(parameters[offset + 6] + 0.5));
        float entryThreshold = max(0.0, parameters[offset + 7]);
        float exitThreshold = min(entryThreshold, max(0.0, parameters[offset + 8]));
        float reversalThreshold = max(0.0, parameters[offset + 9]);
        float theta0 = max(0.0, parameters[offset + 10]);
        float confSlope = max(1.0e-6, parameters[offset + 11]);
        float alpha = fx7_clip(parameters[offset + 12], 0.0, 1.0);
        float tanhScale = max(1.0e-6, parameters[offset + 13]);
        float gammaA = max(0.0, parameters[offset + 14]);
        float gammaER = max(0.0, parameters[offset + 15]);
        float gammaV = max(0.0, parameters[offset + 16]);
        float gammaD = max(0.0, parameters[offset + 17]);
        float v0 = max(1.0e-6, parameters[offset + 18]);
        float gammaB = max(1.0e-6, parameters[offset + 19]);
        float gammaP = max(0.0, parameters[offset + 20]);
        float vPanic = max(1.0e-6, parameters[offset + 21]);
        float etaCost = max(0.0, parameters[offset + 22]);
        float etaVol = max(0.0, parameters[offset + 23]);
        float etaBreakout = max(0.0, parameters[offset + 24]);
        float gammaCost = max(0.0, parameters[offset + 25]);
        float roundTripCostPct = max(0.0, parameters[offset + 26]);
        float minConfidence = fx7_clip(parameters[offset + 27], 0.0, 1.0);
        float minRegimeGate = fx7_clip(parameters[offset + 28], 0.0, 1.0);
        float hardMinRegimeGate = fx7_clip(parameters[offset + 29], 0.0, 1.0);
        float minExecGate = fx7_clip(parameters[offset + 30], 0.0, 1.0);
        int persistenceBars = max(1, int(parameters[offset + 31] + 0.5));
        float referenceLots = max(0.01, parameters[offset + 34]);
        float riskPct = max(1.0e-6, parameters[offset + 35]);
        float maxPortfolioRiskPct = max(riskPct, parameters[offset + 36]);
        float stopATR = max(1.0e-6, parameters[offset + 37]);
        float singlePositionTakeProfit = max(0.0, parameters[offset + 38]);
        float sessionResetProfit = max(0.0, parameters[offset + 39]);
        bool trailingEnabled = parameters[offset + 40] >= 0.5;
        float trailStartPct = fx7_clip(parameters[offset + 41], 10.0, 100.0);
        float trailSpacingPct = fx7_clip(parameters[offset + 42], 10.0, 100.0);
        bool allowLong = parameters[offset + 52] >= 0.5;
        bool allowShort = parameters[offset + 53] >= 0.5;
        float w1Raw = max(0.0, parameters[offset + 54]);
        float w2Raw = max(0.0, parameters[offset + 55]);
        float w3Raw = max(0.0, parameters[offset + 56]);
        float wSum = w1Raw + w2Raw + w3Raw;
        float w1 = wSum > 1.0e-12 ? w1Raw / wSum : 0.45;
        float w2 = wSum > 1.0e-12 ? w2Raw / wSum : 0.35;
        float w3 = wSum > 1.0e-12 ? w3Raw / wSum : 0.20;
        int erWindow = max(2, int(parameters[offset + 57] + 0.5));
        int breakoutWindow = max(2, int(parameters[offset + 58] + 0.5));
        int shortReversalWindow = max(2, int(parameters[offset + 59] + 0.5));

        FXBTMetalResult result;
        result.combinationIndex = job.combinationIndex;
        result.netProfit = 0.0;
        result.grossProfit = 0.0;
        result.grossLoss = 0.0;
        result.maxDrawdown = 0.0;
        result.totalTrades = 0;
        result.winningTrades = 0;
        result.losingTrades = 0;
        result.winRate = 0.0;
        result.profitFactor = 0.0;
        result.barsProcessed = barCount;
        result.flags = 0;

        int retLen = max(max(int(parameters[offset + 43] + 0.5), volLong + 10), 110);
        int trendMax = max(h3, max(erWindow, max(breakoutWindow, shortReversalWindow)));
        int requiredSignals = max(trendMax + 3, retLen + 3) + 100;
        if (barCount <= uint((requiredSignals + 1) * signalStride)) {
            result.flags = 1;
            results[id] = result;
            return;
        }

        float balance = config.initialDeposit;
        float equity = config.initialDeposit;
        float peak = config.initialDeposit;
        float maxDrawdown = 0.0;
        float grossProfit = 0.0;
        float grossLoss = 0.0;
        int wins = 0;
        int losses = 0;
        int trades = 0;
        int position = 0;
        long entry = 0;
        float positionLots = 0.0;
        float openRiskCash = 0.0;
        float sessionStartEquity = config.initialDeposit;
        float peakProfit = 0.0;
        float smoothed = 0.0;
        int lastDirection = 0;
        int persistence = 0;
        int completedSignals = 0;

        for (uint index = 1; index < barCount; ++index) {
            long price = close[index];
            if (position != 0) {
                float floating = float(position) * (float(price - entry) / config.priceScale) * config.contractSize * positionLots;
                equity = balance + floating;
                peak = max(peak, equity);
                maxDrawdown = max(maxDrawdown, peak - equity);
                peakProfit = max(peakProfit, floating);
                bool closeForOverlay = false;
                if (trailingEnabled && singlePositionTakeProfit > 0.0) {
                    float startProfit = singlePositionTakeProfit * trailStartPct / 100.0;
                    float giveback = trailSpacingPct / 100.0;
                    closeForOverlay = peakProfit >= startProfit && floating <= peakProfit * (1.0 - giveback);
                } else if (singlePositionTakeProfit > 0.0) {
                    closeForOverlay = floating >= singlePositionTakeProfit;
                }
                if (!closeForOverlay && sessionResetProfit > 0.0 && equity - sessionStartEquity >= sessionResetProfit) {
                    closeForOverlay = true;
                    sessionStartEquity = equity;
                }
                if (closeForOverlay) {
                    balance += floating;
                    grossProfit += max(0.0, floating);
                    grossLoss += min(0.0, floating);
                    trades += 1;
                    if (floating >= 0.0) { wins += 1; } else { losses += 1; }
                    position = 0;
                    positionLots = 0.0;
                    openRiskCash = 0.0;
                    peakProfit = 0.0;
                    equity = balance;
                    peak = max(peak, equity);
                }
            } else {
                equity = balance;
                peak = max(peak, equity);
            }

            if (!fx7_is_complete_signal_execution(utc, index, signalStride)) {
                continue;
            }
            completedSignals += 1;
            if (completedSignals < requiredSignals) {
                continue;
            }

            uint decision = index - 1;
            int retCount = min(retLen, completedSignals - 1);
            float sigmaShort = fx7_signal_ewma_std(close, decision, signalStride, retCount, volShort);
            float sigmaLong = fx7_signal_ewma_std(close, decision, signalStride, retCount, volLong);
            float c = max(float(close[decision]), 1.0e-12);
            float z1 = fx7_clip(log(c / max(float(close[decision - uint(h1 * signalStride)]), 1.0e-12)) / (sigmaLong * sqrt(float(h1)) + 1.0e-12), -6.0, 6.0);
            float z2 = fx7_clip(log(c / max(float(close[decision - uint(h2 * signalStride)]), 1.0e-12)) / (sigmaLong * sqrt(float(h2)) + 1.0e-12), -6.0, 6.0);
            float z3 = fx7_clip(log(c / max(float(close[decision - uint(h3 * signalStride)]), 1.0e-12)) / (sigmaLong * sqrt(float(h3)) + 1.0e-12), -6.0, 6.0);
            float m = w1 * tanh(z1 / tanhScale) + w2 * tanh(z2 / tanhScale) + w3 * tanh(z3 / tanhScale);
            float a = abs(w1 * float(fx7_sign(z1)) + w2 * float(fx7_sign(z2)) + w3 * float(fx7_sign(z3)));
            float path = 0.0;
            for (int lb = 0; lb < erWindow; ++lb) {
                path += abs(fx7_signal_return(close, decision, signalStride, lb));
            }
            float er = abs(log(c / max(float(close[decision - uint(erWindow * signalStride)]), 1.0e-12))) / (path + 1.0e-12);
            float v = sigmaShort / (sigmaLong + 1.0e-12);
            float zrev = fx7_clip(log(c / max(float(close[decision - uint(shortReversalWindow * signalStride)]), 1.0e-12)) / (sigmaLong * sqrt(float(shortReversalWindow)) + 1.0e-12), -6.0, 6.0);
            float d = max(0.0, -float(fx7_sign(m)) * zrev);
            float highest = -1.0e30;
            float lowest = 1.0e30;
            for (int lb = 1; lb <= breakoutWindow; ++lb) {
                float value = float(close[decision - uint(lb * signalStride)]);
                highest = max(highest, value);
                lowest = min(lowest, value);
            }
            float mid = 0.5 * (highest + lowest);
            float halfRange = 0.5 * max(highest - lowest, 1.0e-12);
            float breakout = 0.5 * (1.0 + tanh(float(fx7_sign(m)) * (c - mid) / halfRange));
            float gate = pow(max(a, 0.0), gammaA) * pow(max(er, 0.0), gammaER)
                * exp(-gammaV * max(0.0, v - v0))
                * exp(-gammaD * d * max(0.0, v - v0));
            int panicCount = min(5, retCount);
            float zu5 = 0.0;
            for (int lag = 0; lag < panicCount; ++lag) {
                zu5 += fx7_signal_return(close, decision, signalStride, lag) / max(sigmaLong, 1.0e-12);
            }
            zu5 = panicCount > 0 ? zu5 / sqrt(float(panicCount)) : 0.0;
            float su = fx7_signal_ewma_stdret(close, decision, signalStride, min(20, retCount), 20, sigmaLong);
            float lu = fx7_signal_ewma_stdret(close, decision, signalStride, min(100, retCount), 100, sigmaLong);
            float vu = su / (lu + 1.0e-12);
            float panicGate = exp(-gammaP * max(0.0, vu - vPanic) * max(0.0, -float(fx7_sign(m)) * zu5));
            float score = m * panicGate * (0.5 + 0.5 * pow(fx7_clip(breakout, 0.0, 1.0), gammaB));
            smoothed = alpha * score + (1.0 - alpha) * smoothed;
            float confidence = fx7_sigmoid(confSlope * (max(abs(smoothed), abs(score)) - theta0));
            float atrPct = fx7_signal_atr_pct(high, low, close, decision, signalStride, atrWindow);
            float cost = roundTripCostPct / (atrPct + 1.0e-12);
            float execGate = exp(-gammaCost * cost);
            float longThreshold = max(entryThreshold + 0.20 * etaCost * max(0.0, cost - 1.0) + 0.20 * etaVol * max(0.0, v - 1.0) + 0.10 * etaBreakout * (1.0 - fx7_clip(breakout, 0.0, 1.0)), entryThreshold);
            float shortThreshold = longThreshold;
            int direction = 0;
            if (allowLong && smoothed >= longThreshold) {
                direction = 1;
            } else if (allowShort && smoothed <= -shortThreshold) {
                direction = -1;
            } else if (allowLong && fx7_sign(m) > 0 && smoothed >= 0.60 * longThreshold && score > 0.0) {
                direction = 1;
            } else if (allowShort && fx7_sign(m) < 0 && smoothed <= -0.60 * shortThreshold && score < 0.0) {
                direction = -1;
            }
            if (direction == lastDirection && direction != 0) {
                persistence += 1;
            } else {
                lastDirection = direction;
                persistence = direction == 0 ? 0 : 1;
            }
            bool gateOK = confidence >= minConfidence && gate >= minRegimeGate && execGate >= minExecGate && persistence >= persistenceBars;

            if (position != 0) {
                bool exitLong = position > 0 && smoothed <= exitThreshold;
                bool exitShort = position < 0 && smoothed >= -exitThreshold;
                bool gateExit = gate < hardMinRegimeGate || execGate < minExecGate;
                bool reversalExit = gateOK && direction != 0 && direction != position && abs(smoothed) >= reversalThreshold;
                if (exitLong || exitShort || gateExit || reversalExit) {
                    float pnl = float(position) * (float(price - entry) / config.priceScale) * config.contractSize * positionLots;
                    balance += pnl;
                    grossProfit += max(0.0, pnl);
                    grossLoss += min(0.0, pnl);
                    trades += 1;
                    if (pnl >= 0.0) { wins += 1; } else { losses += 1; }
                    position = 0;
                    positionLots = 0.0;
                    openRiskCash = 0.0;
                    peakProfit = 0.0;
                }
            }
            if (position == 0 && gateOK && direction != 0) {
                float rawPrice = float(price) / config.priceScale;
                float riskPerLot = rawPrice * atrPct * stopATR * config.contractSize;
                float perTradeRoom = riskPerLot <= 0.0 ? referenceLots : (equity * riskPct / 100.0) / riskPerLot;
                float portfolioRoom = riskPerLot <= 0.0 ? referenceLots : max(0.0, equity * maxPortfolioRiskPct / 100.0 - openRiskCash) / riskPerLot;
                float riskLots = min(referenceLots, min(perTradeRoom, portfolioRoom));
                float normalizedLots = floor(riskLots * 100.0) / 100.0;
                if (normalizedLots > 0.0) {
                    position = direction;
                    entry = price;
                    positionLots = normalizedLots;
                    openRiskCash = riskPerLot * normalizedLots;
                    peakProfit = 0.0;
                }
            }
            float floating = position == 0 ? 0.0 : float(position) * (float(price - entry) / config.priceScale) * config.contractSize * positionLots;
            equity = balance + floating;
            peak = max(peak, equity);
            maxDrawdown = max(maxDrawdown, peak - equity);
        }

        if (position != 0) {
            long finalPrice = close[barCount - 1];
            float pnl = float(position) * (float(finalPrice - entry) / config.priceScale) * config.contractSize * positionLots;
            balance += pnl;
            grossProfit += max(0.0, pnl);
            grossLoss += min(0.0, pnl);
            trades += 1;
            if (pnl >= 0.0) { wins += 1; } else { losses += 1; }
        }

        result.netProfit = balance - config.initialDeposit;
        result.grossProfit = grossProfit;
        result.grossLoss = grossLoss;
        result.maxDrawdown = maxDrawdown;
        result.totalTrades = uint(trades);
        result.winningTrades = uint(wins);
        result.losingTrades = uint(losses);
        result.winRate = trades == 0 ? 0.0 : float(wins) / float(trades);
        result.profitFactor = grossLoss == 0.0 ? (grossProfit > 0.0 ? 999999.0 : 0.0) : grossProfit / abs(grossLoss);
        results[id] = result;
    }
    """
}
