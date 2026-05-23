import Foundation

public struct FeatureRegistry: Sendable {
    public init() {}

    public func group(for featureIndex: Int) -> FeatureGroup {
        if featureIndex < 0 || featureIndex >= FXDataEngineConstants.aiFeatures {
            return .price
        }
        if featureIndex <= 5 { return .price }
        if featureIndex == 6 { return .volume }
        if featureIndex <= 9 { return .multiTimeframe }
        if featureIndex <= 12 { return .context }
        if featureIndex <= 14 { return .multiTimeframe }
        if featureIndex <= 17 { return .time }
        if featureIndex <= 21 { return .price }
        if featureIndex <= 37 { return .multiTimeframe }
        if featureIndex <= 45 { return .volatility }
        if featureIndex <= 49 { return .filters }
        if featureIndex <= 65 { return .context }
        if featureIndex <= 71 { return .microstructure }
        if featureIndex <= 73 { return .time }
        if featureIndex <= 78 { return .volume }
        if featureIndex == 79 { return .filters }
        if featureIndex <= 83 { return .volume }
        if featureIndex < FXDataEngineConstants.contextMTFFeatureOffset { return .multiTimeframe }
        if featureIndex < FXDataEngineConstants.macroEventFeatureOffset { return .context }

        let macroRelative = featureIndex - FXDataEngineConstants.macroEventFeatureOffset
        if macroRelative <= 2 { return .time }
        if macroRelative == 8 { return .context }
        return .filters
    }

    public func provenance(for featureIndex: Int) -> FeatureProvenance {
        if featureIndex < 0 || featureIndex >= FXDataEngineConstants.aiFeatures {
            return .derivedFilter
        }
        if featureIndex >= FXDataEngineConstants.mainMTFFeatureOffset,
           featureIndex < FXDataEngineConstants.contextMTFFeatureOffset {
            return .multiTimeframe
        }
        if featureIndex >= FXDataEngineConstants.contextMTFFeatureOffset,
           featureIndex < FXDataEngineConstants.macroEventFeatureOffset {
            return .contextSymbol
        }
        if featureIndex >= FXDataEngineConstants.macroEventFeatureOffset {
            return .eventMacro
        }
        if featureIndex <= 6 ||
            (18...21).contains(featureIndex) ||
            (66...71).contains(featureIndex) ||
            (74...78).contains(featureIndex) ||
            (80...83).contains(featureIndex) {
            return .priceBar
        }
        if (7...9).contains(featureIndex) || (13...37).contains(featureIndex) {
            return .multiTimeframe
        }
        if (10...12).contains(featureIndex) || (50...65).contains(featureIndex) {
            return .contextSymbol
        }
        if (15...17).contains(featureIndex) || featureIndex == 72 || featureIndex == 73 {
            return .timeCalendar
        }
        return .derivedFilter
    }

    public func name(for featureIndex: Int) -> String {
        switch featureIndex {
        case 0: "m1_ret_1"
        case 1: "m1_ret_3"
        case 2: "m1_ret_5"
        case 3: "m1_slope_10"
        case 4: "m1_zscore_10"
        case 5: "m1_return_vol_10"
        case 6: "volume_norm"
        case 7: "m5_ret"
        case 8: "m15_ret"
        case 9: "h1_ret"
        case 10: "ctx_ret_mean"
        case 11: "ctx_ret_std"
        case 12: "ctx_up_ratio"
        case 13: "m5_slope"
        case 14: "h1_slope"
        case 15: "weekday_norm"
        case 16: "hour_norm"
        case 17: "minute_norm"
        case 18: "body_edge"
        case 19: "upper_wick_edge"
        case 20: "lower_wick_edge"
        case 21: "bar_range_norm"
        case 22: "m5_sma100_edge"
        case 23: "m5_sma200_edge"
        case 24: "m15_sma100_edge"
        case 25: "m15_sma200_edge"
        case 26: "m30_sma100_edge"
        case 27: "m30_sma200_edge"
        case 28: "h1_sma100_edge"
        case 29: "h1_sma200_edge"
        case 30: "m5_ema100_edge"
        case 31: "m5_ema200_edge"
        case 32: "m15_ema100_edge"
        case 33: "m15_ema200_edge"
        case 34: "m30_ema100_edge"
        case 35: "m30_ema200_edge"
        case 36: "h1_ema100_edge"
        case 37: "h1_ema200_edge"
        case 38: "qsdema100_edge"
        case 39: "qsdema200_edge"
        case 40: "rsi14"
        case 41: "atr14_unit"
        case 42: "natr14"
        case 43: "parkinson20"
        case 44: "rogers_satchell20"
        case 45: "garman_klass20"
        case 46: "median21_edge"
        case 47: "hampel21"
        case 48: "kalman34_edge"
        case 49: "supersmoother20_edge"
        case 50: "ctx_top1_ret"
        case 51: "ctx_top1_lag"
        case 52: "ctx_top1_rel"
        case 53: "ctx_top1_corr"
        case 54: "ctx_top2_ret"
        case 55: "ctx_top2_lag"
        case 56: "ctx_top2_rel"
        case 57: "ctx_top2_corr"
        case 58: "ctx_top3_ret"
        case 59: "ctx_top3_lag"
        case 60: "ctx_top3_rel"
        case 61: "ctx_top3_corr"
        case 62: "shared_util"
        case 63: "shared_stability"
        case 64: "shared_lead"
        case 65: "shared_coverage"
        case 66: "close_location"
        case 67: "wick_imbalance"
        case 68: "volume_shock"
        case 69: "volume_accel"
        case 70: "volume_to_range"
        case 71: "volume_trend"
        case 72: "session_transition"
        case 73: "session_overlap"
        case 74: "volume_session_activity"
        case 75: "volume_available_flag"
        case 76: "volume_rank_50"
        case 77: "volume_price_alignment"
        case 78: "volume_persistence"
        case 79: "feature_family_drift"
        case 80: "volume_log"
        case 81: "volume_zscore_20"
        case 82: "volume_vol_ratio_20"
        case 83: "volume_rank_20"
        case FXDataEngineConstants.macroEventFeatureOffset + 0: "macro_pre_event_embargo"
        case FXDataEngineConstants.macroEventFeatureOffset + 1: "macro_post_event_embargo"
        case FXDataEngineConstants.macroEventFeatureOffset + 2: "macro_event_importance"
        case FXDataEngineConstants.macroEventFeatureOffset + 3: "macro_surprise_signed"
        case FXDataEngineConstants.macroEventFeatureOffset + 4: "macro_surprise_abs"
        case FXDataEngineConstants.macroEventFeatureOffset + 5: "macro_event_class_bias"
        case FXDataEngineConstants.macroEventFeatureOffset + 6: "macro_surprise_zscore"
        case FXDataEngineConstants.macroEventFeatureOffset + 7: "macro_revision_abs"
        case FXDataEngineConstants.macroEventFeatureOffset + 8: "macro_currency_relevance"
        case FXDataEngineConstants.macroEventFeatureOffset + 9: "macro_provenance_trust"
        case FXDataEngineConstants.macroEventFeatureOffset + 10: "macro_rates_activity"
        case FXDataEngineConstants.macroEventFeatureOffset + 11: "macro_inflation_activity"
        case FXDataEngineConstants.macroEventFeatureOffset + 12: "macro_labor_activity"
        case FXDataEngineConstants.macroEventFeatureOffset + 13: "macro_growth_activity"
        case FXDataEngineConstants.macroEventFeatureOffset + 14: "macro_policy_divergence"
        case FXDataEngineConstants.macroEventFeatureOffset + 15: "macro_policy_pressure"
        case FXDataEngineConstants.macroEventFeatureOffset + 16: "macro_inflation_pressure"
        case FXDataEngineConstants.macroEventFeatureOffset + 17: "macro_labor_pressure"
        case FXDataEngineConstants.macroEventFeatureOffset + 18: "macro_growth_pressure"
        case FXDataEngineConstants.macroEventFeatureOffset + 19: "macro_state_quality"
        default:
            dynamicName(for: featureIndex)
        }
    }

    private func dynamicName(for featureIndex: Int) -> String {
        if featureIndex >= FXDataEngineConstants.mainMTFFeatureOffset,
           featureIndex < FXDataEngineConstants.contextMTFFeatureOffset {
            let relative = featureIndex - FXDataEngineConstants.mainMTFFeatureOffset
            let timeframeSlot = relative / FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            let metric = relative % FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            return "\(Self.mainMTFSlotName(timeframeSlot))_\(Self.mtfMetricName(metric))"
        }
        if featureIndex >= FXDataEngineConstants.contextMTFFeatureOffset,
           featureIndex < FXDataEngineConstants.macroEventFeatureOffset {
            let relative = featureIndex - FXDataEngineConstants.contextMTFFeatureOffset
            let slot = relative / FXDataEngineConstants.contextSlotMTFFeatures
            let slotRelative = relative % FXDataEngineConstants.contextSlotMTFFeatures
            let timeframeSlot = slotRelative / FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            let metric = slotRelative % FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            return "ctx_top\(slot + 1)_\(Self.contextMTFSlotName(timeframeSlot))_\(Self.mtfMetricName(metric))"
        }
        return ""
    }

    private static func mtfMetricName(_ metric: Int) -> String {
        switch metric {
        case 0: "body_bias"
        case 1: "close_location"
        case 2: "range_pressure"
        case 3: "volume_pressure"
        default: "unknown"
        }
    }

    private static func mainMTFSlotName(_ slot: Int) -> String {
        switch slot {
        case 0: "m5"
        case 1: "m15"
        case 2: "m30"
        case 3: "h1"
        default: "m1"
        }
    }

    private static func contextMTFSlotName(_ slot: Int) -> String {
        switch slot {
        case 0: "m1"
        case 1: "m5"
        case 2: "m15"
        case 3: "m30"
        case 4: "h1"
        default: "m1"
        }
    }
}

public struct FeatureSchemaPolicy: Sendable {
    public let registry: FeatureRegistry

    public init(registry: FeatureRegistry = FeatureRegistry()) {
        self.registry = registry
    }

    public func defaultGroups(for family: AIFamily) -> FeatureGroupMask {
        switch family {
        case .linear, .distributional:
            [.price, .multiTimeframe, .volatility, .time, .context, .volume, .microstructure]
        case .tree:
            .all
        case .ruleBased:
            [.price, .time, .volume]
        default:
            .all
        }
    }

    public func defaultSchema(for family: AIFamily) -> FeatureSchema {
        switch family {
        case .linear, .distributional:
            .sparseStat
        case .ruleBased:
            .rule
        case .tree:
            .tree
        case .recurrent, .convolutional, .transformer, .stateSpace:
            .sequence
        case .retrieval, .mixture, .worldModel:
            .contextual
        default:
            .full
        }
    }

    public func isFeatureEnabled(featureIndex: Int, schema: FeatureSchema, groups: FeatureGroupMask) -> Bool {
        if featureIndex >= FXDataEngineConstants.mainMTFFeatureOffset,
           featureIndex < FXDataEngineConstants.macroEventFeatureOffset {
            return true
        }

        let group = registry.group(for: featureIndex)
        let bit = FeatureGroupMask(group: group)
        guard groups.contains(bit) else { return false }

        switch schema {
        case .sparseStat:
            if (46...49).contains(featureIndex) { return false }
            if (50...71).contains(featureIndex) { return false }
            return true
        case .rule:
            return group == .price || group == .time || group == .volume
        case .contextual:
            if group == .time, (15...17).contains(featureIndex) { return false }
            return true
        case .tree, .sequence, .full:
            return true
        }
    }

    public func modelInput(from features: [Double]) -> [Double] {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[0] = 1.0
        let n = min(features.count, FXDataEngineConstants.aiFeatures)
        for index in 0..<n {
            x[index + 1] = fxSafeFinite(features[index])
        }
        return x
    }

    public func feature(_ x: [Double], at featureIndex: Int) -> Double {
        let index = featureIndex + 1
        guard index >= 0, index < x.count else { return 0.0 }
        return x[index]
    }

    public func apply(schema: FeatureSchema, groups: FeatureGroupMask, to input: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        output[0] = 1.0
        let sourceCount = min(input.count, FXDataEngineConstants.aiWeights)
        for index in 1..<sourceCount {
            let featureIndex = index - 1
            output[index] = isFeatureEnabled(featureIndex: featureIndex, schema: schema, groups: groups)
                ? fxSafeFinite(input[index])
                : 0.0
        }
        return output
    }

    public func apply(schema: FeatureSchema, groups: FeatureGroupMask, toWindow window: [[Double]]) -> [[Double]] {
        window.map { apply(schema: schema, groups: groups, to: $0) }
    }
}
