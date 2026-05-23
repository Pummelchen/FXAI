import Foundation

public enum RuntimeArtifactConstants {
    public static let directory = "FXAI/Runtime"
    public static let version = 15
    public static let normalizationRollWindowMax = 512
    public static let replayCapacity = 384
    public static let maxHorizons = 8
    public static let conformalDepth = 96
    public static let reliabilityPendingCapacity = 2_048
}

public enum RuntimeArtifactPaths {
    public static func safeSymbol(_ symbol: String, defaultValue: String = "default") -> String {
        var clean = symbol.isEmpty ? defaultValue : symbol
        for character in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"] {
            clean = clean.replacingOccurrences(of: character, with: "_")
        }
        return clean
    }

    public static func runtimeArtifactFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_runtime_\(safeSymbol(symbol)).bin"
    }

    public static func persistenceManifestFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_persistence_\(safeSymbol(symbol)).tsv"
    }

    public static func featureManifestFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_features_\(safeSymbol(symbol)).tsv"
    }

    public static func macroManifestFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_macro_\(safeSymbol(symbol)).tsv"
    }

    public static func performanceManifestFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_perf_\(safeSymbol(symbol)).tsv"
    }

    public static func shadowLedgerFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_shadow_\(safeSymbol(symbol)).tsv"
    }
}

public enum RuntimeArtifactTSV {
    public static func bool(_ value: Bool) -> String {
        value ? "1" : "0"
    }

    public static func double(_ value: Double, decimals: Int = 6) -> String {
        String(format: "%.\(decimals)f", locale: Locale(identifier: "en_US_POSIX"), fxSafeFinite(value))
    }

    public static func field(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\t", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
    }

    public static func document(header: [String], rows: [[String]]) -> String {
        var lines = [header.map(field).joined(separator: "\t")]
        lines.append(contentsOf: rows.map { $0.map(field).joined(separator: "\t") })
        return lines.joined(separator: "\r\n") + "\r\n"
    }
}

public extension ReferenceTier {
    var runtimeArtifactName: String {
        switch self {
        case .fullNative: "full_native"
        case .compressedNative: "compressed_native"
        case .surrogate: "surrogate"
        case .ruleBaseline: "rule_baseline"
        }
    }
}

public struct RuntimeArtifactHeader: Codable, Hashable, Sendable {
    public static let binaryFieldCount = 10
    public static let binaryByteCount = binaryFieldCount * MemoryLayout<Int32>.size

    public var version: Int
    public var featureCount: Int
    public var normalizationMethodCount: Int
    public var normalizationRollWindowMax: Int
    public var replayCapacity: Int
    public var aiCount: Int
    public var regimeCount: Int
    public var maxHorizons: Int
    public var conformalDepth: Int
    public var reliabilityPendingCapacity: Int

    public init(
        version: Int = RuntimeArtifactConstants.version,
        featureCount: Int = FXDataEngineConstants.aiFeatures,
        normalizationMethodCount: Int = FXDataEngineConstants.normMethodCount,
        normalizationRollWindowMax: Int = RuntimeArtifactConstants.normalizationRollWindowMax,
        replayCapacity: Int = RuntimeArtifactConstants.replayCapacity,
        aiCount: Int = FXDataEngineConstants.aiCount,
        regimeCount: Int = FXDataEngineConstants.pluginRegimeBuckets,
        maxHorizons: Int = RuntimeArtifactConstants.maxHorizons,
        conformalDepth: Int = RuntimeArtifactConstants.conformalDepth,
        reliabilityPendingCapacity: Int = RuntimeArtifactConstants.reliabilityPendingCapacity
    ) {
        self.version = version
        self.featureCount = featureCount
        self.normalizationMethodCount = normalizationMethodCount
        self.normalizationRollWindowMax = normalizationRollWindowMax
        self.replayCapacity = replayCapacity
        self.aiCount = aiCount
        self.regimeCount = regimeCount
        self.maxHorizons = maxHorizons
        self.conformalDepth = conformalDepth
        self.reliabilityPendingCapacity = reliabilityPendingCapacity
    }

    public var isCompatibleWithCurrentContract: Bool {
        self == RuntimeArtifactHeader()
    }

    public var binaryFields: [Int] {
        [
            version,
            featureCount,
            normalizationMethodCount,
            normalizationRollWindowMax,
            replayCapacity,
            aiCount,
            regimeCount,
            maxHorizons,
            conformalDepth,
            reliabilityPendingCapacity
        ]
    }
}

public struct RuntimeArtifactEnvelope: Codable, Hashable, Sendable {
    public var header: RuntimeArtifactHeader
    public var payload: Data

    public init(header: RuntimeArtifactHeader = RuntimeArtifactHeader(), payload: Data = Data()) {
        self.header = header
        self.payload = payload
    }

    public var payloadByteCount: Int {
        payload.count
    }

    public var isCompatibleWithCurrentContract: Bool {
        header.isCompatibleWithCurrentContract
    }
}

public enum RuntimeArtifactBinaryCodec {
    public static func encodeHeader(_ header: RuntimeArtifactHeader) throws -> Data {
        var data = Data()
        data.reserveCapacity(RuntimeArtifactHeader.binaryByteCount)
        for value in header.binaryFields {
            try appendInt32(value, to: &data)
        }
        return data
    }

    public static func decodeHeader(from data: Data) throws -> RuntimeArtifactHeader {
        guard data.count >= RuntimeArtifactHeader.binaryByteCount else {
            throw FXDataEngineError.validation(
                "runtime artifact header is \(data.count) bytes, need \(RuntimeArtifactHeader.binaryByteCount)"
            )
        }
        let fields = try decodeHeaderFields(from: data)
        return RuntimeArtifactHeader(
            version: fields[0],
            featureCount: fields[1],
            normalizationMethodCount: fields[2],
            normalizationRollWindowMax: fields[3],
            replayCapacity: fields[4],
            aiCount: fields[5],
            regimeCount: fields[6],
            maxHorizons: fields[7],
            conformalDepth: fields[8],
            reliabilityPendingCapacity: fields[9]
        )
    }

    public static func encodeEnvelope(_ envelope: RuntimeArtifactEnvelope) throws -> Data {
        var data = try encodeHeader(envelope.header)
        data.append(envelope.payload)
        return data
    }

    public static func decodeEnvelope(from data: Data) throws -> RuntimeArtifactEnvelope {
        let header = try decodeHeader(from: data)
        let payloadStart = RuntimeArtifactHeader.binaryByteCount
        let payload = data.count > payloadStart ? data.subdata(in: payloadStart..<data.count) : Data()
        return RuntimeArtifactEnvelope(header: header, payload: payload)
    }

    private static func decodeHeaderFields(from data: Data) throws -> [Int] {
        let bytes = [UInt8](data.prefix(RuntimeArtifactHeader.binaryByteCount))
        var output: [Int] = []
        output.reserveCapacity(RuntimeArtifactHeader.binaryFieldCount)
        for offset in stride(from: 0, to: RuntimeArtifactHeader.binaryByteCount, by: MemoryLayout<Int32>.size) {
            output.append(Int(readInt32(bytes: bytes, offset: offset)))
        }
        return output
    }

    private static func appendInt32(_ value: Int, to data: inout Data) throws {
        guard value >= Int(Int32.min), value <= Int(Int32.max) else {
            throw FXDataEngineError.validation("runtime artifact integer field exceeds int32 range")
        }
        var raw = Int32(value).littleEndian
        withUnsafeBytes(of: &raw) {
            data.append(contentsOf: $0)
        }
    }

    private static func readInt32(bytes: [UInt8], offset: Int) -> Int32 {
        var raw = UInt32(bytes[offset])
        raw |= UInt32(bytes[offset + 1]) << 8
        raw |= UInt32(bytes[offset + 2]) << 16
        raw |= UInt32(bytes[offset + 3]) << 24
        return Int32(bitPattern: raw)
    }
}

public enum RuntimeArtifactSavePolicy {
    public static let defaultMinSaveIntervalSeconds: Int64 = 900

    public static func shouldSave(
        dirty: Bool,
        lastSaveTimeUTC: Int64,
        barTimeUTC: Int64,
        nowUTC: Int64,
        minIntervalSeconds: Int64 = defaultMinSaveIntervalSeconds
    ) -> Bool {
        guard dirty else { return false }
        let effectiveNow = barTimeUTC > 0 ? barTimeUTC : nowUTC
        guard effectiveNow > 0 else { return true }
        guard lastSaveTimeUTC > 0 else { return true }
        return (effectiveNow - lastSaveTimeUTC) >= minIntervalSeconds
    }
}

public struct FeatureClipBounds: Codable, Hashable, Sendable {
    public let lower: Double
    public let upper: Double

    public init(lower: Double, upper: Double) {
        self.lower = lower
        self.upper = upper
    }
}

public extension FeatureRegistry {
    func clipBounds(for featureIndex: Int) -> FeatureClipBounds {
        var lower = -8.0
        var upper = 8.0

        if featureIndex == 5 {
            lower = 0.0
            upper = 10.0
        } else if featureIndex == 6 {
            lower = 0.0
            upper = 12.0
        } else if featureIndex == 12 {
            lower = -1.0
            upper = 1.0
        } else if (15...17).contains(featureIndex) {
            lower = -1.2
            upper = 1.2
        } else if featureIndex == 40 {
            lower = -1.2
            upper = 1.2
        } else if (41...43).contains(featureIndex) {
            lower = 0.0
            upper = 40.0
        } else if (44...45).contains(featureIndex) {
            lower = 0.0
            upper = 40.0
        } else if featureIndex == 47 {
            lower = -12.0
            upper = 12.0
        } else if (62...67).contains(featureIndex) {
            lower = -1.2
            upper = 1.2
        } else if (68...69).contains(featureIndex) {
            lower = -8.0
            upper = 8.0
        } else if featureIndex == 70 {
            lower = 0.0
            upper = 8.0
        } else if featureIndex == 71 {
            lower = -6.0
            upper = 6.0
        } else if (72...75).contains(featureIndex) {
            lower = -1.2
            upper = 1.2
        } else if featureIndex == 76 || featureIndex == 77 {
            lower = -4.5
            upper = 4.5
        } else if featureIndex == 78 || featureIndex == 79 {
            lower = -6.0
            upper = 6.0
        } else if featureIndex == 80 {
            lower = 0.0
            upper = 6.0
        } else if featureIndex == 81 {
            lower = -8.0
            upper = 8.0
        } else if featureIndex == 82 {
            lower = 0.0
            upper = 4.5
        } else if featureIndex == 83 {
            lower = -1.1
            upper = 1.1
        } else if featureIndex >= FXDataEngineConstants.mainMTFFeatureOffset,
                  featureIndex < FXDataEngineConstants.macroEventFeatureOffset {
            let relative: Int
            if featureIndex >= FXDataEngineConstants.contextMTFFeatureOffset {
                relative = (featureIndex - FXDataEngineConstants.contextMTFFeatureOffset) %
                    FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            } else {
                relative = (featureIndex - FXDataEngineConstants.mainMTFFeatureOffset) %
                    FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            }
            if relative <= 1 {
                lower = -1.2
                upper = 1.2
            } else if relative == 2 {
                lower = -6.0
                upper = 6.0
            } else {
                lower = -6.0
                upper = 8.0
            }
        } else if featureIndex >= FXDataEngineConstants.macroEventFeatureOffset,
                  featureIndex < FXDataEngineConstants.aiFeatures {
            let relative = featureIndex - FXDataEngineConstants.macroEventFeatureOffset
            if relative <= 2 {
                lower = 0.0
                upper = 1.2
            } else if relative == 3 {
                lower = -6.0
                upper = 6.0
            } else if relative == 4 {
                lower = 0.0
                upper = 6.0
            } else {
                lower = -1.2
                upper = 1.2
            }
        } else if featureIndex >= 50, featureIndex < FXDataEngineConstants.aiFeatures {
            let relative = (featureIndex - 50) % 4
            if relative == 3 {
                lower = -1.1
                upper = 1.1
            }
        }

        return FeatureClipBounds(lower: lower, upper: upper)
    }

    func leakageGuarded(for featureIndex: Int) -> Bool {
        let provenance = provenance(for: featureIndex)
        return FeatureProvenance.allCases.contains(provenance)
    }
}

public struct PersistenceCoverageManifestRow: Codable, Hashable, Sendable {
    public static let header = [
        "ai_id",
        "ai_name",
        "reference_tier",
        "coverage_tag",
        "checkpoint_depth",
        "persistent",
        "state_version",
        "capability_mask",
        "stateful_checkpoint",
        "native_snapshot",
        "deterministic_replay",
        "native_required",
        "promotion_ready",
        "state_file_size",
        "state_file",
        "coverage_note"
    ]

    public var aiID: Int
    public var aiName: String
    public var referenceTier: ReferenceTier
    public var coverageTag: String
    public var checkpointDepth: String
    public var persistent: Bool
    public var stateVersion: Int
    public var capabilityMask: PluginCapability
    public var nativeSnapshot: Bool
    public var deterministicReplay: Bool
    public var stateFileSize: Int64
    public var stateFile: String

    public init(
        aiID: Int,
        aiName: String,
        referenceTier: ReferenceTier,
        coverageTag: String,
        checkpointDepth: String,
        persistent: Bool,
        stateVersion: Int,
        capabilityMask: PluginCapability,
        nativeSnapshot: Bool,
        deterministicReplay: Bool,
        stateFileSize: Int64,
        stateFile: String
    ) {
        self.aiID = aiID
        self.aiName = aiName
        self.referenceTier = referenceTier
        self.coverageTag = coverageTag
        self.checkpointDepth = checkpointDepth
        self.persistent = persistent
        self.stateVersion = stateVersion
        self.capabilityMask = capabilityMask
        self.nativeSnapshot = nativeSnapshot
        self.deterministicReplay = deterministicReplay
        self.stateFileSize = stateFileSize
        self.stateFile = stateFile
    }

    public var statefulCheckpoint: Bool {
        capabilityMask.contains(.onlineLearning) ||
            capabilityMask.contains(.replay) ||
            capabilityMask.contains(.stateful)
    }

    public var nativeRequired: Bool {
        statefulCheckpoint
    }

    public var promotionReady: Bool {
        !nativeRequired || (coverageTag == "native_model" && nativeSnapshot)
    }

    public var coverageNote: String {
        if nativeRequired && !promotionReady {
            "stateful model blocked from live promotion until native parameter snapshot coverage is implemented"
        } else if coverageTag == "native_model" {
            "native checkpoint verified"
        } else if coverageTag == "native_replay" {
            "deterministic replay checkpoint available for audit and research recovery only"
        } else {
            "checkpoint not required"
        }
    }

    public var tsvColumns: [String] {
        [
            String(aiID),
            aiName,
            referenceTier.runtimeArtifactName,
            coverageTag,
            checkpointDepth,
            RuntimeArtifactTSV.bool(persistent),
            String(stateVersion),
            String(capabilityMask.rawValue),
            RuntimeArtifactTSV.bool(statefulCheckpoint),
            RuntimeArtifactTSV.bool(nativeSnapshot),
            RuntimeArtifactTSV.bool(deterministicReplay),
            RuntimeArtifactTSV.bool(nativeRequired),
            RuntimeArtifactTSV.bool(promotionReady),
            String(stateFileSize),
            stateFile,
            coverageNote
        ]
    }
}

public struct FeatureRegistryManifestRow: Codable, Hashable, Sendable {
    public static let header = [
        "feature_idx",
        "feature_name",
        "feature_group",
        "provenance",
        "leakage_guarded",
        "clip_lo",
        "clip_hi"
    ]

    public var featureIndex: Int
    public var featureName: String
    public var featureGroup: String
    public var provenance: String
    public var leakageGuarded: Bool
    public var clipLower: Double
    public var clipUpper: Double

    public init(featureIndex: Int, registry: FeatureRegistry = FeatureRegistry()) {
        let bounds = registry.clipBounds(for: featureIndex)
        self.featureIndex = featureIndex
        self.featureName = registry.name(for: featureIndex)
        self.featureGroup = registry.group(for: featureIndex).name
        self.provenance = registry.provenance(for: featureIndex).rawValue
        self.leakageGuarded = registry.leakageGuarded(for: featureIndex)
        self.clipLower = bounds.lower
        self.clipUpper = bounds.upper
    }

    public var tsvColumns: [String] {
        [
            String(featureIndex),
            featureName,
            featureGroup,
            provenance,
            RuntimeArtifactTSV.bool(leakageGuarded),
            RuntimeArtifactTSV.double(clipLower),
            RuntimeArtifactTSV.double(clipUpper)
        ]
    }
}

public struct MacroDatasetManifestRow: Codable, Hashable, Sendable {
    public static let header = [
        "symbol",
        "schema_version",
        "record_count",
        "parse_errors",
        "distinct_symbols",
        "distinct_sources",
        "distinct_event_ids",
        "distinct_countries",
        "distinct_currencies",
        "distinct_revision_chains",
        "family_rates_count",
        "family_inflation_count",
        "family_labor_count",
        "family_growth_count",
        "family_trade_count",
        "first_event_time",
        "last_event_time",
        "avg_importance",
        "avg_pre_window_min",
        "avg_post_window_min",
        "avg_surprise_z_abs",
        "avg_revision_abs",
        "avg_source_trust",
        "avg_currency_relevance",
        "checksum01",
        "provenance_hash01",
        "leakage_guard_score",
        "leakage_safe"
    ]

    public var symbol: String
    public var schemaVersion: Int
    public var recordCount: Int
    public var parseErrors: Int
    public var distinctSymbols: Int
    public var distinctSources: Int
    public var distinctEventIDs: Int
    public var distinctCountries: Int
    public var distinctCurrencies: Int
    public var distinctRevisionChains: Int
    public var familyRatesCount: Int
    public var familyInflationCount: Int
    public var familyLaborCount: Int
    public var familyGrowthCount: Int
    public var familyTradeCount: Int
    public var firstEventTimeUTC: Int64
    public var lastEventTimeUTC: Int64
    public var avgImportance: Double
    public var avgPreWindowMinutes: Double
    public var avgPostWindowMinutes: Double
    public var avgSurpriseZAbs: Double
    public var avgRevisionAbs: Double
    public var avgSourceTrust: Double
    public var avgCurrencyRelevance: Double
    public var checksum01: Double
    public var provenanceHash01: Double
    public var leakageGuardScore: Double
    public var leakageSafe: Bool

    public init(
        symbol: String,
        schemaVersion: Int = 0,
        recordCount: Int = 0,
        parseErrors: Int = 0,
        distinctSymbols: Int = 0,
        distinctSources: Int = 0,
        distinctEventIDs: Int = 0,
        distinctCountries: Int = 0,
        distinctCurrencies: Int = 0,
        distinctRevisionChains: Int = 0,
        familyRatesCount: Int = 0,
        familyInflationCount: Int = 0,
        familyLaborCount: Int = 0,
        familyGrowthCount: Int = 0,
        familyTradeCount: Int = 0,
        firstEventTimeUTC: Int64 = 0,
        lastEventTimeUTC: Int64 = 0,
        avgImportance: Double = 0.0,
        avgPreWindowMinutes: Double = 0.0,
        avgPostWindowMinutes: Double = 0.0,
        avgSurpriseZAbs: Double = 0.0,
        avgRevisionAbs: Double = 0.0,
        avgSourceTrust: Double = 0.0,
        avgCurrencyRelevance: Double = 0.0,
        checksum01: Double = 0.0,
        provenanceHash01: Double = 0.0,
        leakageGuardScore: Double = 0.0,
        leakageSafe: Bool = false
    ) {
        self.symbol = RuntimeArtifactPaths.safeSymbol(symbol)
        self.schemaVersion = schemaVersion
        self.recordCount = recordCount
        self.parseErrors = parseErrors
        self.distinctSymbols = distinctSymbols
        self.distinctSources = distinctSources
        self.distinctEventIDs = distinctEventIDs
        self.distinctCountries = distinctCountries
        self.distinctCurrencies = distinctCurrencies
        self.distinctRevisionChains = distinctRevisionChains
        self.familyRatesCount = familyRatesCount
        self.familyInflationCount = familyInflationCount
        self.familyLaborCount = familyLaborCount
        self.familyGrowthCount = familyGrowthCount
        self.familyTradeCount = familyTradeCount
        self.firstEventTimeUTC = firstEventTimeUTC
        self.lastEventTimeUTC = lastEventTimeUTC
        self.avgImportance = avgImportance
        self.avgPreWindowMinutes = avgPreWindowMinutes
        self.avgPostWindowMinutes = avgPostWindowMinutes
        self.avgSurpriseZAbs = avgSurpriseZAbs
        self.avgRevisionAbs = avgRevisionAbs
        self.avgSourceTrust = avgSourceTrust
        self.avgCurrencyRelevance = avgCurrencyRelevance
        self.checksum01 = checksum01
        self.provenanceHash01 = provenanceHash01
        self.leakageGuardScore = leakageGuardScore
        self.leakageSafe = leakageSafe
    }

    public var tsvColumns: [String] {
        [
            symbol,
            String(schemaVersion),
            String(recordCount),
            String(parseErrors),
            String(distinctSymbols),
            String(distinctSources),
            String(distinctEventIDs),
            String(distinctCountries),
            String(distinctCurrencies),
            String(distinctRevisionChains),
            String(familyRatesCount),
            String(familyInflationCount),
            String(familyLaborCount),
            String(familyGrowthCount),
            String(familyTradeCount),
            String(firstEventTimeUTC),
            String(lastEventTimeUTC),
            RuntimeArtifactTSV.double(avgImportance),
            RuntimeArtifactTSV.double(avgPreWindowMinutes),
            RuntimeArtifactTSV.double(avgPostWindowMinutes),
            RuntimeArtifactTSV.double(avgSurpriseZAbs),
            RuntimeArtifactTSV.double(avgRevisionAbs),
            RuntimeArtifactTSV.double(avgSourceTrust),
            RuntimeArtifactTSV.double(avgCurrencyRelevance),
            RuntimeArtifactTSV.double(checksum01),
            RuntimeArtifactTSV.double(provenanceHash01),
            RuntimeArtifactTSV.double(leakageGuardScore),
            RuntimeArtifactTSV.bool(leakageSafe)
        ]
    }
}

public struct RuntimePerformanceManifestRow: Codable, Hashable, Sendable {
    public static let header = [
        "row_type",
        "stage_name",
        "ai_id",
        "ai_name",
        "mean_ms",
        "max_ms",
        "obs",
        "predict_mean_ms",
        "predict_max_ms",
        "predict_obs",
        "update_mean_ms",
        "update_max_ms",
        "update_obs",
        "working_set_kb",
        "active_models"
    ]

    public var rowType: String
    public var stageName: String
    public var aiID: Int?
    public var aiName: String
    public var meanMS: Double?
    public var maxMS: Double?
    public var observations: Int?
    public var predictMeanMS: Double?
    public var predictMaxMS: Double?
    public var predictObservations: Int?
    public var updateMeanMS: Double?
    public var updateMaxMS: Double?
    public var updateObservations: Int?
    public var workingSetKB: Double?
    public var activeModels: Int

    public static func stage(
        _ stage: RuntimeStage,
        meanMS: Double,
        maxMS: Double,
        observations: Int,
        activeModels: Int
    ) -> RuntimePerformanceManifestRow {
        RuntimePerformanceManifestRow(
            rowType: "stage",
            stageName: stage.name,
            aiID: nil,
            aiName: "",
            meanMS: meanMS,
            maxMS: maxMS,
            observations: observations,
            predictMeanMS: nil,
            predictMaxMS: nil,
            predictObservations: nil,
            updateMeanMS: nil,
            updateMaxMS: nil,
            updateObservations: nil,
            workingSetKB: nil,
            activeModels: activeModels
        )
    }

    public static func plugin(
        aiID: Int,
        aiName: String,
        predictMeanMS: Double,
        predictMaxMS: Double,
        predictObservations: Int,
        updateMeanMS: Double,
        updateMaxMS: Double,
        updateObservations: Int,
        workingSetKB: Double,
        activeModels: Int
    ) -> RuntimePerformanceManifestRow {
        RuntimePerformanceManifestRow(
            rowType: "plugin",
            stageName: "",
            aiID: aiID,
            aiName: aiName,
            meanMS: nil,
            maxMS: nil,
            observations: nil,
            predictMeanMS: predictMeanMS,
            predictMaxMS: predictMaxMS,
            predictObservations: predictObservations,
            updateMeanMS: updateMeanMS,
            updateMaxMS: updateMaxMS,
            updateObservations: updateObservations,
            workingSetKB: workingSetKB,
            activeModels: activeModels
        )
    }

    public var tsvColumns: [String] {
        [
            rowType,
            stageName,
            aiID.map(String.init) ?? "",
            aiName,
            meanMS.map { RuntimeArtifactTSV.double($0) } ?? "",
            maxMS.map { RuntimeArtifactTSV.double($0) } ?? "",
            observations.map(String.init) ?? "",
            predictMeanMS.map { RuntimeArtifactTSV.double($0) } ?? "",
            predictMaxMS.map { RuntimeArtifactTSV.double($0) } ?? "",
            predictObservations.map(String.init) ?? "",
            updateMeanMS.map { RuntimeArtifactTSV.double($0) } ?? "",
            updateMaxMS.map { RuntimeArtifactTSV.double($0) } ?? "",
            updateObservations.map(String.init) ?? "",
            workingSetKB.map { RuntimeArtifactTSV.double($0, decimals: 3) } ?? "",
            String(activeModels)
        ]
    }
}

public struct ShadowFleetLedgerRow: Codable, Hashable, Sendable {
    public static let header = [
        "symbol",
        "ai_id",
        "ai_name",
        "family_id",
        "meta_weight",
        "reliability",
        "global_edge",
        "context_edge",
        "context_regret",
        "portfolio_objective",
        "portfolio_stability",
        "portfolio_corr",
        "portfolio_div",
        "route_value",
        "route_regret",
        "route_counterfactual",
        "shadow_score",
        "regime_id",
        "horizon_minutes",
        "obs",
        "policy_enter_prob",
        "policy_no_trade_prob",
        "policy_exit_prob",
        "policy_add_prob",
        "policy_reduce_prob",
        "policy_timeout_prob",
        "policy_tighten_prob",
        "policy_portfolio_fit",
        "policy_capital_efficiency",
        "policy_lifecycle_action",
        "portfolio_pressure",
        "control_plane_score",
        "portfolio_supervisor_score"
    ]

    public var symbol: String
    public var aiID: Int
    public var aiName: String
    public var familyID: Int
    public var metaWeight: Double
    public var reliability: Double
    public var globalEdge: Double
    public var contextEdge: Double
    public var contextRegret: Double
    public var portfolioObjective: Double
    public var portfolioStability: Double
    public var portfolioCorrelationPenalty: Double
    public var portfolioDiversification: Double
    public var routeValue: Double
    public var routeRegret: Double
    public var routeCounterfactual: Double
    public var shadowScore: Double
    public var regimeID: Int
    public var horizonMinutes: Int
    public var observations: Int
    public var policyEnterProb: Double
    public var policyNoTradeProb: Double
    public var policyExitProb: Double
    public var policyAddProb: Double
    public var policyReduceProb: Double
    public var policyTimeoutProb: Double
    public var policyTightenProb: Double
    public var policyPortfolioFit: Double
    public var policyCapitalEfficiency: Double
    public var policyLifecycleAction: PolicyLifecycleAction
    public var portfolioPressure: Double
    public var controlPlaneScore: Double
    public var portfolioSupervisorScore: Double

    public init(
        symbol: String,
        aiID: Int,
        aiName: String,
        family: AIFamily,
        metaWeight: Double,
        reliability: Double,
        globalEdge: Double,
        contextEdge: Double,
        contextRegret: Double,
        portfolioObjective: Double,
        portfolioStability: Double,
        portfolioCorrelationPenalty: Double,
        portfolioDiversification: Double,
        routeValue: Double,
        routeRegret: Double,
        routeCounterfactual: Double,
        regimeID: Int,
        horizonMinutes: Int,
        observations: Int,
        policyEnterProb: Double,
        policyNoTradeProb: Double,
        policyExitProb: Double,
        policyAddProb: Double,
        policyReduceProb: Double,
        policyTimeoutProb: Double,
        policyTightenProb: Double,
        policyPortfolioFit: Double,
        policyCapitalEfficiency: Double,
        policyLifecycleAction: PolicyLifecycleAction,
        portfolioPressure: Double,
        controlPlaneScore: Double,
        portfolioSupervisorScore: Double
    ) {
        self.symbol = RuntimeArtifactPaths.safeSymbol(symbol)
        self.aiID = aiID
        self.aiName = aiName
        self.familyID = family.rawValue
        self.metaWeight = fxClamp(metaWeight, 0.20, 3.00)
        self.reliability = fxClamp(reliability, 0.0, 1.0)
        self.globalEdge = fxClamp(globalEdge, -1.0, 1.0)
        self.contextEdge = fxClamp(contextEdge, -1.0, 1.0)
        self.contextRegret = fxClamp(contextRegret, 0.0, 1.0)
        self.portfolioObjective = fxClamp(portfolioObjective, -1.0, 1.0)
        self.portfolioStability = fxClamp(portfolioStability, 0.0, 1.0)
        self.portfolioCorrelationPenalty = fxClamp(portfolioCorrelationPenalty, 0.0, 1.0)
        self.portfolioDiversification = fxClamp(portfolioDiversification, 0.0, 1.0)
        self.routeValue = fxClamp(routeValue, -1.0, 1.0)
        self.routeRegret = fxClamp(routeRegret, 0.0, 1.0)
        self.routeCounterfactual = fxClamp(routeCounterfactual, -1.0, 1.0)
        self.shadowScore = Self.computeShadowScore(
            metaWeight: self.metaWeight,
            reliability: self.reliability,
            globalEdge: self.globalEdge,
            contextEdge: self.contextEdge,
            contextRegret: self.contextRegret,
            portfolioObjective: self.portfolioObjective,
            portfolioStability: self.portfolioStability,
            portfolioCorrelationPenalty: self.portfolioCorrelationPenalty,
            portfolioDiversification: self.portfolioDiversification,
            routeValue: self.routeValue,
            routeRegret: self.routeRegret,
            routeCounterfactual: self.routeCounterfactual
        )
        self.regimeID = regimeID
        self.horizonMinutes = max(1, horizonMinutes)
        self.observations = max(0, observations)
        self.policyEnterProb = fxClamp(policyEnterProb, 0.0, 1.0)
        self.policyNoTradeProb = fxClamp(policyNoTradeProb, 0.0, 1.0)
        self.policyExitProb = fxClamp(policyExitProb, 0.0, 1.0)
        self.policyAddProb = fxClamp(policyAddProb, 0.0, 1.0)
        self.policyReduceProb = fxClamp(policyReduceProb, 0.0, 1.0)
        self.policyTimeoutProb = fxClamp(policyTimeoutProb, 0.0, 1.0)
        self.policyTightenProb = fxClamp(policyTightenProb, 0.0, 1.0)
        self.policyPortfolioFit = fxClamp(policyPortfolioFit, 0.0, 1.0)
        self.policyCapitalEfficiency = fxClamp(policyCapitalEfficiency, 0.0, 1.0)
        self.policyLifecycleAction = policyLifecycleAction
        self.portfolioPressure = fxClamp(portfolioPressure, 0.0, 2.0)
        self.controlPlaneScore = fxClamp(controlPlaneScore, 0.0, 2.0)
        self.portfolioSupervisorScore = fxClamp(portfolioSupervisorScore, 0.0, 3.0)
    }

    public static func computeShadowScore(
        metaWeight: Double,
        reliability: Double,
        globalEdge: Double,
        contextEdge: Double,
        contextRegret: Double,
        portfolioObjective: Double,
        portfolioStability: Double,
        portfolioCorrelationPenalty: Double,
        portfolioDiversification: Double,
        routeValue: Double,
        routeRegret: Double,
        routeCounterfactual: Double
    ) -> Double {
        let meta = fxClamp(metaWeight, 0.20, 3.00)
        let rel = fxClamp(reliability, 0.0, 1.0)
        let global = fxClamp(globalEdge, -1.0, 1.0)
        let context = fxClamp(contextEdge, -1.0, 1.0)
        let regret = fxClamp(contextRegret, 0.0, 1.0)
        let objective = fxClamp(portfolioObjective, -1.0, 1.0)
        let stability = fxClamp(portfolioStability, 0.0, 1.0)
        let correlation = fxClamp(portfolioCorrelationPenalty, 0.0, 1.0)
        let diversification = fxClamp(portfolioDiversification, 0.0, 1.0)
        let value = fxClamp(routeValue, -1.0, 1.0)
        let routeLoss = fxClamp(routeRegret, 0.0, 1.0)
        let counterfactual = fxClamp(routeCounterfactual, -1.0, 1.0)
        return fxClamp(
            0.22 * rel +
                0.16 * global +
                0.16 * context -
                0.14 * regret +
                0.14 * objective +
                0.12 * value +
                0.10 * counterfactual -
                0.12 * routeLoss +
                0.08 * stability -
                0.06 * correlation +
                0.06 * diversification +
                0.05 * fxClamp((meta - 0.20) / 2.80, 0.0, 1.0),
            -1.0,
            1.0
        )
    }

    public var tsvColumns: [String] {
        [
            symbol,
            String(aiID),
            aiName,
            String(familyID),
            RuntimeArtifactTSV.double(metaWeight),
            RuntimeArtifactTSV.double(reliability),
            RuntimeArtifactTSV.double(globalEdge),
            RuntimeArtifactTSV.double(contextEdge),
            RuntimeArtifactTSV.double(contextRegret),
            RuntimeArtifactTSV.double(portfolioObjective),
            RuntimeArtifactTSV.double(portfolioStability),
            RuntimeArtifactTSV.double(portfolioCorrelationPenalty),
            RuntimeArtifactTSV.double(portfolioDiversification),
            RuntimeArtifactTSV.double(routeValue),
            RuntimeArtifactTSV.double(routeRegret),
            RuntimeArtifactTSV.double(routeCounterfactual),
            RuntimeArtifactTSV.double(shadowScore),
            String(regimeID),
            String(horizonMinutes),
            String(observations),
            RuntimeArtifactTSV.double(policyEnterProb),
            RuntimeArtifactTSV.double(policyNoTradeProb),
            RuntimeArtifactTSV.double(policyExitProb),
            RuntimeArtifactTSV.double(policyAddProb),
            RuntimeArtifactTSV.double(policyReduceProb),
            RuntimeArtifactTSV.double(policyTimeoutProb),
            RuntimeArtifactTSV.double(policyTightenProb),
            RuntimeArtifactTSV.double(policyPortfolioFit),
            RuntimeArtifactTSV.double(policyCapitalEfficiency),
            String(policyLifecycleAction.rawValue),
            RuntimeArtifactTSV.double(portfolioPressure),
            RuntimeArtifactTSV.double(controlPlaneScore),
            RuntimeArtifactTSV.double(portfolioSupervisorScore)
        ]
    }
}

public enum RuntimeStage: Int, Codable, Sendable, CaseIterable {
    case total = 0
    case featurePipeline
    case transfer
    case router
    case policy
    case shadow
    case controlPlane

    public var name: String {
        switch self {
        case .total: "total"
        case .featurePipeline: "feature_pipeline"
        case .transfer: "transfer"
        case .router: "router"
        case .policy: "policy"
        case .shadow: "shadow"
        case .controlPlane: "control_plane"
        }
    }
}

public struct RuntimeArtifactFileRepository {
    public let rootURL: URL
    public let fileManager: FileManager

    public init(rootURL: URL, fileManager: FileManager = .default) {
        self.rootURL = rootURL
        self.fileManager = fileManager
    }

    public func url(for relativePath: String) -> URL {
        rootURL.appendingPathComponent(relativePath)
    }

    public func writeTSV(relativePath: String, header: [String], rows: [[String]]) throws {
        let fileURL = url(for: relativePath)
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let text = RuntimeArtifactTSV.document(header: header, rows: rows)
        try text.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    public func fileSize(relativePath: String) throws -> Int64? {
        let fileURL = url(for: relativePath)
        guard fileManager.fileExists(atPath: fileURL.path) else { return nil }
        let attributes = try fileManager.attributesOfItem(atPath: fileURL.path)
        return (attributes[.size] as? NSNumber)?.int64Value
    }

    public func readRuntimeArtifact(symbol: String) throws -> RuntimeArtifactEnvelope? {
        let fileURL = url(for: RuntimeArtifactPaths.runtimeArtifactFile(symbol: symbol))
        guard fileManager.fileExists(atPath: fileURL.path) else { return nil }
        let data = try Data(contentsOf: fileURL)
        return try RuntimeArtifactBinaryCodec.decodeEnvelope(from: data)
    }

    public func writeRuntimeArtifact(symbol: String, envelope: RuntimeArtifactEnvelope) throws {
        let fileURL = url(for: RuntimeArtifactPaths.runtimeArtifactFile(symbol: symbol))
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try RuntimeArtifactBinaryCodec.encodeEnvelope(envelope)
        try data.write(to: fileURL, options: .atomic)
    }

    public func runtimeArtifactFileSize(symbol: String) throws -> Int64? {
        try fileSize(relativePath: RuntimeArtifactPaths.runtimeArtifactFile(symbol: symbol))
    }

    public func writePersistenceManifest(symbol: String, rows: [PersistenceCoverageManifestRow]) throws {
        try writeTSV(
            relativePath: RuntimeArtifactPaths.persistenceManifestFile(symbol: symbol),
            header: PersistenceCoverageManifestRow.header,
            rows: rows.map(\.tsvColumns)
        )
    }

    public func writeFeatureRegistryManifest(
        symbol: String,
        registry: FeatureRegistry = FeatureRegistry()
    ) throws {
        let rows = (0..<FXDataEngineConstants.aiFeatures).map {
            FeatureRegistryManifestRow(featureIndex: $0, registry: registry).tsvColumns
        }
        try writeTSV(
            relativePath: RuntimeArtifactPaths.featureManifestFile(symbol: symbol),
            header: FeatureRegistryManifestRow.header,
            rows: rows
        )
    }

    public func writeMacroDatasetManifest(symbol: String, row: MacroDatasetManifestRow) throws {
        var outputRow = row
        outputRow.symbol = RuntimeArtifactPaths.safeSymbol(symbol)
        try writeTSV(
            relativePath: RuntimeArtifactPaths.macroManifestFile(symbol: symbol),
            header: MacroDatasetManifestRow.header,
            rows: [outputRow.tsvColumns]
        )
    }

    public func writeRuntimePerformanceManifest(symbol: String, rows: [RuntimePerformanceManifestRow]) throws {
        try writeTSV(
            relativePath: RuntimeArtifactPaths.performanceManifestFile(symbol: symbol),
            header: RuntimePerformanceManifestRow.header,
            rows: rows.map(\.tsvColumns)
        )
    }

    public func writeShadowFleetLedger(symbol: String, rows: [ShadowFleetLedgerRow]) throws {
        let safeSymbol = RuntimeArtifactPaths.safeSymbol(symbol)
        let outputRows = rows.map { row -> ShadowFleetLedgerRow in
            var outputRow = row
            outputRow.symbol = safeSymbol
            return outputRow
        }
        try writeTSV(
            relativePath: RuntimeArtifactPaths.shadowLedgerFile(symbol: symbol),
            header: ShadowFleetLedgerRow.header,
            rows: outputRows.map(\.tsvColumns)
        )
    }
}
