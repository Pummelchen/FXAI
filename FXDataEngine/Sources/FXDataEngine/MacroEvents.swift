import Foundation

public enum MacroEventClass: Int, Codable, Sendable, CaseIterable {
    case unknown = 0
    case rates = 1
    case inflation = 2
    case labor = 3
    case growth = 4
    case trade = 5

    public var bias: Double {
        switch self {
        case .rates: 0.50
        case .inflation: 0.30
        case .labor: 0.40
        case .growth: 0.20
        case .trade: 0.10
        case .unknown: 0.0
        }
    }
}

public struct MacroEventRecord: Codable, Hashable, Sendable {
    public var symbol: String
    public var eventID: String
    public var country: String
    public var currency: String
    public var source: String
    public var eventTimeUTC: Int64
    public var preWindowMinutes: Int
    public var postWindowMinutes: Int
    public var importance: Double
    public var surprise: Double
    public var actualDelta: Double
    public var forecastDelta: Double
    public var priorDelta: Double
    public var revisionDelta: Double
    public var surpriseZ: Double
    public var relevanceHint: Double
    public var provenanceHash01: Double
    public var sourceTrust: Double
    public var releaseHash01: Double
    public var revisionChainHash01: Double
    public var eventClassRaw: Int

    public init(
        symbol: String,
        eventID: String,
        country: String,
        currency: String,
        source: String,
        eventTimeUTC: Int64,
        preWindowMinutes: Int,
        postWindowMinutes: Int,
        importance: Double,
        surprise: Double,
        actualDelta: Double,
        forecastDelta: Double,
        priorDelta: Double,
        revisionDelta: Double,
        surpriseZ: Double,
        relevanceHint: Double,
        provenanceHash01: Double,
        sourceTrust: Double,
        releaseHash01: Double,
        revisionChainHash01: Double,
        eventClassRaw: Int
    ) {
        self.symbol = symbol
        self.eventID = eventID
        self.country = country
        self.currency = currency
        self.source = source
        self.eventTimeUTC = eventTimeUTC
        self.preWindowMinutes = max(0, preWindowMinutes)
        self.postWindowMinutes = max(0, postWindowMinutes)
        self.importance = fxClamp(importance, 0.0, 1.0)
        self.surprise = fxClamp(surprise, -6.0, 6.0)
        self.actualDelta = fxClamp(actualDelta, -12.0, 12.0)
        self.forecastDelta = fxClamp(forecastDelta, -12.0, 12.0)
        self.priorDelta = fxClamp(priorDelta, -12.0, 12.0)
        self.revisionDelta = fxClamp(revisionDelta, -12.0, 12.0)
        self.surpriseZ = fxClamp(surpriseZ, -8.0, 8.0)
        self.relevanceHint = fxClamp(relevanceHint, 0.0, 1.0)
        self.provenanceHash01 = fxClamp(provenanceHash01, 0.0, 1.0)
        self.sourceTrust = fxClamp(sourceTrust, 0.0, 1.0)
        self.releaseHash01 = fxClamp(releaseHash01, 0.0, 1.0)
        self.revisionChainHash01 = fxClamp(revisionChainHash01, 0.0, 1.0)
        self.eventClassRaw = eventClassRaw
    }

    public var eventClass: MacroEventClass {
        MacroEventClass(rawValue: eventClassRaw) ?? .unknown
    }
}

public struct MacroEventDatasetStats: Codable, Hashable, Sendable {
    public static let schemaVersion = 2

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

    public init(
        schemaVersion: Int = MacroEventDatasetStats.schemaVersion,
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
        avgImportance: Double = 0,
        avgPreWindowMinutes: Double = 0,
        avgPostWindowMinutes: Double = 0,
        avgSurpriseZAbs: Double = 0,
        avgRevisionAbs: Double = 0,
        avgSourceTrust: Double = 0,
        avgCurrencyRelevance: Double = 0,
        checksum01: Double = 0,
        provenanceHash01: Double = 0,
        leakageGuardScore: Double = 0
    ) {
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
        self.avgImportance = fxSafeFinite(avgImportance)
        self.avgPreWindowMinutes = fxSafeFinite(avgPreWindowMinutes)
        self.avgPostWindowMinutes = fxSafeFinite(avgPostWindowMinutes)
        self.avgSurpriseZAbs = fxSafeFinite(avgSurpriseZAbs)
        self.avgRevisionAbs = fxSafeFinite(avgRevisionAbs)
        self.avgSourceTrust = fxSafeFinite(avgSourceTrust)
        self.avgCurrencyRelevance = fxSafeFinite(avgCurrencyRelevance)
        self.checksum01 = fxClamp(checksum01, 0.0, 1.0)
        self.provenanceHash01 = fxClamp(provenanceHash01, 0.0, 1.0)
        self.leakageGuardScore = fxClamp(leakageGuardScore, 0.0, 1.0)
    }
}

public struct MacroEventFeatures: Codable, Hashable, Sendable {
    public static let eventFeatureCount = 14
    public static let zero = MacroEventFeatures(
        preEmbargo: 0,
        postEmbargo: 0,
        eventImportance: 0,
        surpriseSigned: 0,
        surpriseAbs: 0,
        eventClassBias: 0,
        surpriseZScore: 0,
        revisionAbs: 0,
        currencyRelevance: 0,
        provenanceTrust: 0,
        ratesActivity: 0,
        inflationActivity: 0,
        laborActivity: 0,
        growthActivity: 0,
        clampOpen: false
    )

    public var preEmbargo: Double
    public var postEmbargo: Double
    public var eventImportance: Double
    public var surpriseSigned: Double
    public var surpriseAbs: Double
    public var eventClassBias: Double
    public var surpriseZScore: Double
    public var revisionAbs: Double
    public var currencyRelevance: Double
    public var provenanceTrust: Double
    public var ratesActivity: Double
    public var inflationActivity: Double
    public var laborActivity: Double
    public var growthActivity: Double

    public init(
        preEmbargo: Double = 0,
        postEmbargo: Double = 0,
        eventImportance: Double = 0,
        surpriseSigned: Double = 0,
        surpriseAbs: Double = 0,
        eventClassBias: Double = 0,
        surpriseZScore: Double = 0,
        revisionAbs: Double = 0,
        currencyRelevance: Double = 0,
        provenanceTrust: Double = 0,
        ratesActivity: Double = 0,
        inflationActivity: Double = 0,
        laborActivity: Double = 0,
        growthActivity: Double = 0
    ) {
        self.init(
            preEmbargo: preEmbargo,
            postEmbargo: postEmbargo,
            eventImportance: eventImportance,
            surpriseSigned: surpriseSigned,
            surpriseAbs: surpriseAbs,
            eventClassBias: eventClassBias,
            surpriseZScore: surpriseZScore,
            revisionAbs: revisionAbs,
            currencyRelevance: currencyRelevance,
            provenanceTrust: provenanceTrust,
            ratesActivity: ratesActivity,
            inflationActivity: inflationActivity,
            laborActivity: laborActivity,
            growthActivity: growthActivity,
            clampOpen: true
        )
    }

    private init(
        preEmbargo: Double,
        postEmbargo: Double,
        eventImportance: Double,
        surpriseSigned: Double,
        surpriseAbs: Double,
        eventClassBias: Double,
        surpriseZScore: Double,
        revisionAbs: Double,
        currencyRelevance: Double,
        provenanceTrust: Double,
        ratesActivity: Double,
        inflationActivity: Double,
        laborActivity: Double,
        growthActivity: Double,
        clampOpen: Bool
    ) {
        func unitClamp(_ value: Double) -> Double {
            clampOpen ? fxClampUnit(value) : fxClamp(value, 0.0, 1.0)
        }
        self.preEmbargo = unitClamp(preEmbargo)
        self.postEmbargo = unitClamp(postEmbargo)
        self.eventImportance = unitClamp(eventImportance)
        self.surpriseSigned = fxClamp(surpriseSigned, -6.0, 6.0)
        self.surpriseAbs = fxClamp(surpriseAbs, 0.0, 6.0)
        self.eventClassBias = fxClampSignedUnit(eventClassBias)
        self.surpriseZScore = fxClamp(surpriseZScore, -8.0, 8.0)
        self.revisionAbs = fxClamp(revisionAbs, 0.0, 8.0)
        self.currencyRelevance = unitClamp(currencyRelevance)
        self.provenanceTrust = unitClamp(provenanceTrust)
        self.ratesActivity = unitClamp(ratesActivity)
        self.inflationActivity = unitClamp(inflationActivity)
        self.laborActivity = unitClamp(laborActivity)
        self.growthActivity = unitClamp(growthActivity)
    }

    public var vector: [Double] {
        [
            preEmbargo,
            postEmbargo,
            eventImportance,
            surpriseSigned,
            surpriseAbs,
            eventClassBias,
            surpriseZScore,
            revisionAbs,
            currencyRelevance,
            provenanceTrust,
            ratesActivity,
            inflationActivity,
            laborActivity,
            growthActivity
        ]
    }
}

public struct MacroState: Codable, Hashable, Sendable {
    public static let zero = MacroState(
        policyDivergence: 0,
        policyPressure: 0,
        inflationPressure: 0,
        laborPressure: 0,
        growthPressure: 0,
        carryPressure: 0,
        eventDecay: 0,
        stateQuality: 0,
        clampOpen: false
    )

    public var policyDivergence: Double
    public var policyPressure: Double
    public var inflationPressure: Double
    public var laborPressure: Double
    public var growthPressure: Double
    public var carryPressure: Double
    public var eventDecay: Double
    public var stateQuality: Double

    public init(
        policyDivergence: Double = 0,
        policyPressure: Double = 0,
        inflationPressure: Double = 0,
        laborPressure: Double = 0,
        growthPressure: Double = 0,
        carryPressure: Double = 0,
        eventDecay: Double = 0,
        stateQuality: Double = 0
    ) {
        self.init(
            policyDivergence: policyDivergence,
            policyPressure: policyPressure,
            inflationPressure: inflationPressure,
            laborPressure: laborPressure,
            growthPressure: growthPressure,
            carryPressure: carryPressure,
            eventDecay: eventDecay,
            stateQuality: stateQuality,
            clampOpen: true
        )
    }

    private init(
        policyDivergence: Double,
        policyPressure: Double,
        inflationPressure: Double,
        laborPressure: Double,
        growthPressure: Double,
        carryPressure: Double,
        eventDecay: Double,
        stateQuality: Double,
        clampOpen: Bool
    ) {
        self.policyDivergence = fxClampSignedUnit(policyDivergence)
        self.policyPressure = fxClampSignedUnit(policyPressure)
        self.inflationPressure = fxClampSignedUnit(inflationPressure)
        self.laborPressure = fxClampSignedUnit(laborPressure)
        self.growthPressure = fxClampSignedUnit(growthPressure)
        self.carryPressure = fxClampSignedUnit(carryPressure)
        self.eventDecay = clampOpen ? fxClampUnit(eventDecay) : fxClamp(eventDecay, 0.0, 1.0)
        self.stateQuality = clampOpen ? fxClampUnit(stateQuality) : fxClamp(stateQuality, 0.0, 1.0)
    }
}

public struct MacroEventDataset: Sendable {
    public let records: [MacroEventRecord]
    public let stats: MacroEventDatasetStats

    public init(records: [MacroEventRecord], stats: MacroEventDatasetStats? = nil) {
        self.records = records
        self.stats = stats ?? MacroEventDataset.buildStats(records: records, parseErrors: 0)
    }

    public static func loadTSV(from url: URL, encoding: String.Encoding = .utf8) throws -> MacroEventDataset {
        try parseTSV(String(contentsOf: url, encoding: encoding))
    }

    public static func parseTSV(_ text: String) -> MacroEventDataset {
        var records: [MacroEventRecord] = []
        var parseErrors = 0
        var checksum = 0.0
        var provenanceHash = 0.0

        for rawLine in text.split(whereSeparator: \.isNewline) {
            let line = String(rawLine).trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
            let fields = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            let symbolTrim = MacroEventTools.normalizedToken(Self.field(fields, 0))
            let symbolLower = symbolTrim.lowercased()
            guard !symbolTrim.isEmpty, symbolTrim.first != "#", symbolLower != "symbol" else { continue }

            guard let eventTime = MacroEventTools.parseEventTimeUTC(Self.field(fields, 1)), eventTime > 0 else {
                parseErrors += 1
                continue
            }

            let preWindow = max(MacroEventTools.parseInt(Self.field(fields, 2)), 0)
            let postWindow = max(MacroEventTools.parseInt(Self.field(fields, 3)), 0)
            guard preWindow > 0 || postWindow > 0 else {
                parseErrors += 1
                continue
            }

            let importance = fxClamp(MacroEventTools.parseDouble(Self.field(fields, 4)), 0.0, 1.0)
            let surprise = fxClamp(MacroEventTools.parseDouble(Self.field(fields, 5)), -6.0, 6.0)
            let actualDelta = fxClamp(MacroEventTools.parseDouble(Self.field(fields, 6)), -12.0, 12.0)
            let forecastDelta = fxClamp(MacroEventTools.parseDouble(Self.field(fields, 7)), -12.0, 12.0)
            let classToken = Self.field(fields, 8)
            let eventClassRaw = MacroEventTools.parseEventClass(classToken)
            var eventID = MacroEventTools.normalizedToken(Self.field(fields, 9))
            if eventID.isEmpty {
                eventID = "\(symbolTrim)|\(eventTime)|\(eventClassRaw)"
            }
            let country = MacroEventTools.normalizedToken(Self.field(fields, 10))
            var currency = MacroEventTools.normalizedCurrencyToken(Self.field(fields, 11))
            if currency.isEmpty {
                currency = symbolTrim.count == 3
                    ? MacroEventTools.normalizedCurrencyToken(symbolTrim)
                    : MacroEventTools.countryToCurrency(country)
            }
            let source = MacroEventTools.normalizedToken(Self.field(fields, 12))
            let revisionDelta = fxClamp(MacroEventTools.parseDouble(Self.field(fields, 13)), -12.0, 12.0)
            let priorDelta = fxClamp(MacroEventTools.parseDouble(Self.field(fields, 14)), -12.0, 12.0)
            var surpriseZ = fxClamp(MacroEventTools.parseDouble(Self.field(fields, 15)), -8.0, 8.0)
            let standardizedSurpriseZ = MacroEventTools.normalizedSurpriseZ(
                surprise: surprise,
                actualDelta: actualDelta,
                forecastDelta: forecastDelta,
                priorDelta: priorDelta,
                revisionDelta: revisionDelta,
                importance: importance,
                eventClassRaw: eventClassRaw
            )
            if abs(surpriseZ) <= 1e-9 {
                surpriseZ = standardizedSurpriseZ
            } else {
                surpriseZ = fxClamp(0.55 * surpriseZ + 0.45 * standardizedSurpriseZ, -8.0, 8.0)
            }

            let sourceTrust = MacroEventTools.sourceTrust(source)
            let revisionChainKey = MacroEventTools.revisionChainKey(
                eventID: eventID,
                currency: currency,
                country: country,
                eventClassRaw: eventClassRaw
            )
            let releaseHash = MacroEventTools.stringChecksum01("\(revisionChainKey)|\(eventTime)")
            let revisionChainHash = MacroEventTools.stringChecksum01(revisionChainKey)
            let recordProvenanceHash = MacroEventTools.stringChecksum01("\(eventID)|\(source)|\(currency)|\(country)")
            let relevanceHint = MacroEventTools.currencyRelevance(currency: currency, symbol: symbolTrim)

            let record = MacroEventRecord(
                symbol: symbolTrim,
                eventID: eventID,
                country: country,
                currency: currency,
                source: source,
                eventTimeUTC: eventTime,
                preWindowMinutes: preWindow,
                postWindowMinutes: postWindow,
                importance: importance,
                surprise: surprise,
                actualDelta: actualDelta,
                forecastDelta: forecastDelta,
                priorDelta: priorDelta,
                revisionDelta: revisionDelta,
                surpriseZ: surpriseZ,
                relevanceHint: relevanceHint,
                provenanceHash01: recordProvenanceHash,
                sourceTrust: sourceTrust,
                releaseHash01: releaseHash,
                revisionChainHash01: revisionChainHash,
                eventClassRaw: eventClassRaw
            )

            records.append(record)
            checksum += 0.31 * MacroEventTools.stringChecksum01(symbolTrim)
                + 0.11 * MacroEventTools.stringChecksum01(classToken)
                + 0.09 * releaseHash
                + 0.07 * revisionChainHash
                + 0.07 * importance
                + 0.03 * abs(surprise)
                + 0.05 * sourceTrust
            provenanceHash += 0.65 * recordProvenanceHash + 0.35 * releaseHash
        }

        let stats = buildStats(
            records: records,
            parseErrors: parseErrors,
            checksumAccumulator: checksum,
            provenanceHashAccumulator: provenanceHash
        )
        return MacroEventDataset(records: records, stats: stats)
    }

    public var leakageSafe: Bool {
        stats.recordCount > 0
            && stats.schemaVersion >= MacroEventDatasetStats.schemaVersion
            && stats.parseErrors == 0
            && stats.distinctEventIDs > 0
            && stats.distinctRevisionChains > 0
            && stats.distinctCurrencies > 0
            && stats.avgSourceTrust >= 0.60
            && stats.leakageGuardScore >= 0.78
    }

    public func features(symbol: String, sampleTimeUTC: Int64) -> MacroEventFeatures {
        guard sampleTimeUTC > 0, !records.isEmpty else { return .zero }
        var preEmbargo = 0.0
        var postEmbargo = 0.0
        var eventImportance = 0.0
        var surpriseSigned = 0.0
        var surpriseAbs = 0.0
        var eventClassBias = 0.0
        var surpriseZScore = 0.0
        var revisionAbs = 0.0
        var currencyRelevance = 0.0
        var provenanceTrust = 0.0
        var ratesActivity = 0.0
        var inflationActivity = 0.0
        var laborActivity = 0.0
        var growthActivity = 0.0
        var signedWeight = 0.0
        var zWeight = 0.0
        var classWeight = 0.0
        var revisionWeight = 0.0

        for event in records {
            guard MacroEventTools.eventAffectsSymbol(
                eventSymbol: event.symbol,
                currency: event.currency,
                country: event.country,
                symbol: symbol
            ) else { continue }

            let dtMinutes = Double(sampleTimeUTC - event.eventTimeUTC) / 60.0
            let inPre = dtMinutes < 0.0 && -dtMinutes <= Double(max(event.preWindowMinutes, 1))
            let inPost = dtMinutes >= 0.0 && dtMinutes <= Double(max(event.postWindowMinutes, 1))
            guard inPre || inPost else { continue }

            let baseImportance = fxClamp(event.importance, 0.0, 1.0)
            let proximity: Double
            if inPre {
                proximity = fxClamp(1.0 - ((-dtMinutes) / Double(max(event.preWindowMinutes, 1))), 0.0, 1.0)
            } else {
                proximity = fxClamp(1.0 - (dtMinutes / Double(max(event.postWindowMinutes, 1))), 0.0, 1.0)
            }

            var sourceTrust = fxClamp(event.sourceTrust, 0.0, 1.0)
            if sourceTrust <= 1e-6 {
                sourceTrust = MacroEventTools.sourceTrust(event.source)
            }
            var relevance = max(
                MacroEventTools.currencyRelevance(currency: event.currency, symbol: symbol),
                fxClamp(event.relevanceHint, 0.0, 1.0)
            )
            if relevance <= 1e-6, event.currency.isEmpty, event.symbol.isEmpty {
                relevance = 1.0
            }
            let knownWeight = fxClamp(
                baseImportance
                    * (0.35 + 0.65 * proximity)
                    * max(sourceTrust, 0.25)
                    * max(relevance, 0.35),
                0.0,
                1.0
            )
            eventImportance = max(eventImportance, knownWeight)
            currencyRelevance = max(currencyRelevance, fxClamp(relevance, 0.0, 1.0))
            provenanceTrust = max(provenanceTrust, fxClamp(sourceTrust, 0.0, 1.0))

            if inPre {
                preEmbargo = max(preEmbargo, knownWeight)
                eventClassBias = fxClamp(eventClassBias + knownWeight * event.eventClass.bias, -2.0, 2.0)
                classWeight += knownWeight
                continue
            }

            postEmbargo = max(postEmbargo, knownWeight)
            let realizedSurprise = fxClamp(
                event.surprise
                    + 0.20 * (event.actualDelta - event.forecastDelta)
                    + 0.15 * event.revisionDelta,
                -6.0,
                6.0
            )
            surpriseSigned += knownWeight * realizedSurprise
            surpriseAbs += knownWeight * abs(realizedSurprise)
            signedWeight += knownWeight
            surpriseZScore += knownWeight * event.surpriseZ
            zWeight += knownWeight
            revisionAbs += knownWeight * abs(event.revisionDelta)
            revisionWeight += knownWeight
            eventClassBias = fxClamp(eventClassBias + knownWeight * event.eventClass.bias, -2.0, 2.0)
            classWeight += knownWeight

            switch event.eventClass {
            case .rates: ratesActivity = max(ratesActivity, knownWeight)
            case .inflation: inflationActivity = max(inflationActivity, knownWeight)
            case .labor: laborActivity = max(laborActivity, knownWeight)
            case .growth: growthActivity = max(growthActivity, knownWeight)
            case .trade, .unknown: break
            }
        }

        if signedWeight > 1e-6 {
            surpriseSigned /= signedWeight
            surpriseAbs /= signedWeight
        } else {
            surpriseSigned = 0.0
            surpriseAbs = 0.0
        }
        eventClassBias = classWeight > 1e-6 ? eventClassBias / classWeight : 0.0
        surpriseZScore = zWeight > 1e-6 ? surpriseZScore / zWeight : 0.0
        revisionAbs = revisionWeight > 1e-6 ? revisionAbs / revisionWeight : 0.0

        return MacroEventFeatures(
            preEmbargo: preEmbargo,
            postEmbargo: postEmbargo,
            eventImportance: eventImportance,
            surpriseSigned: surpriseSigned,
            surpriseAbs: surpriseAbs,
            eventClassBias: eventClassBias,
            surpriseZScore: surpriseZScore,
            revisionAbs: revisionAbs,
            currencyRelevance: currencyRelevance,
            provenanceTrust: provenanceTrust,
            ratesActivity: ratesActivity,
            inflationActivity: inflationActivity,
            laborActivity: laborActivity,
            growthActivity: growthActivity
        )
    }

    public func state(symbol: String, sampleTimeUTC: Int64) -> MacroState {
        guard sampleTimeUTC > 0, !records.isEmpty else { return .zero }

        var policyAcc = 0.0
        var policyWeight = 0.0
        var inflationAcc = 0.0
        var inflationWeight = 0.0
        var laborAcc = 0.0
        var laborWeight = 0.0
        var growthAcc = 0.0
        var growthWeight = 0.0
        var tradeAcc = 0.0
        var tradeWeight = 0.0
        var trustSum = 0.0
        var relevanceSum = 0.0
        var coverageWeight = 0.0
        var familyHits = 0
        var eventDecay = 0.0

        for event in records {
            guard MacroEventTools.eventAffectsSymbol(
                eventSymbol: event.symbol,
                currency: event.currency,
                country: event.country,
                symbol: symbol
            ) else { continue }

            let dtMinutes = Double(sampleTimeUTC - event.eventTimeUTC) / 60.0
            let lookback = Double(max(event.postWindowMinutes * 3, 240))
            let lookahead = Double(max(event.preWindowMinutes, 60))
            guard dtMinutes >= -lookahead, dtMinutes <= lookback else { continue }

            let orientation = MacroEventTools.currencyOrientation(currency: event.currency, symbol: symbol)
            guard abs(orientation) > 1e-9 else { continue }

            var sourceTrust = fxClamp(event.sourceTrust, 0.0, 1.0)
            if sourceTrust <= 1e-6 {
                sourceTrust = MacroEventTools.sourceTrust(event.source)
            }
            let relevance = max(
                MacroEventTools.currencyRelevance(currency: event.currency, symbol: symbol),
                fxClamp(event.relevanceHint, 0.0, 1.0)
            )
            guard relevance > 1e-6 else { continue }

            let importance = fxClamp(event.importance, 0.0, 1.0)
            let temporal: Double
            if dtMinutes < 0.0 {
                temporal = fxClamp(0.60 + 0.40 * (1.0 - ((-dtMinutes) / max(lookahead, 1.0))), 0.0, 1.0)
            } else {
                temporal = fxClamp(exp(-dtMinutes / max(lookback, 1.0)), 0.0, 1.0)
            }
            let weight = fxClamp(
                (0.25 + 0.75 * importance)
                    * (0.20 + 0.80 * sourceTrust)
                    * (0.25 + 0.75 * relevance)
                    * temporal,
                0.0,
                1.0
            )
            guard weight > 1e-6 else { continue }

            let impact = orientation * MacroEventTools.impactSigned(event)
            trustSum += weight * sourceTrust
            relevanceSum += weight * fxClamp(relevance, 0.0, 1.0)
            coverageWeight += weight
            eventDecay = max(eventDecay, weight)

            switch event.eventClass {
            case .rates:
                policyAcc += weight * impact
                policyWeight += weight
                familyHits += 1
            case .inflation:
                inflationAcc += weight * impact
                inflationWeight += weight
                familyHits += 1
            case .labor:
                laborAcc += weight * impact
                laborWeight += weight
                familyHits += 1
            case .growth:
                growthAcc += weight * impact
                growthWeight += weight
                familyHits += 1
            case .trade:
                tradeAcc += weight * impact
                tradeWeight += weight
                familyHits += 1
            case .unknown:
                break
            }
        }

        let policyNorm = policyWeight > 1e-6 ? policyAcc / policyWeight : 0.0
        let inflationNorm = inflationWeight > 1e-6 ? inflationAcc / inflationWeight : 0.0
        let laborNorm = laborWeight > 1e-6 ? laborAcc / laborWeight : 0.0
        let growthNorm = growthWeight > 1e-6 ? growthAcc / growthWeight : 0.0
        let tradeNorm = tradeWeight > 1e-6 ? tradeAcc / tradeWeight : 0.0
        let policyDivergence = fxClampSignedUnit(policyNorm - 0.35 * inflationNorm + 0.20 * growthNorm)
        let policyPressure = fxClampSignedUnit(0.70 * policyNorm + 0.30 * inflationNorm)
        let growthPressure = fxClampSignedUnit(0.78 * growthNorm + 0.22 * tradeNorm)
        let carryPressure = fxClampSignedUnit(0.60 * policyPressure + 0.25 * policyDivergence + 0.15 * growthPressure)
        let trustMean = coverageWeight > 1e-6 ? trustSum / coverageWeight : 0.0
        let relevanceMean = coverageWeight > 1e-6 ? relevanceSum / coverageWeight : 0.0
        let familyDiversity = fxClampUnit(Double(familyHits) / 5.0)
        let density = fxClampUnit(coverageWeight / 2.0)
        let stateQuality = fxClampUnit(
            0.34 * trustMean
                + 0.28 * relevanceMean
                + 0.20 * density
                + 0.18 * familyDiversity
        )

        return MacroState(
            policyDivergence: policyDivergence,
            policyPressure: policyPressure,
            inflationPressure: inflationNorm,
            laborPressure: laborNorm,
            growthPressure: growthPressure,
            carryPressure: carryPressure,
            eventDecay: eventDecay,
            stateQuality: stateQuality
        )
    }

    public func macroFeatureVector(symbol: String, sampleTimeUTC: Int64) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.macroEventFeatures)
        let eventFeatures = features(symbol: symbol, sampleTimeUTC: sampleTimeUTC).vector
        for index in 0..<min(eventFeatures.count, 14) {
            output[index] = eventFeatures[index]
        }
        let state = state(symbol: symbol, sampleTimeUTC: sampleTimeUTC)
        output[14] = state.policyDivergence
        output[15] = state.policyPressure
        output[16] = state.inflationPressure
        output[17] = state.laborPressure
        output[18] = state.growthPressure
        output[19] = state.stateQuality
        return output.map { fxSafeFinite($0) }
    }

    public func fillMacroFeatures(into features: inout [Double], symbol: String, sampleTimeUTC: Int64) {
        guard features.count >= FXDataEngineConstants.aiFeatures else { return }
        let vector = macroFeatureVector(symbol: symbol, sampleTimeUTC: sampleTimeUTC)
        for index in 0..<min(vector.count, FXDataEngineConstants.macroEventFeatures) {
            features[FXDataEngineConstants.macroEventFeatureOffset + index] = vector[index]
        }
    }

    public func windowScore(symbol: String, sampleTimesUTC: [Int64], startIndex: Int = 0, bars: Int? = nil) -> Double {
        guard !records.isEmpty, startIndex >= 0, startIndex < sampleTimesUTC.count else { return 0.0 }
        let resolvedBars = max(0, bars ?? (sampleTimesUTC.count - startIndex))
        guard resolvedBars > 0, startIndex + resolvedBars <= sampleTimesUTC.count else { return 0.0 }

        let stride = max(1, resolvedBars / 24)
        var samples = 0
        var activitySum = 0.0
        var importanceSum = 0.0
        var surpriseSum = 0.0
        var index = startIndex
        while index < startIndex + resolvedBars {
            let eventFeatures = features(symbol: symbol, sampleTimeUTC: sampleTimesUTC[index])
            let activity = max(eventFeatures.eventImportance, max(eventFeatures.preEmbargo, eventFeatures.postEmbargo))
            activitySum += activity
            importanceSum += eventFeatures.eventImportance
            surpriseSum += eventFeatures.surpriseAbs
            samples += 1
            index += stride
        }

        guard samples > 0 else { return 0.0 }
        let coverage = activitySum / Double(samples)
        let importanceMean = importanceSum / Double(samples)
        let surpriseMean = surpriseSum / Double(samples)
        return fxClamp(
            0.55 * coverage
                + 0.25 * importanceMean
                + 0.20 * fxClamp(surpriseMean / 6.0, 0.0, 1.0),
            0.0,
            1.0
        )
    }

    private static func field(_ fields: [String], _ index: Int) -> String {
        index < fields.count ? fields[index] : ""
    }

    private static func buildStats(
        records: [MacroEventRecord],
        parseErrors: Int,
        checksumAccumulator: Double = 0,
        provenanceHashAccumulator: Double = 0
    ) -> MacroEventDatasetStats {
        guard !records.isEmpty else {
            return MacroEventDatasetStats(parseErrors: parseErrors)
        }

        var stats = MacroEventDatasetStats(recordCount: records.count, parseErrors: parseErrors)
        var symbols: Set<String> = []
        var sources: Set<String> = []
        var eventIDs: Set<String> = []
        var countries: Set<String> = []
        var currencies: Set<String> = []
        var revisionChains: Set<Double> = []

        for record in records {
            symbols.insert(record.symbol)
            if !record.source.isEmpty { sources.insert(record.source) }
            eventIDs.insert(record.eventID)
            if !record.country.isEmpty { countries.insert(record.country) }
            if !record.currency.isEmpty { currencies.insert(record.currency) }
            revisionChains.insert(record.revisionChainHash01)

            stats.avgImportance += record.importance
            stats.avgPreWindowMinutes += Double(record.preWindowMinutes)
            stats.avgPostWindowMinutes += Double(record.postWindowMinutes)
            stats.avgSurpriseZAbs += abs(record.surpriseZ)
            stats.avgRevisionAbs += abs(record.revisionDelta)
            stats.avgSourceTrust += record.sourceTrust
            stats.avgCurrencyRelevance += fxClamp(record.relevanceHint, 0.0, 1.0)
            if stats.firstEventTimeUTC <= 0 || record.eventTimeUTC < stats.firstEventTimeUTC {
                stats.firstEventTimeUTC = record.eventTimeUTC
            }
            if stats.lastEventTimeUTC <= 0 || record.eventTimeUTC > stats.lastEventTimeUTC {
                stats.lastEventTimeUTC = record.eventTimeUTC
            }

            switch record.eventClass {
            case .rates: stats.familyRatesCount += 1
            case .inflation: stats.familyInflationCount += 1
            case .labor: stats.familyLaborCount += 1
            case .growth: stats.familyGrowthCount += 1
            case .trade: stats.familyTradeCount += 1
            case .unknown: break
            }
        }

        let denominator = Double(records.count)
        stats.distinctSymbols = symbols.count
        stats.distinctSources = sources.count
        stats.distinctEventIDs = eventIDs.count
        stats.distinctCountries = countries.count
        stats.distinctCurrencies = currencies.count
        stats.distinctRevisionChains = revisionChains.count
        stats.avgImportance /= denominator
        stats.avgPreWindowMinutes /= denominator
        stats.avgPostWindowMinutes /= denominator
        stats.avgSurpriseZAbs /= denominator
        stats.avgRevisionAbs /= denominator
        stats.avgSourceTrust /= denominator
        stats.avgCurrencyRelevance /= denominator
        let checksumFraction = checksumAccumulator / 8_192.0
        stats.checksum01 = checksumFraction - floor(checksumFraction)
        let provenanceFraction = provenanceHashAccumulator / denominator
        stats.provenanceHash01 = provenanceFraction - floor(provenanceFraction)

        let parseRatio = Double(parseErrors) / Double(max(records.count + parseErrors, 1))
        let eventCoverage = fxClamp(Double(stats.distinctEventIDs) / Double(max(stats.recordCount, 1)), 0.0, 1.0)
        let chainCoverage = fxClamp(Double(stats.distinctRevisionChains) / Double(max(stats.distinctEventIDs, 1)), 0.0, 1.0)
        let trustScore = fxClamp(stats.avgSourceTrust, 0.0, 1.0)
        let relevanceScore = fxClamp(stats.avgCurrencyRelevance, 0.0, 1.0)
        let diversityScore = fxClamp(Double(stats.distinctCurrencies + stats.distinctCountries) / 12.0, 0.0, 1.0)
        var score = 1.0
        score -= 0.55 * fxClamp(parseRatio, 0.0, 0.60)
        score *= fxClamp(0.55 + 0.45 * trustScore, 0.0, 1.0)
        score *= fxClamp(0.68 + 0.32 * relevanceScore, 0.0, 1.0)
        score *= fxClamp(0.72 + 0.28 * chainCoverage, 0.0, 1.0)
        score *= fxClamp(0.76 + 0.24 * max(eventCoverage, diversityScore), 0.0, 1.0)
        if stats.schemaVersion < MacroEventDatasetStats.schemaVersion {
            score = 0.0
        }
        stats.leakageGuardScore = fxClamp(score, 0.0, 1.0)
        return stats
    }
}

public enum MacroEventTools {
    public static func stripUtfBOM(_ rawValue: String) -> String {
        guard let first = rawValue.unicodeScalars.first, first.value == 65_279 else {
            return rawValue
        }
        return String(rawValue.dropFirst())
    }

    public static func normalizedToken(_ rawValue: String) -> String {
        stripUtfBOM(rawValue).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func normalizedCurrencyToken(_ rawValue: String) -> String {
        let value = normalizedToken(rawValue).uppercased(with: Locale(identifier: "en_US_POSIX"))
        return String(value.prefix(3))
    }

    public static func stringChecksum01(_ rawValue: String) -> Double {
        let value = normalizedToken(rawValue).uppercased(with: Locale(identifier: "en_US_POSIX"))
        guard !value.isEmpty else { return 0.0 }

        var accumulator = 0.0
        var scale = 1.0
        for (index, scalar) in value.unicodeScalars.enumerated() {
            accumulator += scale * Double((Int(scalar.value) % 97) + 1)
            scale *= 1.131
            if scale > 17.0 {
                scale = 1.0 + 0.17 * Double(index + 1)
            }
        }
        let fraction = accumulator / 104_729.0
        return fraction - floor(fraction)
    }

    public static func parseEventTimeUTC(_ rawValue: String) -> Int64? {
        let token = normalizedToken(rawValue)
            .replacingOccurrences(of: "T", with: " ")
            .replacingOccurrences(of: "Z", with: "")
            .replacingOccurrences(of: "-", with: ".")
        guard !token.isEmpty else { return nil }
        if let seconds = Int64(token), seconds > 0 {
            return seconds
        }

        let formats = ["yyyy.MM.dd HH:mm:ss", "yyyy.MM.dd HH:mm", "yyyy.MM.dd"]
        for format in formats {
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = TimeZone(secondsFromGMT: 0)
            formatter.dateFormat = format
            formatter.isLenient = false
            if let date = formatter.date(from: token) {
                return Int64(date.timeIntervalSince1970)
            }
        }
        return nil
    }

    public static func parseInt(_ rawValue: String) -> Int {
        let token = normalizedToken(rawValue)
        if let exact = Int(token) {
            return exact
        }
        if let numeric = Double(token), numeric.isFinite {
            return Int(numeric)
        }
        return 0
    }

    public static func parseDouble(_ rawValue: String) -> Double {
        let token = normalizedToken(rawValue)
        guard let value = Double(token), value.isFinite else { return 0.0 }
        return value
    }

    public static func parseEventClass(_ rawValue: String) -> Int {
        let value = normalizedToken(rawValue)
        let lower = value.lowercased()
        if let parsed = Int(value), String(parsed) == value {
            return parsed
        }
        if lower == "rates" || lower == "central_bank" || lower == "cb"
            || lower.contains("rate") || lower.contains("yield")
            || lower.contains("fomc") || lower.contains("ecb")
            || lower.contains("boe") || lower.contains("boj")
            || lower.contains("rba") || lower.contains("rbnz")
            || lower.contains("boc") || lower.contains("snb") {
            return MacroEventClass.rates.rawValue
        }
        if lower == "inflation" || lower == "cpi" || lower == "ppi"
            || lower.contains("pce") || lower.contains("hicp")
            || lower.contains("price") || lower.contains("deflator") {
            return MacroEventClass.inflation.rawValue
        }
        if lower == "labor" || lower == "employment" || lower == "nfp"
            || lower.contains("payroll") || lower.contains("unemployment")
            || lower.contains("jobless") || lower.contains("wage")
            || lower.contains("earnings") {
            return MacroEventClass.labor.rawValue
        }
        if lower == "growth" || lower == "gdp" || lower == "pmi"
            || lower.contains("manufacturing") || lower.contains("services")
            || lower.contains("retail") || lower.contains("industrial")
            || lower.contains("production") || lower.contains("consumer")
            || lower.contains("confidence") || lower.contains("sentiment")
            || lower.contains("housing") {
            return MacroEventClass.growth.rawValue
        }
        if lower == "trade" || lower == "balance"
            || lower.contains("current account")
            || lower.contains("export")
            || lower.contains("import") {
            return MacroEventClass.trade.rawValue
        }
        return MacroEventClass.unknown.rawValue
    }

    public static func countryToCurrency(_ rawCountry: String) -> String {
        switch normalizedToken(rawCountry).uppercased() {
        case "US", "USA", "UNITED STATES": "USD"
        case "EU", "EUR", "EUROZONE": "EUR"
        case "GB", "UK", "UNITED KINGDOM": "GBP"
        case "JP", "JAPAN": "JPY"
        case "AU", "AUSTRALIA": "AUD"
        case "NZ", "NEW ZEALAND": "NZD"
        case "CA", "CANADA": "CAD"
        case "CH", "SWITZERLAND": "CHF"
        case "CN", "CHINA": "CNY"
        case "SE", "SWEDEN": "SEK"
        case "NO", "NORWAY": "NOK"
        case "DK", "DENMARK": "DKK"
        case "SG", "SINGAPORE": "SGD"
        case "HK", "HONG KONG": "HKD"
        case "MX", "MEXICO": "MXN"
        case "ZA", "SOUTH AFRICA": "ZAR"
        default: ""
        }
    }

    public static func currencyBlock(_ rawCurrency: String) -> Int {
        switch normalizedCurrencyToken(rawCurrency) {
        case "EUR", "CHF", "GBP", "SEK", "NOK", "DKK": 1
        case "AUD", "NZD", "CAD": 2
        case "JPY", "CNY", "HKD", "SGD", "KRW": 3
        case "USD": 4
        default: 0
        }
    }

    public static func symbolBaseCurrency(_ rawSymbol: String) -> String {
        let symbol = normalizedToken(rawSymbol).uppercased()
        guard symbol.count >= 6 else { return "" }
        return String(symbol.prefix(3))
    }

    public static func symbolQuoteCurrency(_ rawSymbol: String) -> String {
        let symbol = normalizedToken(rawSymbol).uppercased()
        guard symbol.count >= 6 else { return "" }
        return String(symbol.dropFirst(3).prefix(3))
    }

    public static func sourceTrust(_ rawSource: String) -> Double {
        let source = normalizedToken(rawSource)
        let lower = source.lowercased()
        if source.isEmpty { return 0.45 }
        if lower.contains("official")
            || lower.contains("central bank")
            || lower.contains("statistics")
            || lower.contains("bureau")
            || lower.contains("ministry")
            || lower.contains("government") {
            return 1.0
        }
        if lower.contains("reuters") || lower.contains("bloomberg") || lower.contains("econoday") {
            return 0.92
        }
        if lower.contains("calendar")
            || lower.contains("consensus")
            || lower.contains("forexfactory")
            || lower.contains("investing") {
            return 0.85
        }
        return 0.70
    }

    public static func revisionChainKey(
        eventID: String,
        currency: String,
        country: String,
        eventClassRaw: Int
    ) -> String {
        "\(normalizedToken(eventID))|\(normalizedCurrencyToken(currency))|\(normalizedToken(country))|\(eventClassRaw)"
    }

    public static func normalizedSurpriseZ(
        surprise: Double,
        actualDelta: Double,
        forecastDelta: Double,
        priorDelta: Double,
        revisionDelta: Double,
        importance: Double,
        eventClassRaw: Int
    ) -> Double {
        let realized = surprise
            + 0.28 * (actualDelta - forecastDelta)
            + 0.18 * revisionDelta
            + 0.10 * (forecastDelta - priorDelta)
        var baseScale = 0.35
            + 0.22 * abs(forecastDelta)
            + 0.16 * abs(priorDelta)
            + 0.12 * abs(revisionDelta)
        baseScale *= 0.85 + 0.30 * fxClamp(importance, 0.0, 1.0)
        switch MacroEventClass(rawValue: eventClassRaw) ?? .unknown {
        case .rates: baseScale *= 1.28
        case .inflation: baseScale *= 1.12
        case .labor: baseScale *= 1.18
        case .growth: baseScale *= 0.96
        case .trade: baseScale *= 0.92
        case .unknown: break
        }
        return fxClamp(realized / max(baseScale, 0.22), -8.0, 8.0)
    }

    public static func currencyRelevance(currency rawCurrency: String, symbol rawSymbol: String) -> Double {
        let currency = normalizedCurrencyToken(rawCurrency)
        let base = symbolBaseCurrency(rawSymbol)
        let quote = symbolQuoteCurrency(rawSymbol)
        guard !currency.isEmpty else { return 0.0 }
        if currency == base || currency == quote {
            return 1.0
        }
        let eventBlock = currencyBlock(currency)
        let baseBlock = currencyBlock(base)
        let quoteBlock = currencyBlock(quote)
        if eventBlock > 0, eventBlock == baseBlock || eventBlock == quoteBlock {
            return 0.55
        }
        if currency == "USD", base == "CAD" || quote == "CAD" || base == "MXN" || quote == "MXN" {
            return 0.45
        }
        if currency == "CNY", base == "AUD" || quote == "AUD" || base == "NZD" || quote == "NZD" {
            return 0.40
        }
        return 0.0
    }

    public static func eventAffectsSymbol(
        eventSymbol rawEventSymbol: String,
        currency rawCurrency: String,
        country rawCountry: String,
        symbol rawSymbol: String
    ) -> Bool {
        let eventSymbol = normalizedToken(rawEventSymbol).uppercased()
        let symbol = normalizedToken(rawSymbol).uppercased()
        if eventSymbol.isEmpty || eventSymbol == "ALL" || eventSymbol == "*" {
            return true
        }
        if eventSymbol == symbol {
            return true
        }
        if eventSymbol.count == 3, symbol.contains(eventSymbol) {
            return true
        }
        var currency = normalizedCurrencyToken(rawCurrency)
        if currency.isEmpty {
            currency = countryToCurrency(rawCountry)
        }
        if currency.count == 3, currencyRelevance(currency: currency, symbol: symbol) > 0.35 {
            return true
        }
        return false
    }

    public static func currencyOrientation(currency rawCurrency: String, symbol rawSymbol: String) -> Double {
        let currency = normalizedCurrencyToken(rawCurrency)
        guard !currency.isEmpty else { return 0.0 }

        let base = symbolBaseCurrency(rawSymbol)
        let quote = symbolQuoteCurrency(rawSymbol)
        if currency == base { return 1.0 }
        if currency == quote { return -1.0 }

        let currentBlock = currencyBlock(currency)
        let baseBlock = currencyBlock(base)
        let quoteBlock = currencyBlock(quote)
        if currentBlock > 0, currentBlock == baseBlock, currentBlock != quoteBlock {
            return 0.35
        }
        if currentBlock > 0, currentBlock == quoteBlock, currentBlock != baseBlock {
            return -0.35
        }
        return 0.0
    }

    public static func impactSigned(_ event: MacroEventRecord) -> Double {
        let realized = fxClamp(
            0.55 * event.surpriseZ
                + 0.30 * event.surprise
                + 0.10 * (event.actualDelta - event.forecastDelta)
                + 0.05 * event.revisionDelta,
            -8.0,
            8.0
        )
        return realized / 8.0
    }
}
