import FXDataEngine
import Foundation

public enum FXPluginCertificationGate: String, Codable, CaseIterable, Hashable, Sendable {
    case registryCoverage
    case swiftCPURuntimeSmoke
    case ohlcvVolumeContract
    case historicalOrReferenceParity
    case backendSelectionPolicy
    case sineTestRuntimeSmoke
    case externalBackendDiscovery
    case metalSourceCompilation
    case metalLiveBufferParity
    case pyTorchLiveTrainPredictPersistence
    case tensorFlowLiveTrainPredictPersistence
    case nlpLiveContextPayload
    case coreMLNeuralEngineLiveParity
    case fxDatabaseAPIOnlyDataPath
    case fullVerificationRun
}

public struct FXPluginBackendCertificationStatus: Codable, Hashable, Sendable {
    public let backend: FXPluginAccelerationBackend
    public let requiredGates: [FXPluginCertificationGate]
    public let satisfiedGates: [FXPluginCertificationGate]

    public init(
        backend: FXPluginAccelerationBackend,
        requiredGates: [FXPluginCertificationGate],
        satisfiedGates: [FXPluginCertificationGate]
    ) {
        self.backend = backend
        self.requiredGates = requiredGates
        self.satisfiedGates = satisfiedGates
    }

    public var blockingGates: [FXPluginCertificationGate] {
        let satisfied = Set(satisfiedGates)
        return requiredGates.filter { !satisfied.contains($0) }
    }

    public var is100PercentCertified: Bool {
        blockingGates.isEmpty
    }
}

public struct FXPluginCertificationReport: Codable, Hashable, Sendable {
    public let pluginName: String
    public let backendStatuses: [FXPluginBackendCertificationStatus]

    public init(pluginName: String, backendStatuses: [FXPluginBackendCertificationStatus]) {
        self.pluginName = pluginName
        self.backendStatuses = backendStatuses
    }

    public var blockingGates: [FXPluginCertificationGate] {
        let gates = Set(backendStatuses.flatMap(\.blockingGates))
        return FXPluginCertificationGate.allCases.filter { gates.contains($0) }
    }

    public var is100PercentCertified: Bool {
        !backendStatuses.isEmpty && backendStatuses.allSatisfy(\.is100PercentCertified)
    }
}

public enum FXPluginCertificationError: Error, Equatable, Sendable, CustomStringConvertible {
    case invalidAuditCoverage(String)
    case incompleteCertification([FXPluginCertificationReport])

    public var description: String {
        switch self {
        case .invalidAuditCoverage(let reason):
            return "FXPlugins certification audit coverage is invalid: \(reason)"
        case .incompleteCertification(let reports):
            let blockedBackendCount = reports.reduce(0) { count, report in
                count + report.backendStatuses.filter { !$0.is100PercentCertified }.count
            }
            let sample = reports.prefix(6)
                .map { report in
                    let gates = report.blockingGates.map(\.rawValue).joined(separator: ",")
                    return "\(report.pluginName)[\(gates)]"
                }
                .joined(separator: "; ")
            return "FXPlugins 100% certification is incomplete: \(reports.count) plugins and \(blockedBackendCount) backend declarations still have blocking gates. \(sample)"
        }
    }
}

public enum FXAIPluginCertificationRegistry {
    public static func certificationReports() -> [FXPluginCertificationReport] {
        FXAIPluginRegistry.availablePlugins()
            .compactMap { $0 as? any FXAIPlannedPlugin }
            .map { certificationReport(for: $0) }
            .sorted { $0.pluginName < $1.pluginName }
    }

    public static func uncertifiedReports() -> [FXPluginCertificationReport] {
        certificationReports().filter { !$0.is100PercentCertified }
    }

    public static func validateAuditCoverage() throws {
        let plannedPlugins = FXAIPluginRegistry.availablePlugins().compactMap { $0 as? any FXAIPlannedPlugin }
        guard plannedPlugins.count == FXAIPluginRegistry.availablePlugins().count else {
            throw FXPluginCertificationError.invalidAuditCoverage("not every registry plugin conforms to FXAIPlannedPlugin")
        }

        let reports = certificationReports()
        guard reports.count == plannedPlugins.count else {
            throw FXPluginCertificationError.invalidAuditCoverage("report count \(reports.count) does not match plugin count \(plannedPlugins.count)")
        }

        let expectedPluginNames = Set(plannedPlugins.map(\.manifest.aiName))
        let reportedPluginNames = Set(reports.map(\.pluginName))
        guard expectedPluginNames == reportedPluginNames else {
            let missing = expectedPluginNames.subtracting(reportedPluginNames).sorted().joined(separator: ",")
            let extra = reportedPluginNames.subtracting(expectedPluginNames).sorted().joined(separator: ",")
            throw FXPluginCertificationError.invalidAuditCoverage("plugin name mismatch missing=[\(missing)] extra=[\(extra)]")
        }

        for plugin in plannedPlugins {
            guard plugin.accelerationPlan.pluginName == plugin.manifest.aiName else {
                throw FXPluginCertificationError.invalidAuditCoverage(
                    "\(plugin.manifest.aiName) acceleration plan is named \(plugin.accelerationPlan.pluginName)"
                )
            }
            let report = reports.first { $0.pluginName == plugin.manifest.aiName }
            guard let report else {
                throw FXPluginCertificationError.invalidAuditCoverage("\(plugin.manifest.aiName) has no certification report")
            }
            let expectedBackends = plugin.accelerationPlan.declaredBackends
            let reportedBackends = report.backendStatuses.map(\.backend)
            guard expectedBackends == reportedBackends else {
                throw FXPluginCertificationError.invalidAuditCoverage(
                    "\(plugin.manifest.aiName) backend mismatch expected=\(expectedBackends.map(\.rawValue)) reported=\(reportedBackends.map(\.rawValue))"
                )
            }
            for status in report.backendStatuses {
                guard !status.requiredGates.isEmpty else {
                    throw FXPluginCertificationError.invalidAuditCoverage("\(plugin.manifest.aiName) \(status.backend.rawValue) has no required gates")
                }
                let required = Set(status.requiredGates)
                let satisfied = Set(status.satisfiedGates)
                guard satisfied.isSubset(of: required) else {
                    let invalid = satisfied.subtracting(required).map(\.rawValue).sorted().joined(separator: ",")
                    throw FXPluginCertificationError.invalidAuditCoverage(
                        "\(plugin.manifest.aiName) \(status.backend.rawValue) satisfies gates that are not required: \(invalid)"
                    )
                }
            }
        }
    }

    public static func requireAllPlugins100PercentCertified() throws {
        try validateAuditCoverage()
        let blocked = uncertifiedReports()
        guard blocked.isEmpty else {
            throw FXPluginCertificationError.incompleteCertification(blocked)
        }
    }

    private static func certificationReport(for plugin: any FXAIPlannedPlugin) -> FXPluginCertificationReport {
        let plan = plugin.accelerationPlan
        let statuses = plan.declaredBackends.map { backend in
            let required = requiredGates(for: backend)
            let satisfied = satisfiedGates(for: plan, backend: backend)
            return FXPluginBackendCertificationStatus(
                backend: backend,
                requiredGates: ordered(required),
                satisfiedGates: ordered(satisfied.intersection(required))
            )
        }
        return FXPluginCertificationReport(pluginName: plugin.manifest.aiName, backendStatuses: statuses)
    }

    private static func requiredGates(for backend: FXPluginAccelerationBackend) -> Set<FXPluginCertificationGate> {
        var gates: Set<FXPluginCertificationGate> = [
            .registryCoverage,
            .swiftCPURuntimeSmoke,
            .ohlcvVolumeContract,
            .historicalOrReferenceParity,
            .backendSelectionPolicy,
            .sineTestRuntimeSmoke,
            .fxDatabaseAPIOnlyDataPath,
            .fullVerificationRun
        ]

        switch backend {
        case .swiftScalar, .swiftSIMD, .accelerate:
            break
        case .metal:
            gates.formUnion([.metalSourceCompilation, .metalLiveBufferParity])
        case .pyTorchMPS:
            gates.formUnion([.externalBackendDiscovery, .pyTorchLiveTrainPredictPersistence])
        case .tensorFlowMetal:
            gates.formUnion([.externalBackendDiscovery, .tensorFlowLiveTrainPredictPersistence])
        case .foundationNLP:
            gates.formUnion([.externalBackendDiscovery, .nlpLiveContextPayload])
        case .coreMLNeuralEngine:
            gates.insert(.coreMLNeuralEngineLiveParity)
        }
        return gates
    }

    private static func satisfiedGates(
        for plan: FXPluginAccelerationPlan,
        backend: FXPluginAccelerationBackend
    ) -> Set<FXPluginCertificationGate> {
        var gates: Set<FXPluginCertificationGate> = [
            .registryCoverage,
            .swiftCPURuntimeSmoke,
            .backendSelectionPolicy,
            .sineTestRuntimeSmoke
        ]

        if plan.usesVolumeWhenAvailable {
            gates.insert(.ohlcvVolumeContract)
        }
        if backend.requiresExternalPython || backend == .foundationNLP {
            if FXAIPluginBackendDiscovery.externalPythonDescriptor(pluginName: plan.pluginName, backend: backend) != nil {
                gates.insert(.externalBackendDiscovery)
            }
        }
        return gates
    }

    private static func ordered(_ gates: Set<FXPluginCertificationGate>) -> [FXPluginCertificationGate] {
        FXPluginCertificationGate.allCases.filter { gates.contains($0) }
    }
}
