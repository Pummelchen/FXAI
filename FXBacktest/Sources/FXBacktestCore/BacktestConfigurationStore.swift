import Foundation
import FXBacktestAPI

public enum BacktestCommonConfigurationDefaults {
    public static func parameters(defaultMaxWorkers: Int = max(1, ProcessInfo.processInfo.activeProcessorCount)) -> [FXBacktestConfigurationParameterDTO] {
        [
            parameter(
                key: "initial_deposit_usd",
                displayName: "Virtual Trade Account USD",
                kind: .decimal,
                defaultValue: 1_000,
                minimum: 100,
                step: 100,
                maximum: 1_000_000,
                unit: "USD",
                description: "Starting virtual account balance used for every offline backtest run."
            ),
            parameter(
                key: "contract_size_units",
                displayName: "FX Contract Size",
                kind: .decimal,
                defaultValue: 100_000,
                minimum: 1_000,
                step: 1_000,
                maximum: 1_000_000,
                unit: "base-units",
                description: "Contract size per 1.00 lot for FX pair profit and loss calculations."
            ),
            parameter(
                key: "lot_size_lots",
                displayName: "FX Lot Size",
                kind: .decimal,
                defaultValue: 0.01,
                minimum: 0.01,
                step: 0.01,
                maximum: 100,
                unit: "lots",
                description: "Default trade size for FX pair backtests."
            ),
            parameter(
                key: "max_workers",
                displayName: "CPU Workers",
                kind: .integer,
                defaultValue: Double(defaultMaxWorkers),
                minimum: 1,
                step: 1,
                maximum: Double(max(defaultMaxWorkers * 2, 1)),
                unit: "workers",
                description: "Maximum local CPU workers used for CPU or hybrid optimization passes."
            ),
            parameter(
                key: "chunk_size_bars",
                displayName: "Executor Chunk Size",
                kind: .integer,
                defaultValue: 128,
                minimum: 1,
                step: 1,
                maximum: 1_000_000,
                unit: "bars",
                description: "Work chunk size used by CPU and Metal executors."
            )
        ]
    }

    public static var runSettings: BacktestRunSettings {
        BacktestRunSettings(initialDeposit: 1_000, contractSize: 100_000, lotSize: 0.01)
    }

    private static func parameter(
        key: String,
        displayName: String,
        kind: FXBacktestConfigurationValueKind,
        defaultValue: Double,
        minimum: Double,
        step: Double,
        maximum: Double,
        unit: String,
        description: String
    ) -> FXBacktestConfigurationParameterDTO {
        FXBacktestConfigurationParameterDTO(
            key: key,
            displayName: displayName,
            valueKind: kind,
            defaultValue: defaultValue,
            minimum: minimum,
            step: step,
            maximum: maximum,
            unit: unit,
            description: description
        )
    }
}

public protocol FXDatabaseBacktestConfigurationAPIClient: Sendable {
    func ensureBacktestConfigurationSchema() async throws -> FXBacktestResultMutationResponse
    func registerBacktestConfiguration(_ registration: FXBacktestConfigurationRegistrationRequest) async throws -> FXBacktestResultMutationResponse
    func getBacktestConfiguration(_ request: FXBacktestConfigurationGetRequest) async throws -> FXBacktestConfigurationSnapshotResponse
}

extension FXBacktestAPIClient: FXDatabaseBacktestConfigurationAPIClient {}

public struct FXDatabaseBacktestConfigurationStore: Sendable {
    public let connection: FXDatabaseConnectionSettings
    private let client: any FXDatabaseBacktestConfigurationAPIClient

    public init(
        connection: FXDatabaseConnectionSettings = FXDatabaseConnectionSettings(),
        client: (any FXDatabaseBacktestConfigurationAPIClient)? = nil
    ) {
        self.connection = connection
        self.client = client ?? FXBacktestAPIClient(
            baseURL: connection.apiBaseURL,
            requestTimeoutSeconds: connection.requestTimeoutSeconds
        )
    }

    public func ensureSchema() async throws {
        _ = try await client.ensureBacktestConfigurationSchema()
    }

    public func register(_ registration: FXBacktestConfigurationRegistrationRequest) async throws -> FXBacktestResultMutationResponse {
        _ = try await client.ensureBacktestConfigurationSchema()
        return try await client.registerBacktestConfiguration(registration)
    }

    public func register(
        plugins: [AnyFXBacktestPlugin],
        sharedParameters: [FXBacktestConfigurationParameterDTO] = BacktestCommonConfigurationDefaults.parameters()
    ) async throws -> FXBacktestResultMutationResponse {
        try await register(Self.registrationRequest(plugins: plugins, sharedParameters: sharedParameters))
    }

    public func fetch(pluginIds: [String]? = nil) async throws -> FXBacktestConfigurationSnapshotResponse {
        try await client.getBacktestConfiguration(FXBacktestConfigurationGetRequest(pluginIds: pluginIds))
    }

    public static func registrationRequest(
        plugins: [AnyFXBacktestPlugin],
        sharedParameters: [FXBacktestConfigurationParameterDTO] = BacktestCommonConfigurationDefaults.parameters()
    ) -> FXBacktestConfigurationRegistrationRequest {
        FXBacktestConfigurationRegistrationRequest(
            sharedParameters: sharedParameters,
            pluginConfigurations: plugins.flatMap(Self.pluginConfigurations)
        )
    }

    public static func pluginConfigurations(_ plugin: AnyFXBacktestPlugin) -> [FXBacktestPluginConfigurationDTO] {
        plugin.accelerationDescriptor.supportedBackends.map { backend in
            FXBacktestPluginConfigurationDTO(
                pluginId: plugin.descriptor.id,
                acceleratorId: backend.rawValue,
                parameters: plugin.parameterDefinitions.map { $0.configurationParameterDTO() }
            )
        }
    }
}

public extension ParameterDefinition {
    func configurationParameterDTO() -> FXBacktestConfigurationParameterDTO {
        FXBacktestConfigurationParameterDTO(
            key: key,
            displayName: displayName,
            valueKind: valueKind.configurationValueKind,
            defaultValue: defaultValue,
            minimum: defaultMinimum,
            step: defaultStep,
            maximum: defaultMaximum,
            unit: "",
            description: "Plugin-owned optimization parameter \(key)."
        )
    }
}

private extension ParameterValueKind {
    var configurationValueKind: FXBacktestConfigurationValueKind {
        switch self {
        case .integer:
            return .integer
        case .decimal:
            return .decimal
        case .boolean:
            return .boolean
        }
    }
}
