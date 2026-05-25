import FXBacktestAgentCore
import Foundation

let capabilities = FXBacktestAgentCapabilityProbe.localCapabilities()
let status = FXBacktestAgentCapabilityProbe.uncertifiedStatus()
let leaseRequest = FXBacktestAgentLeaseRequest(
    maxBatches: 1,
    maxEstimatedSeconds: 60,
    capabilities: capabilities,
    certificationStatus: status
)

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

if CommandLine.arguments.contains("--self-check") {
    do {
        try leaseRequest.validateForWork()
        print("FXBacktestAgent self-check passed")
    } catch {
        FileHandle.standardError.write(Data("FXBacktestAgent self-check failed closed: \(error)\n".utf8))
        Foundation.exit(1)
    }
}

let envelope = FXBacktestAgentEnvelope(
    kind: .leaseRequest,
    agentId: capabilities.hostName,
    sequence: 1,
    payload: leaseRequest
)
let data = try encoder.encode(envelope)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
