import FXBacktestAgentCore
import Testing

@Suite("FXBacktestAgent protocol")
struct AgentProtocolTests {
    @Test func leaseRequestFailsClosedBeforeCertification() throws {
        let capabilities = FXBacktestAgentCapabilityProbe.localCapabilities()
        let status = FXBacktestAgentCapabilityProbe.uncertifiedStatus()
        let request = FXBacktestAgentLeaseRequest(
            maxBatches: 1,
            maxEstimatedSeconds: 10,
            capabilities: capabilities,
            certificationStatus: status
        )

        #expect(throws: FXBacktestAgentProtocolError.self) {
            try request.validateForWork()
        }
    }

    @Test func envelopeRejectsOldVersion() throws {
        let envelope = FXBacktestAgentEnvelope(
            apiVersion: "old",
            kind: .heartbeat,
            agentId: "agent",
            sequence: 1,
            payload: FXBacktestAgentHeartbeat(activeLeaseIds: [], currentLoad: 0, freeMemoryBytes: 1)
        )

        #expect(throws: FXBacktestAgentProtocolError.self) {
            try envelope.validate(expectedKind: .heartbeat)
        }
    }
}
