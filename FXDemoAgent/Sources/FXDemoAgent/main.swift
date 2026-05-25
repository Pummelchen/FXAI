import FXDemoAgentCore
import Foundation

let capability = [
    "api_version": FXDemoAgentProtocolV1.latestVersion,
    "mode": "dry-run-first",
    "status": "ready"
]

let data = try JSONEncoder().encode(capability)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
