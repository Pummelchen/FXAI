import FXLiveAgentCore
import Foundation

let capability = [
    "api_version": FXLiveAgentProtocolV1.latestVersion,
    "mode": "human-release-required",
    "status": "ready"
]

let data = try JSONEncoder().encode(capability)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
