import FXDataEngine
import Foundation

@main
struct FXDataEngineCLI {
    static func main() throws {
        let device = MetalAccelerationDevice.probe()
        let volumeStatus = try FeatureCore.hasUsableVolume(MarketUniverse(series: [.demoEURUSD(barCount: 300)]))
        print("FXDataEngine API v\(FXDataEngineConstants.apiVersionV4) features=\(FXDataEngineConstants.aiFeatures) weights=\(FXDataEngineConstants.aiWeights) maxSequenceBars=\(FXDataEngineConstants.maxSequenceBars)")
        print("OHLCV contract active, demoVolumeAvailable=\(volumeStatus)")
        print("Metal available=\(device.available) device=\(device.deviceName ?? "none")")
    }
}
