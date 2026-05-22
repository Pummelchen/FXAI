import FXDataEngine
import Foundation

enum StatPluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.statistical(.statMSGARCH, "stat_msgarch"),
        FXAIPluginDefinitionFactory.statistical(.statARIMAXGARCH, "stat_arimax_garch"),
        FXAIPluginDefinitionFactory.statistical(.statCointVECM, "stat_coint_vecm"),
        FXAIPluginDefinitionFactory.statistical(.statOUSpread, "stat_ou_spread"),
        FXAIPluginDefinitionFactory.statistical(.statMicroflowProxy, "stat_microflow_proxy"),
        FXAIPluginDefinitionFactory.statistical(.statHMMRegime, "stat_hmm_regime"),
        FXAIPluginDefinitionFactory.statistical(.statEMDHHT, "stat_emd_hht", [.accelerate], [.metal]),
        FXAIPluginDefinitionFactory.statistical(.statVMD, "stat_vmd", [.accelerate], [.metal]),
        FXAIPluginDefinitionFactory.statistical(.statTVPKalman, "stat_tvp_kalman"),
        FXAIPluginDefinitionFactory.statistical(.statXRateConsistency, "stat_xrate_consistency")
    ]
}
