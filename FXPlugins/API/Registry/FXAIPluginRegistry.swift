import FXDataEngine
import Foundation

public enum FXAIPluginRegistry {
    public static func availablePlugins() -> [any FXAIPluginV4] {
        let plugins: [any FXAIPluginV4] = [
            DistQuantilePlugin(),
            FactorPCAPanelPlugin(),
            FactorPPPValuePlugin(),
            FactorCarryPlugin(),
            FactorCMVPanelPlugin(),
            LinEnhashPlugin(),
            LinFTRLPlugin(),
            LinPAPlugin(),
            LinSGDPlugin(),
            LinElasticLogitPlugin(),
            LinProfitLogitPlugin(),
            MemRetrdiffPlugin(),
            MixLoffmPlugin(),
            MixMoeConformalPlugin(),
            RlPPOPlugin(),
            AIAutoformerPlugin(),
            AIChronosPlugin(),
            AIGeodesicPlugin(),
            AILSTMPlugin(),
            AILSTMGPlugin(),
            AIMLPPlugin(),
            AIPatchtstPlugin(),
            AIS4Plugin(),
            AIStmnPlugin(),
            AITCNPlugin(),
            AITFTPlugin(),
            AITimesfmPlugin(),
            AITSTPlugin(),
            AITRRPlugin(),
            AIQCEWPlugin(),
            AIFEWCPlugin(),
            AIGHAPlugin(),
            AITesseractPlugin(),
            AICNNLSTMPlugin(),
            AIAttnCNNBiLSTMPlugin(),
            AIGRUPlugin(),
            AIBiLSTMPlugin(),
            AILSTMTCNPlugin(),
            AIMythosRDTPlugin(),
            StatMSGARCHPlugin(),
            StatARIMAXGARCHPlugin(),
            StatCointVECMPlugin(),
            StatOUSpreadPlugin(),
            StatMicroflowProxyPlugin(),
            StatHMMRegimePlugin(),
            StatEMDHHTPlugin(),
            StatVMDPlugin(),
            StatTVPKalmanPlugin(),
            StatXrateConsistencyPlugin(),
            TreeCatboostPlugin(),
            TreeLgbmPlugin(),
            TreeXGBFastPlugin(),
            TreeXGBPlugin(),
            TreeRFPlugin(),
            TrendTSMOMVolPlugin(),
            TrendXSMOMRankPlugin(),
            TrendVolBreakoutPlugin(),
            WMCFXPlugin(),
            WMGraphPlugin(),
            RuleBuyOnlyPlugin(),
            RuleSellOnlyPlugin(),
            RuleRandomPlugin(),
            RuleM1SyncPlugin(),
            MovingAverageCrossFXDataEnginePlugin(),
            FXStupidFXDataEnginePlugin()
        ]
        return plugins.sorted { $0.manifest.aiID < $1.manifest.aiID }
    }

    public static func accelerationPlans() -> [FXPluginAccelerationPlan] {
        availablePlugins()
            .compactMap { ($0 as? any FXAIPlannedPlugin)?.accelerationPlan }
            .sorted { $0.pluginName < $1.pluginName }
    }
}
