import XCTest
@testable import FXDataEngine

final class FXAIPluginImplementationDescriptorTests: XCTestCase {
    func testReferenceDescriptorFactoriesDoNotDeclareCoreMLBeforeRuntimeExists() {
        let descriptors: [FXAIPluginImplementationDescriptor] = [
            .linear(.sgdLogit, "lin_sgd"),
            .tree(.xgboost, "tree_xgb"),
            .sequence(
                .lstm,
                "ai_lstm",
                .recurrent,
                [.swiftScalar, .pyTorchMPS],
                [.tensorFlowMetal]
            ),
            .distribution(.quantile, "dist_quantile"),
            .statistical(.statARIMAXGARCH, "stat_arimax_garch"),
            .factor(.factorCarry, "factor_carry"),
            .trend(.trendTSMOMVol, "trend_tsmom_vol"),
            .mixture(.moeConformal, "mix_moe_conformal"),
            .memory(.retrDiff, "mem_retrdiff"),
            .world(.cfxWorld, "wm_cfx"),
            .reinforcement(.rlPPO, "rl_ppo")
        ]

        for descriptor in descriptors {
            XCTAssertFalse(
                descriptor.accelerationPlan.declares(.coreMLNeuralEngine),
                "\(descriptor.aiName) must not declare CoreML until export/load/predict/parity runtime exists."
            )
        }
    }
}
