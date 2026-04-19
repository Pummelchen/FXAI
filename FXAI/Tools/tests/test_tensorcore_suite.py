from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_tensorcore_suite_is_promoted_to_dedicated_test_surface():
    suite = _read("Tests/TensorCore/tensorcore_suite.mqh")
    required_tokens = [
        "void FXAI_TensorCoreRunSuite(FXAITestSuiteResult &suite)",
        "FXAI_TensorCoreTestKernelBlocks(",
        "FXAI_TensorCoreTestAdamWConvergence(",
        "FXAI_TensorCoreTestRMSPropConvergence(",
        "tensor_ops_sequence_and_normalization",
        "optimizer_adamw_convergence",
        "optimizer_rmsprop_convergence",
    ]
    for token in required_tokens:
        assert token in suite


def test_tensorcore_runner_and_audit_runner_wire_the_suite():
    runner = _read("Tests/FXAI_TensorCoreRunner.mq5")
    audit = _read("Tests/FXAI_AuditRunner.mq5")
    audit_wrapper = _read("Tests/audit_tensor.mqh")
    harness = _read("Tests/TestHarness/test_harness.mqh")

    assert '#include "audit_core.mqh"' in runner
    assert "FXAI_TestWriteCombinedReport(" in runner
    assert "Audit_RunTensorKernelSanity" in audit
    assert "FXAI_AuditTensorKernelSelfTest(" in audit
    assert "FXAI_TensorCoreRunSuite(suite);" in audit_wrapper
    assert "struct FXAITestSuiteResult" in harness
    assert "void FXAI_TestSuiteAppendJson(" in harness


def test_testlab_exposes_tensorcore_compile_gate():
    cli = _read("Tools/testlab/cli.py")
    verify = _read("Tools/testlab/verify.py")

    assert 'compile-tensorcore' in cli
    assert 'Path("Tests/FXAI_TensorCoreRunner.mq5")' in cli
    assert '"compile_tensorcore"' in verify
    assert 'Path("Tests/FXAI_TensorCoreRunner.mq5")' in verify
