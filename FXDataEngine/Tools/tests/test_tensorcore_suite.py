from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _assert_tokens(text: str, tokens: list[str]) -> None:
    for token in tokens:
        assert token in text


def test_tensor_work_is_declared_as_plugin_backend_bridge_not_mql_tensorcore():
    ml_backend = _read("FXDataEngine/Sources/FXDataEngine/Core/MLBackend.swift")
    plugins_readme = _read("FXPlugins/README.md")

    _assert_tokens(
        ml_backend,
        [
            "public enum MLFramework: String, Codable, Hashable, Sendable",
            "case pyTorch",
            "case tensorFlow",
            "public enum MLBackendMode",
            "case externalPython(framework: MLFramework, executable: String, module: String)",
            "public protocol ExternalMLBackend",
            "public struct PythonMLBackendBridge",
            "precondition(",
            "framework == .pyTorch",
            "framework == .tensorFlow",
        ],
    )
    _assert_tokens(
        plugins_readme,
        [
            "When a Swift plugin needs tensor training or inference",
            "plugin-local PyTorch or TensorFlow backend",
            "FXDataEngine remains responsible for deterministic feature and payload contracts",
        ],
    )


def test_tensor_payloads_preserve_ohlcv_volume_contract():
    ml_backend = _read("FXDataEngine/Sources/FXDataEngine/Core/MLBackend.swift")
    contracts = _read("FXDataEngine/Sources/FXDataEngine/Plugins/PluginContracts.swift")

    _assert_tokens(
        ml_backend,
        [
            "public let dataHasVolume: Bool",
            "public let nextVolumeTarget: Double",
            "dataHasVolume: request.context.dataHasVolume",
            "self.nextVolumeTarget = request.nextVolumeTarget",
            "public static func inferencePayload(",
        ],
    )
    _assert_tokens(
        contracts,
        [
            "public var dataHasVolume: Bool",
            "public let nextVolumeTarget: Double",
            "guard nextVolumeTarget.isFinite, nextVolumeTarget >= 0 else",
        ],
    )


def test_swift_package_owns_the_new_tensor_boundary():
    package = _read("FXDataEngine/Package.swift")
    cli = _read("FXDataEngine/Sources/FXDataEngineCLI/main.swift")

    _assert_tokens(
        package,
        [
            "// swift-tools-version: 6.3",
            ".macOS(\"26.0\")",
            ".library(name: \"FXDataEngine\", targets: [\"FXDataEngine\"])",
            ".executable(name: \"FXDataEngineCLI\", targets: [\"FXDataEngineCLI\"])",
            "swiftLanguageModes: [.v6]",
        ],
    )
    _assert_tokens(
        cli,
        [
            "MetalAccelerationDevice.probe()",
            "FeatureCore.hasUsableVolume(",
            "OHLCV contract active",
            "Metal available=",
        ],
    )
