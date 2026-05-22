import Foundation

public struct TestSuiteCaseResult: Codable, Hashable, Sendable {
    public var name: String
    public var passed: Bool
    public var reason: String

    public init(name: String, passed: Bool, reason: String = "") {
        self.name = name
        self.passed = passed
        self.reason = reason
    }
}

public struct TestSuiteResult: Codable, Hashable, Sendable {
    public var suiteName: String
    public var cases: [TestSuiteCaseResult]

    public init(suiteName: String = "", cases: [TestSuiteCaseResult] = []) {
        self.suiteName = suiteName
        self.cases = cases
    }

    public var total: Int {
        cases.count
    }

    public var failed: Int {
        cases.reduce(0) { $0 + ($1.passed ? 0 : 1) }
    }

    public var passed: Bool {
        total > 0 && failed == 0
    }

    public var legacyReason: String {
        guard let failedCase = cases.first(where: { !$0.passed }) else {
            return ""
        }
        return failedCase.reason.isEmpty ? failedCase.name : "\(failedCase.name):\(failedCase.reason)"
    }

    public mutating func reset(suiteName: String) {
        self.suiteName = suiteName
        cases.removeAll(keepingCapacity: false)
    }

    public mutating func addCase(name: String, passed: Bool, reason: String = "") {
        cases.append(TestSuiteCaseResult(name: name, passed: passed, reason: reason))
    }

    public func jsonDocument() -> String {
        var json = "{"
        json += "\"suite_name\":\"\(TestSuiteTools.jsonEscape(suiteName))\","
        json += "\"total\":\(total),"
        json += "\"failed\":\(failed),"
        json += "\"passed\":\(passed ? "true" : "false"),"
        json += "\"cases\":["
        for (index, item) in cases.enumerated() {
            if index > 0 {
                json += ","
            }
            json += "{"
            json += "\"name\":\"\(TestSuiteTools.jsonEscape(item.name))\","
            json += "\"passed\":\(item.passed ? "true" : "false"),"
            json += "\"reason\":\"\(TestSuiteTools.jsonEscape(item.reason))\""
            json += "}"
        }
        json += "]"
        json += "}"
        return json
    }
}

public enum TestSuiteTools {
    public static func reset(_ suiteName: String) -> TestSuiteResult {
        TestSuiteResult(suiteName: suiteName)
    }

    public static func jsonEscape(_ value: String) -> String {
        var escaped = ""
        escaped.reserveCapacity(value.count)
        for character in value {
            switch character {
            case "\\":
                escaped += "\\\\"
            case "\"":
                escaped += "\\\""
            case "\r":
                escaped += "\\r"
            case "\n":
                escaped += "\\n"
            case "\t":
                escaped += "\\t"
            default:
                escaped.append(character)
            }
        }
        return escaped
    }
}
