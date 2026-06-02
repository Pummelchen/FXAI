import FXBacktestCore
import XCTest

final class ParameterSweepTests: XCTestCase {
    func testSweepCountsAndIndexesCombinationsWithoutMaterializingAll() throws {
        let fast = try ParameterDefinition(
            key: "fast",
            displayName: "Fast",
            defaultValue: 2,
            defaultMinimum: 2,
            defaultStep: 1,
            defaultMaximum: 4,
            valueKind: .integer
        )
        let slow = try ParameterDefinition(
            key: "slow",
            displayName: "Slow",
            defaultValue: 10,
            defaultMinimum: 10,
            defaultStep: 5,
            defaultMaximum: 20,
            valueKind: .integer
        )

        let sweep = try ParameterSweep(dimensions: [
            try ParameterSweepDimension(definition: fast, input: 2, minimum: 2, step: 1, maximum: 4),
            try ParameterSweepDimension(definition: slow, input: 10, minimum: 10, step: 5, maximum: 20)
        ])

        XCTAssertEqual(sweep.combinationCount, 9)
        XCTAssertEqual(try sweep.parameterVector(at: 0).values, [2, 10])
        XCTAssertEqual(try sweep.parameterVector(at: 1).values, [3, 10])
        XCTAssertEqual(try sweep.parameterVector(at: 3).values, [2, 15])
        XCTAssertEqual(try sweep.parameterVector(at: 8).values, [4, 20])
    }

    func testRejectsInvalidAndNonFiniteParameterDimensions() throws {
        let integer = try ParameterDefinition(
            key: "period",
            displayName: "Period",
            defaultValue: 2,
            defaultMinimum: 1,
            defaultStep: 1,
            defaultMaximum: 10,
            valueKind: .integer
        )
        XCTAssertThrowsError(try ParameterSweepDimension(
            definition: integer,
            input: 2.5,
            minimum: 1,
            step: 1,
            maximum: 10
        ))
        XCTAssertThrowsError(try ParameterSweepDimension(
            definition: integer,
            input: 2,
            minimum: 1,
            step: .infinity,
            maximum: 10
        ))

        let decimal = try ParameterDefinition(
            key: "lots",
            displayName: "Lots",
            defaultValue: 0.1,
            defaultMinimum: 0.1,
            defaultStep: 0.1,
            defaultMaximum: 1,
            valueKind: .decimal
        )
        XCTAssertThrowsError(try ParameterSweepDimension(
            definition: decimal,
            input: 0.1,
            minimum: 0.1,
            step: 0,
            maximum: 1
        ))
    }

    func testDecimalSweepCountsWholeStepRangesWithRelativeTolerance() throws {
        let decimal = try ParameterDefinition(
            key: "threshold",
            displayName: "Threshold",
            defaultValue: 0,
            defaultMinimum: 0,
            defaultStep: 0.1,
            defaultMaximum: 0.3,
            valueKind: .decimal
        )

        let dimension = try ParameterSweepDimension(
            definition: decimal,
            input: 0,
            minimum: 0,
            step: 0.1,
            maximum: 0.3
        )

        XCTAssertEqual(dimension.valueCount, 4)
        XCTAssertEqual(dimension.value(at: 3), 0.3, accuracy: 0.0000000001)
    }

    func testDecimalSweepDoesNotEmitClampedDuplicateEndpointWhenStepOvershoots() throws {
        let decimal = try ParameterDefinition(
            key: "fraction",
            displayName: "Fraction",
            defaultValue: 0,
            defaultMinimum: 0,
            defaultStep: 0.34,
            defaultMaximum: 1,
            valueKind: .decimal
        )

        let dimension = try ParameterSweepDimension(
            definition: decimal,
            input: 0,
            minimum: 0,
            step: 0.34,
            maximum: 1
        )

        XCTAssertEqual(dimension.valueCount, 3)
        XCTAssertEqual(dimension.value(at: 2), 0.68, accuracy: 0.0000000001)
    }
}
