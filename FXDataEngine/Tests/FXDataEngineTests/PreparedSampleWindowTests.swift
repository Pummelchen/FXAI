import XCTest
@testable import FXDataEngine

final class PreparedSampleWindowTests: XCTestCase {
    func testPreparedSampleWindowUsesOlderAscendingRowsAndStopsOnInvalidSamples() {
        var samples = (0..<6).map { sample(index: $0, valid: true) }
        var window = TrainingSampleTools.preparedSampleWindow(samples: samples, anchorIndex: 4, requestedBars: 3)
        XCTAssertEqual(window.size, 3)
        XCTAssertEqual(window.rows.map { $0[1] }, [3.0, 2.0, 1.0])

        samples[2] = sample(index: 2, valid: false)
        window = TrainingSampleTools.preparedSampleWindow(samples: samples, anchorIndex: 4, requestedBars: 3)
        XCTAssertEqual(window.size, 1)
        XCTAssertEqual(window.rows.map { $0[1] }, [3.0])
    }

    func testPreparedSampleWindowClampsRequestedBarsAndHandlesBoundaryAnchors() {
        let samples = (0..<(FXDataEngineConstants.maxSequenceBars + 4)).map { sample(index: $0, valid: true) }

        let single = TrainingSampleTools.preparedSampleWindow(samples: samples, anchorIndex: 2, requestedBars: 0)
        XCTAssertEqual(single.size, 1)
        XCTAssertEqual(single.rows.map { $0[1] }, [1.0])

        let capped = TrainingSampleTools.preparedSampleWindow(
            samples: samples,
            anchorIndex: samples.count - 1,
            requestedBars: FXDataEngineConstants.maxSequenceBars + 20
        )
        XCTAssertEqual(capped.size, FXDataEngineConstants.maxSequenceBars)
        XCTAssertEqual(capped.rows.first?[1], Double(samples.count - 2))
        XCTAssertEqual(capped.rows.last?[1], Double(samples.count - 1 - FXDataEngineConstants.maxSequenceBars))

        XCTAssertEqual(
            TrainingSampleTools.preparedSampleWindow(samples: samples, anchorIndex: 0, requestedBars: 4).size,
            0
        )
        XCTAssertEqual(
            TrainingSampleTools.preparedSampleWindow(samples: samples, anchorIndex: samples.count, requestedBars: 4).size,
            0
        )
    }

    private func sample(index: Int, valid: Bool) -> PreparedTrainingSample {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[1] = Double(index)
        x[FXDataEngineConstants.aiWeights - 1] = Double(index) * 0.5
        return PreparedTrainingSample(valid: valid, x: x)
    }
}
