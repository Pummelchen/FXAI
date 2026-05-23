import FXAIPlugins
import XCTest

final class Wave4ReferenceParityTests: XCTestCase {
    func testLinearSGDReferenceMatchesSoftmaxGradientStep() {
        var weights = Array(repeating: Array(repeating: 0.0, count: 3), count: 3)
        let probabilities = LinSGDReference.trainStep(weights: &weights, features: [1.0, 0.5, -0.25], label: 1, learningRate: 0.3)
        let after = LinSGDReference.probabilities(weights: weights, features: [1.0, 0.5, -0.25])

        let expected = 1.0 / 3.0
        XCTAssertEqual(probabilities[0], expected, accuracy: 1.0e-12)
        XCTAssertEqual(probabilities[1], expected, accuracy: 1.0e-12)
        XCTAssertEqual(probabilities[2], expected, accuracy: 1.0e-12)
        XCTAssertGreaterThan(weights[1][0], 0.0)
        XCTAssertLessThan(weights[0][0], 0.0)
        XCTAssertEqual(after.reduce(0.0, +), 1.0, accuracy: 1.0e-12)
    }

    func testFTRLReferenceAppliesZNUpdatesAndL1Shrinkage() {
        var state = LinFTRLReferenceState(featureCount: 2, alpha: 0.2, beta: 1.0, l1: 0.0, l2: 0.1)
        let prediction = LinFTRLReference.update(state: &state, features: [1.0, 2.0], label: 1.0)
        let weights = LinFTRLReference.weights(state: state)
        let sparse = LinFTRLReferenceState(featureCount: 2, alpha: 0.2, beta: 1.0, l1: 100.0, l2: 0.1)

        XCTAssertEqual(prediction, 0.5, accuracy: 1.0e-12)
        XCTAssertTrue(state.n.allSatisfy { $0 > 0.0 })
        XCTAssertTrue(weights.contains { abs($0) > 0.0 })
        XCTAssertEqual(LinFTRLReference.weights(state: sparse), [0.0, 0.0])
    }

    func testPassiveAggressiveReferenceUsesCrammerSingerMargin() {
        var weights = Array(repeating: Array(repeating: 0.0, count: 2), count: 3)
        let update = LinPAReference.update(weights: &weights, features: [2.0, 0.0], label: 2, mode: .paI(c: 0.5))

        XCTAssertEqual(update.predicted, 0)
        XCTAssertEqual(update.loss, 1.0, accuracy: 1.0e-12)
        XCTAssertGreaterThan(update.tau, 0.0)
        XCTAssertGreaterThan(weights[2][0], 0.0)
        XCTAssertLessThan(weights[0][0], 0.0)
    }

    func testElasticAndProfitLogitReferencesExposeCanonicalLosses() {
        var weights = [0.02, -0.01]
        var bias = 0.0
        let prediction = LinElasticLogitReference.trainStep(
            weights: &weights,
            bias: &bias,
            features: [0.1, 0.1],
            label: 1.0,
            learningRate: 0.1,
            l1: 1.0,
            l2: 0.0
        )
        let buyLoss = LinProfitLogitReference.lossGradient(weights: [0.1, -0.2], bias: 0.0, features: [1.0, 1.0], label: 1.0, movePoints: 5.0, costPoints: 1.0, buyAsymmetry: 2.0, sellAsymmetry: 1.0)
        let sellLoss = LinProfitLogitReference.lossGradient(weights: [0.1, -0.2], bias: 0.0, features: [1.0, 1.0], label: 0.0, movePoints: 5.0, costPoints: 1.0, buyAsymmetry: 2.0, sellAsymmetry: 1.0)

        XCTAssertGreaterThan(prediction, 0.0)
        XCTAssertEqual(weights, [0.0, 0.0])
        XCTAssertGreaterThan(buyLoss.effectiveWeight, sellLoss.effectiveWeight)
        XCTAssertEqual(buyLoss.gradient.count, 2)
    }

    func testENHashReferenceIsDeterministicAndReportsCollisions() {
        let fields = [0..<2, 2..<4, 4..<6]
        let first = LinEnhashReference.interactions(features: [1, 2, 3, 4, 5, 6], fieldRanges: fields, bucketCount: 4, seed: 11)
        let second = LinEnhashReference.interactions(features: [1, 2, 3, 4, 5, 6], fieldRanges: fields, bucketCount: 4, seed: 11)

        XCTAssertEqual(first, second)
        XCTAssertEqual(first.count, 6)
        XCTAssertGreaterThan(LinEnhashReference.collisionRate(interactions: first), 0.0)
        XCTAssertEqual(first[0].value, 2.25, accuracy: 1.0e-12)
    }

    func testQuantileAndMemoryReferencesCoverPinballProjectionTopKAndEviction() {
        XCTAssertEqual(DistQuantileReference.pinballLoss(prediction: 8.0, target: 10.0, quantile: 0.75), 1.5, accuracy: 1.0e-12)
        XCTAssertEqual(DistQuantileReference.monotonicProjection([1.0, 0.8, 1.2, 1.1]), [1.0, 1.0, 1.2, 1.2])
        XCTAssertEqual(DistQuantileReference.coverage(predictions: [1, 2, 3], targets: [0, 3, 2]), 2.0 / 3.0, accuracy: 1.0e-12)

        let memory = [
            MemRetrdiffReference.Item(id: "old", vector: [0.0, 0.0], label: -1.0, timestamp: 1),
            MemRetrdiffReference.Item(id: "near", vector: [0.2, 0.1], label: 1.0, timestamp: 2),
            MemRetrdiffReference.Item(id: "new", vector: [2.0, 2.0], label: 0.0, timestamp: 3)
        ]
        let neighbors = MemRetrdiffReference.topK(query: [0.0, 0.0], memory: memory, k: 2)
        let evicted = MemRetrdiffReference.evict(memory: memory, capacity: 2)

        XCTAssertEqual(neighbors.map { $0.item.id }, ["old", "near"])
        XCTAssertEqual(evicted.map(\.id), ["near", "new"])
    }

    func testXGBReferenceGainMissingRoutingAndMulticlassGradients() {
        let gain = TreeXGBFastReference.gain(leftGradient: -3.0, leftHessian: 4.0, rightGradient: 2.0, rightHessian: 3.0, lambda: 1.0, gamma: 0.0)
        let gradient = TreeXGBReference.multiclassGradient(logits: [0.2, 1.2, -0.3], label: 1)

        XCTAssertGreaterThan(gain.totalGain, 0.0)
        XCTAssertEqual(TreeXGBFastReference.leafWeight(gradient: -3.0, hessian: 4.0, lambda: 1.0), 0.6, accuracy: 1.0e-12)
        XCTAssertTrue(TreeXGBFastReference.missingGoesLeft(leftGradient: -3.0, leftHessian: 4.0, rightGradient: 2.0, rightHessian: 3.0, missingGradient: -1.0, missingHessian: 1.0, lambda: 1.0, gamma: 0.0))
        XCTAssertEqual(gradient.probabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-12)
        XCTAssertEqual(gradient.gradients.reduce(0.0, +), 0.0, accuracy: 1.0e-12)
        XCTAssertTrue(TreeXGBReference.route(value: nil, threshold: 0.5, missingGoesLeft: true))
    }

    func testLightGBMReferenceHistogramBestSplitAndDARTMask() {
        let hist = TreeLgbmReference.histogram(
            feature: [0.1, 0.2, 0.8, 0.9],
            gradients: [-1.0, -1.0, 1.0, 1.0],
            hessians: [1.0, 1.0, 1.0, 1.0],
            binCount: 2
        )
        let split = TreeLgbmReference.bestSplit(histogram: hist, lambda: 1.0, minDataInLeaf: 1)
        let maskA = TreeLgbmReference.dartKeepMask(treeCount: 8, dropRate: 0.25, seed: 7)
        let maskB = TreeLgbmReference.dartKeepMask(treeCount: 8, dropRate: 0.25, seed: 7)

        XCTAssertEqual(hist.map(\.count).reduce(0, +), 4)
        XCTAssertNotNil(split)
        XCTAssertGreaterThan(split?.gain ?? 0.0, 0.0)
        XCTAssertEqual(maskA, maskB)
    }

    func testCatboostReferenceOrderedCTRPreventsTargetLeakage() {
        let ctr = TreeCatboostReference.orderedCTR(
            categories: ["A", "A", "B", "A"],
            labels: [1.0, 0.0, 1.0, 1.0],
            permutation: [0, 1, 2, 3],
            prior: 0.5,
            priorWeight: 1.0
        )

        XCTAssertEqual(ctr[0].value, 0.5, accuracy: 1.0e-12)
        XCTAssertEqual(ctr[1].value, 0.75, accuracy: 1.0e-12)
        XCTAssertEqual(TreeCatboostReference.symmetricLeafIndex(features: [0.2, 1.5], featureIndexes: [0, 1], thresholds: [0.5, 1.0]), 2)
    }

    func testRandomForestReferenceBootstrapOOBAndGiniSplit() {
        let bootstrapA = TreeRFReference.bootstrap(rowCount: 10, sampleCount: 10, seed: 42)
        let bootstrapB = TreeRFReference.bootstrap(rowCount: 10, sampleCount: 10, seed: 42)
        let split = TreeRFReference.bestGiniSplit(feature: [0.1, 0.2, 0.8, 0.9], labels: [0, 0, 1, 1], minLeaf: 1)

        XCTAssertEqual(bootstrapA, bootstrapB)
        XCTAssertEqual(bootstrapA.sample.count, 10)
        XCTAssertTrue(bootstrapA.outOfBag.allSatisfy { !bootstrapA.sample.contains($0) })
        XCTAssertNotNil(split)
        XCTAssertLessThan(split?.impurity ?? 1.0, 0.10)
    }
}
