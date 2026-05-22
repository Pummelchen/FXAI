import Foundation
import XCTest
@testable import FXDataEngine

final class OfflineExportTests: XCTestCase {
    func testOfflineExportSafePathsAndPriceFormatting() {
        XCTAssertEqual(OfflineExportTools.safeToken(""), "default")
        XCTAssertEqual(OfflineExportTools.safeToken("EUR/USD live:*?"), "EUR_USD_live___")
        XCTAssertEqual(
            OfflineExportTools.dataFile(outputKey: "demo key", symbol: "EUR/USD"),
            "FXAI/Offline/Exports/fxai_export_demo_key_EUR_USD.tsv"
        )
        XCTAssertEqual(OfflineExportTools.priceString(110_025, digits: 5), "1.10025")
        XCTAssertEqual(OfflineExportTools.timeText(1_704_153_720), "2024.01.02 00:02")
    }

    func testOfflineExportBuildsNewestCappedOHLCVDocuments() throws {
        let series = try Self.series()
        let document = try OfflineExportTools.buildDocument(
            series: series,
            configuration: OfflineExportConfiguration(
                outputKey: "demo",
                windowStartUTC: 1_704_153_600,
                windowEndUTC: 1_704_153_840,
                maxBars: 3
            )
        )

        XCTAssertEqual(document.barsWritten, 3)
        XCTAssertEqual(document.firstTimeUTC, 1_704_153_720)
        XCTAssertEqual(document.lastTimeUTC, 1_704_153_840)
        XCTAssertEqual(
            document.dataDocument,
            """
            time_unix\topen\thigh\tlow\tclose\tvolume\r
            1704153720\t1.10020\t1.10040\t1.10010\t1.10025\t102\r
            1704153780\t1.10030\t1.10050\t1.10020\t1.10035\t103\r
            1704153840\t1.10040\t1.10060\t1.10030\t1.10045\t104\r

            """
        )
        XCTAssertTrue(document.metadataDocument.contains("bars_written\t3\r\n"))
        XCTAssertTrue(document.metadataDocument.contains("first_time_text\t2024.01.02 00:02\r\n"))
        XCTAssertThrowsError(try OfflineExportTools.buildDocument(
            series: series,
            configuration: OfflineExportConfiguration(outputKey: "bad", windowStartUTC: 0, windowEndUTC: 60)
        ))
    }

    func testOfflineExportRepositoryWritesDataAndMetadataFiles() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("OfflineExportTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = OfflineExportRepository(rootURL: root)
        let document = try repository.write(
            series: Self.series(),
            configuration: OfflineExportConfiguration(
                outputKey: "demo",
                windowStartUTC: 1_704_153_660,
                windowEndUTC: 1_704_153_840,
                maxBars: 0
            )
        )

        let dataURL = root.appendingPathComponent(document.dataFile, isDirectory: false)
        let metadataURL = root.appendingPathComponent(document.metadataFile, isDirectory: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dataURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: metadataURL.path))
        XCTAssertEqual(try String(contentsOf: dataURL, encoding: .utf8), document.dataDocument)
        XCTAssertEqual(try String(contentsOf: metadataURL, encoding: .utf8), document.metadataDocument)
        XCTAssertEqual(document.barsWritten, 4)
    }

    private static func series() throws -> M1OHLCVSeries {
        let count = 5
        let baseTime: Int64 = 1_704_153_600
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        for index in 0..<count {
            let openValue = Int64(110_000 + index * 10)
            utc.append(baseTime + Int64(index * 60))
            open.append(openValue)
            high.append(openValue + 20)
            low.append(openValue - 10)
            close.append(openValue + 5)
            volume.append(UInt64(100 + index))
        }
        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "offline_export_tests",
                sourceOrigin: "TEST",
                logicalSymbol: "EURUSD",
                digits: 5,
                firstUTC: utc.first,
                lastUTC: utc.last
            ),
            utcTimestamps: utc,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }
}
