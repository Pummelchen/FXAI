import Foundation

public enum OfflineExportConstants {
    public static let directory = "FXAI/Offline/Exports"
}

public struct OfflineExportConfiguration: Codable, Hashable, Sendable {
    public var outputKey: String
    public var windowStartUTC: Int64
    public var windowEndUTC: Int64
    public var maxBars: Int

    public init(
        outputKey: String = "default",
        windowStartUTC: Int64,
        windowEndUTC: Int64,
        maxBars: Int = 600_000
    ) {
        self.outputKey = outputKey
        self.windowStartUTC = max(0, windowStartUTC)
        self.windowEndUTC = max(0, windowEndUTC)
        self.maxBars = max(0, maxBars)
    }
}

public struct OfflineExportDocument: Codable, Hashable, Sendable {
    public var dataFile: String
    public var metadataFile: String
    public var dataDocument: String
    public var metadataDocument: String
    public var barsWritten: Int
    public var firstTimeUTC: Int64
    public var lastTimeUTC: Int64

    public init(
        dataFile: String,
        metadataFile: String,
        dataDocument: String,
        metadataDocument: String,
        barsWritten: Int,
        firstTimeUTC: Int64,
        lastTimeUTC: Int64
    ) {
        self.dataFile = dataFile
        self.metadataFile = metadataFile
        self.dataDocument = dataDocument
        self.metadataDocument = metadataDocument
        self.barsWritten = max(0, barsWritten)
        self.firstTimeUTC = max(0, firstTimeUTC)
        self.lastTimeUTC = max(0, lastTimeUTC)
    }
}

public enum OfflineExportTools {
    public static func safeToken(_ raw: String, defaultValue: String = "default") -> String {
        var output = raw.isEmpty ? defaultValue : raw
        for character in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", " "] {
            output = output.replacingOccurrences(of: character, with: "_")
        }
        return output.isEmpty ? defaultValue : output
    }

    public static func exportStem(outputKey: String, symbol: String) -> String {
        "fxai_export_\(safeToken(outputKey))_\(safeToken(symbol))"
    }

    public static func dataFile(outputKey: String, symbol: String) -> String {
        "\(OfflineExportConstants.directory)/\(exportStem(outputKey: outputKey, symbol: symbol)).tsv"
    }

    public static func metadataFile(outputKey: String, symbol: String) -> String {
        "\(OfflineExportConstants.directory)/\(exportStem(outputKey: outputKey, symbol: symbol)).meta.tsv"
    }

    public static func buildDocument(
        series: M1OHLCVSeries,
        configuration: OfflineExportConfiguration
    ) throws -> OfflineExportDocument {
        guard configuration.windowStartUTC > 0,
              configuration.windowEndUTC > configuration.windowStartUTC else {
            throw FXDataEngineError.invalidRequest("offline export requires a positive ascending UTC window")
        }

        let selectedIndexes = selectedRows(series: series, configuration: configuration)
        guard !selectedIndexes.isEmpty else {
            throw FXDataEngineError.insufficientData("offline export found no M1 OHLCV bars in the requested window")
        }

        let symbol = series.metadata.logicalSymbol
        let dataFile = dataFile(outputKey: configuration.outputKey, symbol: symbol)
        let metadataFile = metadataFile(outputKey: configuration.outputKey, symbol: symbol)
        let data = dataDocument(series: series, indexes: selectedIndexes)
        let firstTime = series.utcTimestamps[selectedIndexes[0]]
        let lastTime = series.utcTimestamps[selectedIndexes[selectedIndexes.count - 1]]
        let metadata = metadataDocument(
            configuration: configuration,
            series: series,
            barsWritten: selectedIndexes.count,
            firstTimeUTC: firstTime,
            lastTimeUTC: lastTime
        )
        return OfflineExportDocument(
            dataFile: dataFile,
            metadataFile: metadataFile,
            dataDocument: data,
            metadataDocument: metadata,
            barsWritten: selectedIndexes.count,
            firstTimeUTC: firstTime,
            lastTimeUTC: lastTime
        )
    }

    public static func selectedRows(
        series: M1OHLCVSeries,
        configuration: OfflineExportConfiguration
    ) -> [Int] {
        let indexes = (0..<series.count).filter { index in
            let time = series.utcTimestamps[index]
            return time >= configuration.windowStartUTC && time <= configuration.windowEndUTC
        }
        guard configuration.maxBars > 0, configuration.maxBars < indexes.count else {
            return indexes
        }
        return Array(indexes.suffix(configuration.maxBars))
    }

    public static func dataDocument(series: M1OHLCVSeries, indexes: [Int]) -> String {
        var lines = ["time_unix\topen\thigh\tlow\tclose\tvolume"]
        lines.reserveCapacity(indexes.count + 1)
        for index in indexes {
            lines.append([
                String(series.utcTimestamps[index]),
                priceString(series.open[index], digits: series.metadata.digits),
                priceString(series.high[index], digits: series.metadata.digits),
                priceString(series.low[index], digits: series.metadata.digits),
                priceString(series.close[index], digits: series.metadata.digits),
                String(series.volume[index])
            ].joined(separator: "\t"))
        }
        return lines.joined(separator: "\r\n") + "\r\n"
    }

    public static func metadataDocument(
        configuration: OfflineExportConfiguration,
        series: M1OHLCVSeries,
        barsWritten: Int,
        firstTimeUTC: Int64,
        lastTimeUTC: Int64
    ) -> String {
        RuntimeArtifactTSV.document(
            header: ["key", "value"],
            rows: [
                ["output_key", safeToken(configuration.outputKey)],
                ["symbol", series.metadata.logicalSymbol],
                ["timeframe", "M1"],
                ["window_start_unix", String(configuration.windowStartUTC)],
                ["window_end_unix", String(configuration.windowEndUTC)],
                ["bars_written", String(max(0, barsWritten))],
                ["first_time_unix", String(max(0, firstTimeUTC))],
                ["last_time_unix", String(max(0, lastTimeUTC))],
                ["first_time_text", timeText(firstTimeUTC)],
                ["last_time_text", timeText(lastTimeUTC)]
            ]
        )
    }

    public static func priceString(_ scaled: Int64, digits: Int) -> String {
        let safeDigits = min(max(digits, 0), 10)
        let scale = pow(10.0, Double(safeDigits))
        return String(
            format: "%.\(safeDigits)f",
            locale: Locale(identifier: "en_US_POSIX"),
            Double(scaled) / scale
        )
    }

    public static func timeText(_ utc: Int64) -> String {
        guard utc > 0 else { return "" }
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy.MM.dd HH:mm"
        return formatter.string(from: Date(timeIntervalSince1970: TimeInterval(utc)))
    }
}

public struct OfflineExportRepository: Sendable {
    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    @discardableResult
    public func write(
        series: M1OHLCVSeries,
        configuration: OfflineExportConfiguration,
        resetOutput: Bool = true
    ) throws -> OfflineExportDocument {
        let document = try OfflineExportTools.buildDocument(series: series, configuration: configuration)
        let dataURL = rootURL.appendingPathComponent(document.dataFile, isDirectory: false)
        let metadataURL = rootURL.appendingPathComponent(document.metadataFile, isDirectory: false)
        try FileManager.default.createDirectory(
            at: dataURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if resetOutput {
            try? FileManager.default.removeItem(at: dataURL)
            try? FileManager.default.removeItem(at: metadataURL)
        }
        try document.dataDocument.write(to: dataURL, atomically: true, encoding: .utf8)
        try document.metadataDocument.write(to: metadataURL, atomically: true, encoding: .utf8)
        return document
    }
}
