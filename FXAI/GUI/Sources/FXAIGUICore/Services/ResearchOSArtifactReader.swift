import Foundation

public struct ResearchOSArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> ResearchOSControlSnapshot {
        let researchRoot = projectRoot.appendingPathComponent("Tools/OfflineLab/ResearchOS", isDirectory: true)
        guard
            let dashboardURL = latestFile(named: "operator_dashboard.json", under: researchRoot),
            let dashboard = parseJSON(dashboardURL)
        else {
            return ResearchOSControlSnapshot(
                generatedAt: Date(),
                profileName: nil,
                environment: nil,
                branches: [],
                auditEvents: [],
                symbols: [],
                sourceOfTruth: []
            )
        }

        let turso = dashboard["turso"] as? [String: Any] ?? [:]
        let environment = parseEnvironment(turso["environment"] as? [String: Any] ?? [:])
        let branches = parseBranches(turso["branches"] as? [[String: Any]] ?? [])
        let auditEvents = parseAuditEvents(turso["recent_audit_logs"] as? [[String: Any]] ?? [])
        let symbols = parseSymbolControls(dashboard["deployments"] as? [[String: Any]] ?? [])
        let sourceOfTruth = (dashboard["source_of_truth"] as? [String: Any] ?? [:])
            .map { KeyValueRecord(key: $0.key, value: stringify($0.value)) }
            .sorted { $0.key < $1.key }

        return ResearchOSControlSnapshot(
            generatedAt: Date(),
            profileName: dashboard["profile_name"] as? String,
            environment: environment,
            branches: branches,
            auditEvents: auditEvents,
            symbols: symbols
                .sorted { $0.symbol < $1.symbol },
            sourceOfTruth: sourceOfTruth
        )
    }

    private func parseEnvironment(_ raw: [String: Any]) -> ResearchOSEnvironmentStatus? {
        guard !raw.isEmpty else { return nil }

        return ResearchOSEnvironmentStatus(
            backend: raw["backend"] as? String ?? "unknown",
            syncMode: raw["sync_mode"] as? String ?? "unknown",
            databasePath: (raw["database_path"] as? String).flatMap(nonEmptyURL),
            databaseName: raw["database_name"] as? String ?? "",
            organizationSlug: raw["organization_slug"] as? String ?? "",
            groupName: raw["group_name"] as? String ?? "",
            locationName: raw["location_name"] as? String ?? "",
            cliConfigPath: (raw["cli_config_path"] as? String).flatMap(nonEmptyURL),
            syncIntervalSeconds: intValue(raw["sync_interval_seconds"]),
            encryptionEnabled: raw["encryption_enabled"] as? Bool ?? false,
            platformAPIEnabled: raw["platform_api_enabled"] as? Bool ?? false,
            syncEnabled: raw["sync_enabled"] as? Bool ?? false,
            authTokenConfigured: raw["auth_token_configured"] as? Bool ?? false,
            apiTokenConfigured: raw["api_token_configured"] as? Bool ?? false,
            configError: nonEmptyString(raw["config_error"] as? String)
        )
    }

    private func parseBranches(_ rows: [[String: Any]]) -> [ResearchOSBranchRecord] {
        rows.compactMap { row in
            let name = stringValue(row["target_database"]) ?? stringValue(row["name"]) ?? ""
            let parentName = stringValue(row["parent_name"]) ?? ""
            let sourceDatabase = stringValue(row["source_database"]) ?? parentName

            if name.isEmpty && sourceDatabase.isEmpty && parentName.isEmpty {
                return nil
            }

            return ResearchOSBranchRecord(
                name: name.isEmpty ? sourceDatabase : name,
                sourceDatabase: sourceDatabase,
                parentName: parentName,
                branchKind: stringValue(row["branch_kind"]) ?? (boolValue(row["is_branch"]) ? "branch" : "database"),
                status: stringValue(row["status"]) ?? (boolValue(row["is_branch"]) ? "active" : "unknown"),
                groupName: stringValue(row["group_name"]) ?? stringValue(row["group"]) ?? "",
                locationName: stringValue(row["location_name"]) ?? "",
                hostname: stringValue(row["hostname"]) ?? "",
                syncURL: stringValue(row["sync_url"]) ?? "",
                envArtifactPath: stringValue(row["env_artifact_path"]).flatMap(nonEmptyURL),
                isBranch: boolValue(row["is_branch"]) || !parentName.isEmpty || !(stringValue(row["branch_kind"]) ?? "").isEmpty,
                createdAt: parseUnix(row["created_at"]),
                sourceTimestamp: stringValue(row["source_timestamp"]) ?? ""
            )
        }
        .sorted { lhs, rhs in
            if lhs.createdAt == rhs.createdAt {
                return lhs.name < rhs.name
            }
            return (lhs.createdAt ?? .distantPast) > (rhs.createdAt ?? .distantPast)
        }
    }

    private func parseAuditEvents(_ rows: [[String: Any]]) -> [ResearchOSAuditEvent] {
        rows.compactMap { row in
            let eventID = stringValue(row["event_id"]) ?? ""
            let eventType = stringValue(row["event_type"]) ?? ""
            let organization = stringValue(row["organization_slug"]) ?? ""
            if eventID.isEmpty && eventType.isEmpty && organization.isEmpty {
                return nil
            }

            return ResearchOSAuditEvent(
                organizationSlug: organization,
                eventID: eventID,
                eventType: eventType,
                targetName: stringValue(row["target_name"]) ?? "",
                occurredAt: parseUnix(row["occurred_at"]),
                observedAt: parseUnix(row["observed_at"])
            )
        }
    }

    private func parseSymbolControls(_ rows: [[String: Any]]) -> [ResearchOSSymbolControl] {
        rows.compactMap { row in
            let symbol = stringValue(row["symbol"]) ?? ""
            if symbol.isEmpty {
                return nil
            }

            let analogRows = row["analog_neighbors"] as? [[String: Any]] ?? []
            let analogNeighbors = analogRows.compactMap(parseNeighbor)

            return ResearchOSSymbolControl(
                symbol: symbol,
                analogNeighbors: analogNeighbors,
                deploymentArtifactPath: stringValue(row["artifact_path"]).flatMap(nonEmptyURL),
                deploymentCreatedAt: parseUnix(row["created_at"])
            )
        }
    }

    private func parseNeighbor(_ raw: [String: Any]) -> ResearchOSAnalogNeighbor? {
        let sourceKey = stringValue(raw["source_key"]) ?? ""
        let pluginName = stringValue(raw["plugin_name"]) ?? ""
        if sourceKey.isEmpty && pluginName.isEmpty {
            return nil
        }

        let payload = (raw["payload"] as? [String: Any] ?? [:])
            .map { KeyValueRecord(key: $0.key, value: stringify($0.value)) }
            .sorted { $0.key < $1.key }

        let distance = doubleValue(raw["cosine_distance"])
        let similarity = raw["similarity"] as? Double ?? max(0, 1 - distance)

        return ResearchOSAnalogNeighbor(
            sourceKey: sourceKey,
            pluginName: pluginName,
            distance: distance,
            similarity: similarity,
            score: doubleValue(raw["score"]),
            sourceType: stringValue(raw["source_type"]) ?? "",
            scope: stringValue(raw["vector_scope"]) ?? "",
            payload: payload
        )
    }

    private func latestFile(named name: String, under root: URL) -> URL? {
        guard FileManager.default.fileExists(atPath: root.path) else { return nil }

        let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )

        var latest: (url: URL, modifiedAt: Date)?

        while let item = enumerator?.nextObject() as? URL {
            guard item.lastPathComponent == name else { continue }
            let modifiedAt = (try? item.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
            if latest == nil || modifiedAt > latest?.modifiedAt ?? .distantPast {
                latest = (item, modifiedAt)
            }
        }

        return latest?.url
    }

    private func parseJSON(_ url: URL) -> [String: Any]? {
        guard
            let data = try? Data(contentsOf: url),
            let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return raw
    }

    private func parseUnix(_ raw: Any?) -> Date? {
        if let double = raw as? Double { return Date(timeIntervalSince1970: double) }
        if let int = raw as? Int { return Date(timeIntervalSince1970: Double(int)) }
        if let number = raw as? NSNumber { return Date(timeIntervalSince1970: number.doubleValue) }
        if let string = raw as? String, let double = Double(string) {
            return Date(timeIntervalSince1970: double)
        }
        return nil
    }

    private func stringify(_ value: Any) -> String {
        switch value {
        case let string as String:
            return string
        case let number as NSNumber:
            return number.stringValue
        case let bool as Bool:
            return bool ? "true" : "false"
        default:
            if JSONSerialization.isValidJSONObject(value),
               let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
               let string = String(data: data, encoding: .utf8) {
                return string
            }
            return String(describing: value)
        }
    }

    private func stringValue(_ raw: Any?) -> String? {
        if let string = raw as? String { return string }
        if let number = raw as? NSNumber { return number.stringValue }
        return nil
    }

    private func boolValue(_ raw: Any?) -> Bool {
        if let value = raw as? Bool { return value }
        if let value = raw as? NSNumber { return value.boolValue }
        if let string = raw as? String {
            let lowered = string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            return lowered == "true" || lowered == "1" || lowered == "yes"
        }
        return false
    }

    private func intValue(_ raw: Any?) -> Int? {
        if let value = raw as? Int { return value }
        if let value = raw as? NSNumber { return value.intValue }
        if let string = raw as? String, let value = Int(string) { return value }
        return nil
    }

    private func doubleValue(_ raw: Any?) -> Double {
        if let value = raw as? Double { return value }
        if let value = raw as? NSNumber { return value.doubleValue }
        if let string = raw as? String, let value = Double(string) { return value }
        return 0
    }

    private func nonEmptyString(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    private func nonEmptyURL(_ value: String) -> URL? {
        guard let text = nonEmptyString(value) else { return nil }
        return URL(fileURLWithPath: text)
    }
}
