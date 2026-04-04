import Charts
import FXAIGUICore
import SwiftUI

struct ResearchOSControlView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var diagnosticsCommand: String {
        guard let root = model.projectRoot else { return "" }
        return ResearchOSCommandFactory.environmentDiagnostics(
            projectRoot: root,
            profileName: model.researchSnapshot?.profileName ?? model.researchRecoveryDraft.profileName
        )
    }

    private var branchCommand: String {
        guard let root = model.projectRoot else { return "" }
        return ResearchOSCommandFactory.branchCommand(projectRoot: root, draft: model.researchBranchDraft)
    }

    private var auditCommand: String {
        guard let root = model.projectRoot else { return "" }
        return ResearchOSCommandFactory.auditSyncCommand(projectRoot: root, draft: model.researchAuditDraft)
    }

    private var vectorCommand: String {
        guard let root = model.projectRoot else { return "" }
        return ResearchOSCommandFactory.vectorCommand(projectRoot: root, draft: model.researchVectorDraft, reindexOnly: false)
    }

    private var recoveryCommand: String {
        guard let root = model.projectRoot else { return "" }
        return ResearchOSCommandFactory.recoveryCommand(projectRoot: root, draft: model.researchRecoveryDraft)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Research OS Control",
                    subtitle: "Operate Turso-backed FXAI state from one place: environment health, branches, PITR, audit-log sync, analog vectors, and artifact recovery."
                )

                if let snapshot = model.researchSnapshot {
                    environmentSummary(snapshot: snapshot)
                    branchAndAuditSection(snapshot: snapshot)
                    vectorSection(snapshot: snapshot)
                    recoverySection(snapshot: snapshot)
                    sourceOfTruthSection(snapshot: snapshot)
                } else {
                    EmptyStateView(
                        title: "No Research OS state found",
                        message: "Generate an Offline Lab operator dashboard first so the GUI can inspect Turso environment, branch state, audit logs, and analog vector surfaces.",
                        symbolName: "server.rack"
                    )
                }
            }
        }
    }

    private func environmentSummary(snapshot: ResearchOSControlSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            let environment = snapshot.environment

            LazyVGrid(
                columns: [
                    GridItem(.flexible(minimum: 220), spacing: 16),
                    GridItem(.flexible(minimum: 220), spacing: 16),
                    GridItem(.flexible(minimum: 220), spacing: 16),
                    GridItem(.flexible(minimum: 220), spacing: 16)
                ],
                spacing: 16
            ) {
                MetricCard(
                    title: "Backend",
                    value: environment?.backend ?? "Unavailable",
                    footnote: "Current Research OS storage backend detected by FXAI.",
                    symbolName: "externaldrive.connected.to.line.below.fill",
                    tint: FXAITheme.accent
                )
                MetricCard(
                    title: "Sync Mode",
                    value: environment?.syncMode ?? "Unknown",
                    footnote: "Shows whether Turso is local-only, embedded replica, or another configured mode.",
                    symbolName: "arrow.trianglehead.2.clockwise.rotate.90",
                    tint: FXAITheme.accentSoft
                )
                MetricCard(
                    title: "Branches",
                    value: "\(snapshot.activeBranchCount)",
                    footnote: "\(snapshot.branchCount) tracked branch records in the current dashboard.",
                    symbolName: "point.3.connected.trianglepath.dotted",
                    tint: FXAITheme.warning
                )
                MetricCard(
                    title: "Audit Events",
                    value: "\(snapshot.auditEventCount)",
                    footnote: "Recent Turso platform audit events stored in the operator dashboard.",
                    symbolName: "checkmark.shield.fill",
                    tint: FXAITheme.success
                )
            }

            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Text("Environment Health")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Spacer()
                        HStack(spacing: 10) {
                            StatusBadge(
                                title: "Encryption",
                                value: environment?.encryptionEnabled == true ? "On" : "Off",
                                tint: environment?.encryptionEnabled == true ? FXAITheme.success : FXAITheme.warning
                            )
                            StatusBadge(
                                title: "Platform API",
                                value: environment?.platformAPIEnabled == true ? "Ready" : "Off",
                                tint: environment?.platformAPIEnabled == true ? FXAITheme.success : FXAITheme.textMuted
                            )
                            StatusBadge(
                                title: "Sync",
                                value: environment?.syncEnabled == true ? "Enabled" : "Local",
                                tint: environment?.syncEnabled == true ? FXAITheme.success : FXAITheme.textMuted
                            )
                        }
                    }

                    LazyVGrid(
                        columns: [
                            GridItem(.flexible(minimum: 260), spacing: 12),
                            GridItem(.flexible(minimum: 260), spacing: 12)
                        ],
                        spacing: 12
                    ) {
                        keyValuePanel("Database", [
                            KeyValueRecord(key: "name", value: environment?.databaseName ?? ""),
                            KeyValueRecord(key: "organization", value: environment?.organizationSlug ?? ""),
                            KeyValueRecord(key: "group", value: environment?.groupName ?? ""),
                            KeyValueRecord(key: "location", value: environment?.locationName ?? "")
                        ])
                        keyValuePanel("Paths", [
                            KeyValueRecord(key: "database_path", value: environment?.databasePath?.path ?? ""),
                            KeyValueRecord(key: "cli_config_path", value: environment?.cliConfigPath?.path ?? ""),
                            KeyValueRecord(key: "sync_interval_seconds", value: environment?.syncIntervalSeconds.map(String.init) ?? ""),
                            KeyValueRecord(key: "auth_token_configured", value: environment?.authTokenConfigured == true ? "true" : "false")
                        ])
                    }

                    if let error = environment?.configError, !error.isEmpty {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(FXAITheme.warning.opacity(0.14))
                            .overlay(alignment: .leading) {
                                Text(error)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.warning)
                                    .padding(14)
                            }
                            .frame(maxWidth: .infinity, minHeight: 56, alignment: .leading)
                    }
                }
            }

            CommandPreviewCard(
                title: "Environment Diagnostics",
                summary: "Run the same validation and dashboard commands the operators use at the terminal, then inspect the updated Turso-backed operator state.",
                command: diagnosticsCommand,
                onCopy: { model.copyToPasteboard(diagnosticsCommand) },
                onTerminal: { model.handoffCommandToTerminal(diagnosticsCommand) }
            )
        }
    }

    private func branchAndAuditSection(snapshot: ResearchOSControlSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top, spacing: 16) {
                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Branch Workflows")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)

                        Picker("Action", selection: $model.researchBranchDraft.action) {
                            ForEach(ResearchOSBranchAction.allCases) { action in
                                Text(action.title).tag(action)
                            }
                        }
                        .pickerStyle(.segmented)

                        HStack(spacing: 12) {
                            labeledTextField("Profile", text: $model.researchBranchDraft.profileName)
                            labeledTextField("Source DB", text: $model.researchBranchDraft.sourceDatabase)
                        }

                        if model.researchBranchDraft.action == .create || model.researchBranchDraft.action == .pitrRestore || model.researchBranchDraft.action == .destroy {
                            HStack(spacing: 12) {
                                labeledTextField(
                                    model.researchBranchDraft.action == .destroy ? "Target DB" : "Target Branch",
                                    text: $model.researchBranchDraft.targetDatabase
                                )
                                if model.researchBranchDraft.action != .destroy {
                                    labeledTextField("Token Expiration", text: $model.researchBranchDraft.tokenExpiration)
                                }
                            }
                        }

                        if model.researchBranchDraft.action == .create || model.researchBranchDraft.action == .pitrRestore {
                            HStack(spacing: 12) {
                                labeledTextField("Timestamp", text: $model.researchBranchDraft.timestamp)
                                labeledTextField("Group", text: $model.researchBranchDraft.groupName)
                                labeledTextField("Location", text: $model.researchBranchDraft.locationName)
                            }

                            Toggle("Read-only branch token", isOn: $model.researchBranchDraft.readOnlyToken)
                        }

                        CommandPreviewCard(
                            title: "Generated Branch Command",
                            summary: "Create, inventory, restore, or destroy Turso branch surfaces using the exact Offline Lab CLI.",
                            command: branchCommand,
                            onCopy: { model.copyToPasteboard(branchCommand) },
                            onTerminal: { model.handoffCommandToTerminal(branchCommand) }
                        )
                    }
                }

                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Text("Tracked Branches")
                                .font(.headline)
                                .foregroundStyle(FXAITheme.textPrimary)
                            Spacer()
                            StatusBadge(
                                title: "Count",
                                value: "\(snapshot.branchCount)",
                                tint: FXAITheme.accentSoft
                            )
                        }

                        if snapshot.branches.isEmpty {
                            Text("No branch inventory is currently present in the latest operator dashboard.")
                                .foregroundStyle(FXAITheme.textSecondary)
                        } else {
                            VStack(spacing: 10) {
                                ForEach(snapshot.branches.prefix(8)) { branch in
                                    VStack(alignment: .leading, spacing: 8) {
                                        HStack {
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text(branch.name)
                                                    .font(.subheadline.weight(.semibold))
                                                    .foregroundStyle(FXAITheme.textPrimary)
                                                Text(branch.parentName.isEmpty ? branch.sourceDatabase : branch.parentName)
                                                    .font(.caption)
                                                    .foregroundStyle(FXAITheme.textMuted)
                                            }
                                            Spacer()
                                            StatusBadge(
                                                title: branch.branchKind,
                                                value: branch.status,
                                                tint: branch.status.lowercased() == "destroyed" ? FXAITheme.warning : FXAITheme.success
                                            )
                                        }

                                        HStack(spacing: 12) {
                                            compactMeta("Group", branch.groupName)
                                            compactMeta("Location", branch.locationName)
                                            compactMeta("Created", formatDate(branch.createdAt))
                                        }

                                        if let envArtifactPath = branch.envArtifactPath {
                                            Button("Reveal Env Artifact") {
                                                model.openInFinder(envArtifactPath)
                                            }
                                            .buttonStyle(.borderless)
                                            .foregroundStyle(FXAITheme.accent)
                                        }
                                    }
                                    .padding(14)
                                    .background(
                                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                                            .fill(FXAITheme.backgroundSecondary.opacity(0.62))
                                    )
                                }
                            }
                        }
                    }
                }
            }

            HStack(alignment: .top, spacing: 16) {
                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Audit Log Sync")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)

                        HStack(spacing: 12) {
                            labeledStepper("Limit", value: $model.researchAuditDraft.limit, range: 1...250, step: 5)
                            labeledStepper("Pages", value: $model.researchAuditDraft.pages, range: 1...10, step: 1)
                        }

                        CommandPreviewCard(
                            title: "Generated Audit Sync Command",
                            summary: "Refresh Turso platform audit events into the Research OS and immediately re-render the operator dashboard.",
                            command: auditCommand,
                            onCopy: { model.copyToPasteboard(auditCommand) },
                            onTerminal: { model.handoffCommandToTerminal(auditCommand) }
                        )
                    }
                }

                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Text("Recent Audit Events")
                                .font(.headline)
                                .foregroundStyle(FXAITheme.textPrimary)
                            Spacer()
                            StatusBadge(title: "Events", value: "\(snapshot.auditEventCount)", tint: FXAITheme.accentSoft)
                        }

                        if snapshot.auditEvents.isEmpty {
                            Text("No recent Turso audit events are currently recorded in the operator dashboard.")
                                .foregroundStyle(FXAITheme.textSecondary)
                        } else {
                            VStack(spacing: 10) {
                                ForEach(snapshot.auditEvents.prefix(8)) { event in
                                    HStack(alignment: .top, spacing: 12) {
                                        Circle()
                                            .fill(FXAITheme.accent.opacity(0.9))
                                            .frame(width: 8, height: 8)
                                            .padding(.top, 6)

                                        VStack(alignment: .leading, spacing: 4) {
                                            Text(event.eventType)
                                                .font(.subheadline.weight(.semibold))
                                                .foregroundStyle(FXAITheme.textPrimary)
                                            Text(event.targetName.isEmpty ? event.eventID : event.targetName)
                                                .font(.caption)
                                                .foregroundStyle(FXAITheme.textSecondary)
                                            Text([event.organizationSlug, formatDate(event.occurredAt), formatDate(event.observedAt)]
                                                .filter { !$0.isEmpty }
                                                .joined(separator: "  ·  "))
                                                .font(.caption2)
                                                .foregroundStyle(FXAITheme.textMuted)
                                        }

                                        Spacer()
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func vectorSection(snapshot: ResearchOSControlSnapshot) -> some View {
        HStack(alignment: .top, spacing: 16) {
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Text("Analog Vector Browser")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Spacer()
                        if !snapshot.symbols.isEmpty {
                            Picker("Symbol", selection: $model.selectedResearchSymbol) {
                                ForEach(snapshot.symbols.map(\.symbol), id: \.self) { symbol in
                                    Text(symbol).tag(symbol)
                                }
                            }
                            .pickerStyle(.segmented)
                            .frame(maxWidth: 320)
                            .onChange(of: model.selectedResearchSymbol) { _, newValue in
                                model.researchVectorDraft.symbol = newValue
                            }
                        }
                    }

                    HStack(spacing: 12) {
                        labeledTextField("Profile", text: $model.researchVectorDraft.profileName)
                        labeledTextField("Symbol", text: $model.researchVectorDraft.symbol)
                        labeledStepper("Neighbors", value: $model.researchVectorDraft.limit, range: 1...20, step: 1)
                    }

                    CommandPreviewCard(
                        title: "Generated Vector Command",
                        summary: "Refresh native Turso analog vectors and inspect the nearest shadow-state neighbors for the selected symbol.",
                        command: vectorCommand,
                        onCopy: { model.copyToPasteboard(vectorCommand) },
                        onTerminal: { model.handoffCommandToTerminal(vectorCommand) }
                    )
                }
            }

            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Nearest Analog States")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    if let detail = model.selectedResearchSymbolDetail, !detail.analogNeighbors.isEmpty {
                        Chart(detail.analogNeighbors.prefix(8)) { neighbor in
                            BarMark(
                                x: .value("Neighbor", neighbor.pluginName),
                                y: .value("Similarity", neighbor.similarity)
                            )
                            .foregroundStyle(FXAITheme.accent.gradient)
                            .cornerRadius(5)
                        }
                        .frame(height: 180)

                        VStack(spacing: 10) {
                            ForEach(detail.analogNeighbors.prefix(6)) { neighbor in
                                VStack(alignment: .leading, spacing: 8) {
                                    HStack {
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(neighbor.pluginName)
                                                .font(.subheadline.weight(.semibold))
                                                .foregroundStyle(FXAITheme.textPrimary)
                                            Text(neighbor.sourceKey)
                                                .font(.caption)
                                                .foregroundStyle(FXAITheme.textMuted)
                                        }
                                        Spacer()
                                        StatusBadge(
                                            title: neighbor.scope,
                                            value: String(format: "%.3f", neighbor.similarity),
                                            tint: FXAITheme.success
                                        )
                                    }

                                    HStack(spacing: 12) {
                                        compactMeta("Score", String(format: "%.3f", neighbor.score))
                                        compactMeta("Distance", String(format: "%.3f", neighbor.distance))
                                        compactMeta("Type", neighbor.sourceType)
                                    }
                                }
                                .padding(14)
                                .background(
                                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                                        .fill(FXAITheme.backgroundSecondary.opacity(0.62))
                                )
                            }
                        }
                    } else {
                        Text("No analog neighbors are currently attached to the selected symbol in the latest dashboard.")
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                }
            }
        }
    }

    private func recoverySection(snapshot: ResearchOSControlSnapshot) -> some View {
        HStack(alignment: .top, spacing: 16) {
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Recovery Tools")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    HStack(spacing: 12) {
                        labeledTextField("Profile", text: $model.researchRecoveryDraft.profileName)
                        Picker("Runtime Mode", selection: $model.researchRecoveryDraft.runtimeMode) {
                            Text("Research").tag("research")
                            Text("Production").tag("production")
                        }
                        .pickerStyle(.menu)
                    }

                    CommandPreviewCard(
                        title: "Generated Recovery Command",
                        summary: "Rebuild promoted artifacts, regenerate lineage, and emit a minimal bundle after Turso state changes or operator recovery work.",
                        command: recoveryCommand,
                        onCopy: { model.copyToPasteboard(recoveryCommand) },
                        onTerminal: { model.handoffCommandToTerminal(recoveryCommand) }
                    )
                }
            }

            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Recovery Context")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    let symbolRows = snapshot.symbols.map {
                        KeyValueRecord(
                            key: $0.symbol,
                            value: "\($0.analogNeighbors.count) analog neighbors"
                        )
                    }

                    ArtifactSectionCard(
                        section: RuntimeArtifactSection(
                            title: "Dashboard Source Of Truth",
                            sourcePath: nil,
                            values: snapshot.sourceOfTruth + symbolRows
                        ),
                        onReveal: model.openInFinder
                    )
                }
            }
        }
    }

    private func sourceOfTruthSection(snapshot: ResearchOSControlSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Source of Truth")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if snapshot.sourceOfTruth.isEmpty {
                    Text("The latest operator dashboard did not include a source-of-truth section.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    LazyVGrid(
                        columns: [
                            GridItem(.flexible(minimum: 260), spacing: 12),
                            GridItem(.flexible(minimum: 260), spacing: 12)
                        ],
                        spacing: 12
                    ) {
                        ForEach(snapshot.sourceOfTruth) { row in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(row.key.replacingOccurrences(of: "_", with: " ").capitalized)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textSecondary)
                                Text(row.value)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textPrimary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(14)
                            .background(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .fill(FXAITheme.backgroundSecondary.opacity(0.62))
                            )
                        }
                    }
                }
            }
        }
    }

    private func keyValuePanel(_ title: String, _ records: [KeyValueRecord]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
            ForEach(records.filter { !$0.value.isEmpty }) { record in
                HStack(alignment: .top) {
                    Text(record.key.replacingOccurrences(of: "_", with: " "))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(FXAITheme.textSecondary)
                    Spacer()
                    Text(record.value)
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textMuted)
                        .multilineTextAlignment(.trailing)
                        .textSelection(.enabled)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(FXAITheme.backgroundSecondary.opacity(0.62))
        )
    }

    private func labeledTextField(_ title: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textSecondary)
            TextField(title, text: text)
                .textFieldStyle(.roundedBorder)
        }
    }

    private func labeledStepper(_ title: String, value: Binding<Int>, range: ClosedRange<Int>, step: Int) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textSecondary)
            Stepper("\(value.wrappedValue)", value: value, in: range, step: step)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func compactMeta(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(FXAITheme.textMuted)
            Text(value.isEmpty ? "n/a" : value)
                .font(.caption)
                .foregroundStyle(FXAITheme.textSecondary)
        }
    }

    private func formatDate(_ date: Date?) -> String {
        guard let date else { return "" }
        return date.formatted(date: .abbreviated, time: .shortened)
    }
}
