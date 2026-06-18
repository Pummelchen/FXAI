import Charts
import FXGUICore
import SwiftUI

struct PromotionCenterView: View {
    @EnvironmentObject private var model: FXGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Promotion Governance Center",
                    subtitle: "Manage strategy candidates through the staged pipeline: backtest → demo → shadow → live. Approve, reject, annotate, and roll back with full audit trail."
                )

                pipelineOverview

                if !model.promotionCandidates.isEmpty {
                    pipelineStageBoard

                    candidateDetailPanel
                } else {
                    EmptyStateView(
                        title: "No promotion candidates",
                        message: "Run backtests and offline lab promotion flows to generate candidates for governance review.",
                        symbolName: "rosette"
                    )
                }

                if !model.rollbackHistory.isEmpty {
                    rollbackHistorySection
                }
            }
        }
    }

    // MARK: - Pipeline Overview Summary

    private var pipelineOverview: some View {
        LazyVGrid(
            columns: Array(repeating: GridItem(.flexible(minimum: 140), spacing: 12), count: PromotionStage.allCases.count),
            spacing: 12
        ) {
            ForEach(PromotionStage.allCases) { stage in
                let count = model.candidatesByStage[stage]?.count ?? 0
                VStack(spacing: 6) {
                    Image(systemName: stage.symbolName)
                        .font(.title2)
                        .foregroundStyle(FXAITheme.accent)
                    Text(stage.title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(FXAITheme.textPrimary)
                    Text("\(count)")
                        .font(.title3.weight(.bold))
                        .foregroundStyle(FXAITheme.textPrimary)
                }
                .padding(12)
                .frame(maxWidth: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(FXAITheme.panel.opacity(0.32))
                )
            }
        }
    }

    // MARK: - Stage Board (Kanban-style)

    private var pipelineStageBoard: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Pipeline Board")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ScrollView(.horizontal, showsIndicators: true) {
                    HStack(alignment: .top, spacing: 14) {
                        ForEach(PromotionStage.allCases) { stage in
                            stageColumn(stage: stage)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }

    private func stageColumn(stage: PromotionStage) -> some View {
        let candidates = model.candidatesByStage[stage] ?? []
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: stage.symbolName)
                    .foregroundStyle(FXAITheme.accentSoft)
                Text(stage.title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(FXAITheme.textPrimary)
                Spacer()
                Text("\(candidates.count)")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(FXAITheme.textMuted)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(FXAITheme.panel.opacity(0.4)))
            }

            ForEach(candidates) { candidate in
                candidateCard(candidate)
            }

            if candidates.isEmpty {
                Text("No candidates")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
                    .padding(.vertical, 8)
            }
        }
        .frame(width: 220, alignment: .topLeading)
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(FXAITheme.backgroundSecondary.opacity(0.4))
        )
    }

    private func candidateCard(_ candidate: PromotionCandidate) -> some View {
        Button {
            model.selectedCandidateID = candidate.id
        } label: {
            VStack(alignment: .leading, spacing: 4) {
                Text("\(candidate.symbol)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(FXAITheme.textPrimary)
                Text(candidate.pluginName)
                    .font(.caption2)
                    .foregroundStyle(FXAITheme.textSecondary)
                HStack(spacing: 6) {
                    Label("\(candidate.approvalCount)", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(FXAITheme.success)
                    if candidate.rejectionCount > 0 {
                        Label("\(candidate.rejectionCount)", systemImage: "xmark.circle.fill")
                            .foregroundStyle(.red)
                    }
                }
                .font(.caption2)
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(model.selectedCandidateID == candidate.id ? FXAITheme.accent.opacity(0.18) : FXAITheme.panel.opacity(0.5))
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Candidate Detail Panel

    @ViewBuilder
    private var candidateDetailPanel: some View {
        if let candidate = model.selectedCandidate {
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 16) {
                    candidateHeader(candidate)
                    candidateScores(candidate)
                    approvalWorkflow(candidate)
                    auditNotesSection(candidate)
                    rollbackSection(candidate)
                    evidencePackButton(candidate)
                }
            }
        } else {
            EmptyStateView(
                title: "No candidate selected",
                message: "Select a candidate from the pipeline board to inspect, approve, annotate, or roll back.",
                symbolName: "person.crop.rectangle"
            )
        }
    }

    private func candidateHeader(_ candidate: PromotionCandidate) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("\(candidate.symbol) — \(candidate.pluginName)")
                    .font(.system(size: 22, weight: .semibold, design: .rounded))
                    .foregroundStyle(FXAITheme.textPrimary)
                HStack(spacing: 8) {
                    Label(candidate.currentStage.title, systemImage: candidate.currentStage.symbolName)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(FXAITheme.accent)
                    if let profileName = candidate.profileName {
                        Text(profileName)
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textMuted)
                    }
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text(candidate.canAdvance ? "Ready to Advance" : "Awaiting Approvals")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(candidate.canAdvance ? FXAITheme.success : FXAITheme.warning)
                if let nextStage = candidate.currentStage.nextStage {
                    Text("→ \(nextStage.title)")
                        .font(.caption2)
                        .foregroundStyle(FXAITheme.textMuted)
                }
            }
        }
    }

    private func candidateScores(_ candidate: PromotionCandidate) -> some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 160), spacing: 12),
                GridItem(.flexible(minimum: 160), spacing: 12),
                GridItem(.flexible(minimum: 160), spacing: 12),
                GridItem(.flexible(minimum: 160), spacing: 12)
            ],
            spacing: 12
        ) {
            MetricCard(
                title: "Champion",
                value: String(format: "%.4f", candidate.championScore),
                footnote: "Score of the current champion.",
                symbolName: "crown.fill",
                tint: FXAITheme.success
            )
            MetricCard(
                title: "Challenger",
                value: String(format: "%.4f", candidate.challengerScore),
                footnote: "Score of the challenger model.",
                symbolName: "bolt.fill",
                tint: FXAITheme.warning
            )
            MetricCard(
                title: "Portfolio",
                value: String(format: "%.4f", candidate.portfolioScore),
                footnote: "Combined portfolio metric.",
                symbolName: "chart.pie.fill",
                tint: FXAITheme.accentSoft
            )
            MetricCard(
                title: "Delta",
                value: String(format: "%+.4f", candidate.scoreDelta),
                footnote: candidate.scoreDelta > 0 ? "Champion leads." : "Challenger leads.",
                symbolName: candidate.scoreDelta > 0 ? "arrow.up.right" : "arrow.down.right",
                tint: candidate.scoreDelta > 0 ? FXAITheme.success : FXAITheme.warning
            )
        }
    }

    // MARK: - Approval Workflow

    private func approvalWorkflow(_ candidate: PromotionCandidate) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Approval Gate")
                .font(.headline)
                .foregroundStyle(FXAITheme.textPrimary)

            if let nextStage = candidate.currentStage.nextStage {
                Text("Requires \(nextStage.requiredApprovalCount) approval(s) to advance to \(nextStage.title). Currently \(candidate.approvalCount) approved, \(candidate.rejectionCount) rejected.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textSecondary)
            }

            if !candidate.approvals.isEmpty {
                ForEach(candidate.approvals) { approval in
                    HStack(spacing: 8) {
                        Image(systemName: approval.decision.symbolName)
                            .foregroundStyle(approval.decision == .approved ? FXAITheme.success : .red)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(approval.reviewerRole.title) — \(approval.decision.title)")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            if !approval.note.isEmpty {
                                Text(approval.note)
                                    .font(.caption2)
                                    .foregroundStyle(FXAITheme.textSecondary)
                            }
                        }
                        Spacer()
                    }
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 10).fill(FXAITheme.backgroundSecondary.opacity(0.5)))
                }
            }

            if candidate.currentStage.nextStage != nil {
                Divider()

                Text("Submit Decision")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(FXAITheme.textPrimary)

                TextEditor(text: $model.approvalNoteDraft)
                    .font(.caption)
                    .frame(minHeight: 48, maxHeight: 72)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 10).fill(FXAITheme.backgroundSecondary.opacity(0.6)))

                HStack(spacing: 10) {
                    Button {
                        model.approveCandidate(candidate, reviewerRole: model.selectedRole, note: model.approvalNoteDraft)
                    } label: {
                        Label("Approve", systemImage: "checkmark.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(FXAITheme.success)
                    .disabled(candidate.rejectionCount > 0)

                    Button(role: .destructive) {
                        model.rejectCandidate(candidate, reviewerRole: model.selectedRole, note: model.approvalNoteDraft)
                    } label: {
                        Label("Reject", systemImage: "xmark.circle.fill")
                    }
                    .buttonStyle(.bordered)

                    if candidate.canAdvance {
                        Button {
                            model.advanceCandidate(candidate, reviewerRole: model.selectedRole, note: model.approvalNoteDraft)
                        } label: {
                            Label("Advance to \(candidate.currentStage.nextStage?.title ?? "")", systemImage: "arrow.right.circle.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(FXAITheme.accent)
                    }
                }
            }
        }
    }

    // MARK: - Audit Notes

    private func auditNotesSection(_ candidate: PromotionCandidate) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Audit Trail")
                .font(.headline)
                .foregroundStyle(FXAITheme.textPrimary)

            if candidate.auditNotes.isEmpty {
                Text("No audit notes recorded for this candidate.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            } else {
                ForEach(candidate.auditNotes) { note in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "text.quote")
                            .foregroundStyle(FXAITheme.accentSoft)
                            .padding(.top, 2)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(note.authorRole.title) at \(note.stageAtCreation.title)")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text(note.text)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                    }
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 10).fill(FXAITheme.backgroundSecondary.opacity(0.4)))
                }
            }

            Divider()

            HStack {
                TextEditor(text: $model.auditNoteDraft)
                    .font(.caption)
                    .frame(minHeight: 36, maxHeight: 56)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 10).fill(FXAITheme.backgroundSecondary.opacity(0.6)))

                Button {
                    model.addAuditNote(to: candidate, role: model.selectedRole, text: model.auditNoteDraft)
                } label: {
                    Label("Add Note", systemImage: "plus.circle.fill")
                }
                .buttonStyle(.bordered)
                .disabled(model.auditNoteDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }

    // MARK: - Rollback

    private func rollbackSection(_ candidate: PromotionCandidate) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Rollback Controls")
                .font(.headline)
                .foregroundStyle(FXAITheme.textPrimary)

            if candidate.currentStage == .backtest {
                Text("This candidate is at the initial stage and cannot be rolled back further.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            } else {
                let previousStages = PromotionStage.allCases.filter { $0.rawValue < candidate.currentStage.rawValue }
                HStack(spacing: 8) {
                    ForEach(previousStages) { targetStage in
                        Button {
                            model.rollbackCandidate(candidate, toStage: targetStage, reason: model.rollbackReasonDraft, initiatedBy: model.selectedRole)
                        } label: {
                            Label("Rollback to \(targetStage.title)", systemImage: "arrow.uturn.backward.circle")
                                .font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .disabled(model.rollbackReasonDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }

                TextField("Rollback reason (required)", text: $model.rollbackReasonDraft)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
            }
        }
    }

    // MARK: - Evidence Pack Button

    private func evidencePackButton(_ candidate: PromotionCandidate) -> some View {
        HStack {
            Button {
                _ = model.generateEvidencePack(for: candidate)
                model.navigate(to: .evidencePacks)
            } label: {
                Label("Generate Evidence Pack", systemImage: "doc.badge.arrow.up.fill")
            }
            .buttonStyle(.borderedProminent)
            .tint(FXAITheme.accent)

            if let setPath = candidate.setPath {
                Button {
                    model.openInFinder(setPath)
                } label: {
                    Label("Reveal Set", systemImage: "folder")
                }
                .buttonStyle(.bordered)
            }

            Spacer()
        }
    }

    // MARK: - Rollback History

    private var rollbackHistorySection: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 10) {
                Text("Rollback History")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ForEach(model.rollbackHistory.prefix(10)) { record in
                    HStack(spacing: 8) {
                        Image(systemName: "arrow.uturn.backward.circle")
                            .foregroundStyle(FXAITheme.warning)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(record.fromStage.title) → \(record.toStage.title)")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text("By \(record.initiatedBy.title): \(record.reason)")
                                .font(.caption2)
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                        Spacer()
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }
}
