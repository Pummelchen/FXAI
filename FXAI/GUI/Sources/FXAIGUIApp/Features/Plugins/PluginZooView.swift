import FXAIGUICore
import SwiftUI

struct PluginZooView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var filteredPlugins: [PluginDescriptor] {
        guard let snapshot = model.snapshot else { return [] }

        return snapshot.plugins.filter { plugin in
            let matchesFamily = model.selectedPluginFamily == "All" || plugin.family == model.selectedPluginFamily
            let matchesQuery = model.pluginSearchText.isEmpty
                || plugin.name.localizedCaseInsensitiveContains(model.pluginSearchText)
                || plugin.family.localizedCaseInsensitiveContains(model.pluginSearchText)
            return matchesFamily && matchesQuery
        }
    }

    private var families: [String] {
        guard let snapshot = model.snapshot else { return ["All"] }
        return ["All"] + snapshot.pluginFamilies.map(\.family)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader(
                title: "Plugin Zoo",
                subtitle: "Search the model inventory by family and quickly see what FXAI can run today."
            )

            controls

            if filteredPlugins.isEmpty {
                EmptyStateView(
                    title: "No plugins match the current filter",
                    message: "Adjust the family filter or search query to inspect the plugin zoo.",
                    symbolName: "shippingbox"
                )
            } else {
                List(filteredPlugins) { plugin in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(plugin.name)
                                .font(.headline)
                                .foregroundStyle(FXAITheme.textPrimary)
                            Spacer()
                            Text(plugin.family)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.accent)
                        }

                        HStack {
                            Label(plugin.sourceKind.rawValue.capitalized, systemImage: plugin.sourceKind == .folder ? "folder.fill" : "doc.text.fill")
                            Text(plugin.sourcePath.path)
                                .lineLimit(1)
                        }
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textSecondary)
                    }
                    .padding(.vertical, 6)
                }
                .scrollContentBackground(.hidden)
                .background(FXAITheme.panel.opacity(0.22))
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
        }
    }

    private var controls: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) {
                TextField("Search plugins or families", text: $model.pluginSearchText)
                    .textFieldStyle(.roundedBorder)

                Picker("Family", selection: $model.selectedPluginFamily) {
                    ForEach(families, id: \.self) { family in
                        Text(family).tag(family)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 220)
            }

            VStack(alignment: .leading, spacing: 12) {
                TextField("Search plugins or families", text: $model.pluginSearchText)
                    .textFieldStyle(.roundedBorder)

                Picker("Family", selection: $model.selectedPluginFamily) {
                    ForEach(families, id: \.self) { family in
                        Text(family).tag(family)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 280, alignment: .leading)
            }
        }
    }
}
