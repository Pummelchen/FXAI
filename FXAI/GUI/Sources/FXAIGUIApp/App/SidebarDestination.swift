import Foundation

enum SidebarDestination: String, CaseIterable, Identifiable {
    case overview
    case liveOverview
    case demoOverview
    case onboarding
    case roles
    case incidents
    case auditLab
    case backtestBuilder
    case offlineLab
    case researchWorkspace
    case labelEngine
    case newsPulse
    case ratesEngine
    case crossAsset
    case pairNetwork
    case microstructure
    case adaptiveRouter
    case driftGovernance
    case dynamicEnsemble
    case probCalibration
    case executionQuality
    case runtimeMonitor
    case promotionCenter
    case platformControl
    case researchControl
    case advancedVisuals
    case plugins
    case reports
    case commands
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .overview: "Overview"
        case .liveOverview: "Live Overview"
        case .demoOverview: "Demo Overview"
        case .onboarding: "Onboarding"
        case .roles: "Role Workspaces"
        case .incidents: "Incident Center"
        case .auditLab: "Audit Lab"
        case .backtestBuilder: "Backtests"
        case .offlineLab: "Offline Lab"
        case .researchWorkspace: "Research Workspace"
        case .labelEngine: "Label Engine"
        case .newsPulse: "NewsPulse"
        case .ratesEngine: "Rates Engine"
        case .crossAsset: "Cross Asset"
        case .pairNetwork: "Pair Network"
        case .microstructure: "Microstructure"
        case .adaptiveRouter: "Adaptive Router"
        case .driftGovernance: "Drift Governance"
        case .dynamicEnsemble: "Dynamic Ensemble"
        case .probCalibration: "Prob Calibration"
        case .executionQuality: "Execution Quality"
        case .runtimeMonitor: "Runtime Monitor"
        case .promotionCenter: "Promotion Center"
        case .platformControl: "Platform Control"
        case .researchControl: "Research OS"
        case .advancedVisuals: "Visual Analysis"
        case .plugins: "Plugin Zoo"
        case .reports: "Reports"
        case .commands: "Command Center"
        case .settings: "Settings"
        }
    }

    var symbolName: String {
        switch self {
        case .overview: "square.grid.2x2.fill"
        case .liveOverview: "bolt.horizontal.circle.fill"
        case .demoOverview: "waveform.path.ecg.rectangle"
        case .onboarding: "sparkles.rectangle.stack.fill"
        case .roles: "person.3.fill"
        case .incidents: "exclamationmark.triangle.fill"
        case .auditLab: "checklist.checked"
        case .backtestBuilder: "gauge.with.needle.fill"
        case .offlineLab: "gearshape.2.fill"
        case .researchWorkspace: "brain.head.profile"
        case .labelEngine: "target"
        case .newsPulse: "dot.radiowaves.left.and.right"
        case .ratesEngine: "chart.line.text.clipboard.fill"
        case .crossAsset: "globe.americas.fill"
        case .pairNetwork: "point.3.connected.trianglepath.dotted"
        case .microstructure: "waveform.path.ecg.rectangle.fill"
        case .adaptiveRouter: "point.3.filled.connected.trianglepath.dotted"
        case .driftGovernance: "waveform.and.magnifyingglass"
        case .dynamicEnsemble: "dial.high.fill"
        case .probCalibration: "checkmark.shield.fill"
        case .executionQuality: "speedometer"
        case .runtimeMonitor: "waveform.path.ecg.rectangle.fill"
        case .promotionCenter: "rosette"
        case .platformControl: "server.rack"
        case .researchControl: "server.rack"
        case .advancedVisuals: "sparkles.tv"
        case .plugins: "shippingbox.fill"
        case .reports: "doc.text.image.fill"
        case .commands: "terminal.fill"
        case .settings: "slider.horizontal.3"
        }
    }
}
