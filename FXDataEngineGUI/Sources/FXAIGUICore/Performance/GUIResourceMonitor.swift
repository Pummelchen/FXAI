import AppKit
import Combine
import Dispatch
import Foundation

@MainActor
public final class GUIResourceMonitor: ObservableObject {
    public static let shared = GUIResourceMonitor()

    @Published public private(set) var profile: GUIRenderingProfile

    private var memoryPressureLevel: GUIMemoryPressureLevel
    private var isApplicationActive: Bool
    private var cancellables: Set<AnyCancellable> = []
    private var memoryPressureSource: DispatchSourceMemoryPressure?

    public init(
        initialProfile: GUIRenderingProfile = .default
    ) {
        self.profile = initialProfile
        self.memoryPressureLevel = initialProfile.memoryPressure
        self.isApplicationActive = NSApp?.isActive ?? true

        bindLifecycleNotifications()
        bindProcessNotifications()
        bindMemoryPressureNotifications()
        recomputeProfile()
    }

    deinit {
        memoryPressureSource?.cancel()
    }

    private func bindLifecycleNotifications() {
        NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handleApplicationActivityChange(isActive: true)
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handleApplicationActivityChange(isActive: false)
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: NSApplication.didHideNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handleApplicationActivityChange(isActive: false)
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: NSApplication.didUnhideNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handleApplicationActivityChange(isActive: NSApp?.isActive ?? true)
            }
            .store(in: &cancellables)
    }

    private func bindProcessNotifications() {
        NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handleThermalStateDidChange()
            }
            .store(in: &cancellables)
    }

    private func bindMemoryPressureNotifications() {
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.normal, .warning, .critical],
            queue: DispatchQueue.main
        )
        source.setEventHandler { [weak self, weak source] in
            guard let source else { return }
            let level = Self.mapMemoryPressure(source.data)
            self?.handleMemoryPressure(level)
        }
        source.resume()
        memoryPressureSource = source
    }

    func handleApplicationActivityChange(isActive: Bool) {
        isApplicationActive = isActive
        recomputeProfile()
    }

    func handleThermalStateDidChange() {
        recomputeProfile()
    }

    func handleMemoryPressure(_ level: GUIMemoryPressureLevel) {
        memoryPressureLevel = level
        recomputeProfile()
    }

    private func recomputeProfile() {
        profile = GUIRenderingProfile.resolve(
            appIsActive: isApplicationActive,
            thermalState: ProcessInfo.processInfo.thermalState,
            memoryPressure: memoryPressureLevel,
            isLowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
    }

    private static func mapMemoryPressure(_ event: DispatchSource.MemoryPressureEvent) -> GUIMemoryPressureLevel {
        if event.contains(.critical) {
            return .critical
        }
        if event.contains(.warning) {
            return .warning
        }
        return .normal
    }
}
