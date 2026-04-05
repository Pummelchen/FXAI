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
            .sink { [weak self] _ in
                self?.isApplicationActive = true
                self?.recomputeProfile()
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)
            .sink { [weak self] _ in
                self?.isApplicationActive = false
                self?.recomputeProfile()
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: NSApplication.didHideNotification)
            .sink { [weak self] _ in
                self?.isApplicationActive = false
                self?.recomputeProfile()
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: NSApplication.didUnhideNotification)
            .sink { [weak self] _ in
                self?.isApplicationActive = NSApp?.isActive ?? true
                self?.recomputeProfile()
            }
            .store(in: &cancellables)
    }

    private func bindProcessNotifications() {
        NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)
            .sink { [weak self] _ in
                self?.recomputeProfile()
            }
            .store(in: &cancellables)
    }

    private func bindMemoryPressureNotifications() {
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.normal, .warning, .critical],
            queue: DispatchQueue.global(qos: .utility)
        )
        source.setEventHandler { [weak self, weak source] in
            guard let source else { return }
            let level = Self.mapMemoryPressure(source.data)
            Task { @MainActor [weak self] in
                self?.memoryPressureLevel = level
                self?.recomputeProfile()
            }
        }
        source.resume()
        memoryPressureSource = source
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
