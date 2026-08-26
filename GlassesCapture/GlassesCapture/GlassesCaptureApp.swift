import MWDATCore
import SwiftUI
import UIKit

final class AppDelegate: NSObject, UIApplicationDelegate {
  func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
  ) -> Bool {
    true
  }

  func application(
    _ app: UIApplication,
    open url: URL,
    options: [UIApplication.OpenURLOptionsKey: Any] = [:]
  ) -> Bool {
    Task {
      _ = try? await Wearables.shared.handleUrl(url)
    }
    return true
  }
}

@main
struct GlassesCaptureApp: App {
  @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
  @State private var controller: StreamController?
  @State private var configureError: String?

  var body: some Scene {
    WindowGroup {
      Group {
        if let controller {
          ContentView(controller: controller)
            .onOpenURL { url in
              Task { await controller.handleOpenURL(url) }
            }
        } else if let configureError {
          ContentUnavailableView(
            "DAT failed to start",
            systemImage: "eyeglasses",
            description: Text(configureError)
          )
          .padding()
        } else {
          ProgressView("Starting DAT…")
            .task { await startDAT() }
        }
      }
    }
  }

  @MainActor
  private func startDAT() async {
    do {
      try Wearables.configure()
      controller = StreamController()
    } catch {
      configureError = String(describing: error)
    }
  }
}
