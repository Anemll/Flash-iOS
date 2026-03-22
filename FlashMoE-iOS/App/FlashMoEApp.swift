/*
 * FlashMoEApp.swift — Main app entry point
 *
 * Flash-MoE iOS: Run massive MoE models on iPhone.
 * Targets iOS 18+ (iPhone 15 Pro and later with 8GB+ RAM).
 */

import SwiftUI

@main
struct FlashMoEApp: App {
#if os(iOS)
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
#endif
    @State private var engine: FlashMoEEngine = .init()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(engine)
        }
    }
}

