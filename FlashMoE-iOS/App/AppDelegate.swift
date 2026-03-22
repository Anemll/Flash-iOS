#if canImport(UIKit) && !isMacro
/*
 * AppDelegate.swift — Background URL session event handling
 *
 * Required for URLSession background downloads to reconnect
 * when iOS relaunches the app to deliver completed download events.
 */

import UIKit

class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        if identifier == "com.flashmoe.model-download" {
            DownloadManager.shared.backgroundCompletionHandler = completionHandler
        }
    }
}
#endif
