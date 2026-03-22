/*
 * ContentView.swift — Root navigation view
 *
 * Shows model list if no model loaded, chat if model is ready.
 */

import SwiftUI

struct ContentView: View {
    @Environment(FlashMoEEngine.self) private var engine

    var body: some View {
        NavigationStack {
            switch engine.state {
            case .idle, .loading, .error:
                ModelListView()
            case .ready, .generating:
                ChatView()
            }
        }
    }
}
