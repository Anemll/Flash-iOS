/*
 * ModelListView.swift — Model discovery and loading
 *
 * Lists locally available models and allows downloading from HuggingFace.
 * For v1, supports loading models already present on device.
 */

import SwiftUI
import UniformTypeIdentifiers

// MARK: - Local Model Entry

struct LocalModel: Identifiable {
    let id = UUID()
    let name: String
    let path: String
    let sizeBytes: UInt64
    let hasTiered: Bool
    let has4bit: Bool
    let has2bit: Bool

    var sizeMB: Double { Double(sizeBytes) / 1_048_576 }
    var sizeGB: Double { sizeMB / 1024 }
}

// MARK: - Model List View

struct ModelListView: View {
    @Environment(FlashMoEEngine.self) private var engine
    @State private var localModels: [LocalModel] = []
    @State private var isScanning = true
    @State private var loadError: String?
    @State private var selectedModel: LocalModel?
    @AppStorage("cacheIOSplit") private var cacheIOSplit: Int = 1
    @AppStorage("activeExpertsK") private var activeExpertsK: Int = 0
    @State private var showFilePicker = false
    @State private var modelToExport: LocalModel? = nil
    @State private var importedBookmark: Data? = nil
    @State private var showImportActionAlert = false
    @State private var pendingImportURL: URL? = nil
    @State private var importProgress: String? = nil
    @State private var modelToDelete: LocalModel? = nil
    private let downloadManager = DownloadManager.shared

    var body: some View {
        List {
            Section {
                headerView
            }
            .listRowBackground(Color.clear)

            if isScanning {
                Section {
                    HStack {
                        ProgressView()
                        Text("Scanning for models...")
                            .foregroundStyle(.secondary)
                    }
                }
            } else if localModels.isEmpty {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("No models found")
                            .font(.headline)
                        Text("Download a model below, or transfer one via Files.app.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }
            } else {
                Section("On Device") {
                    ForEach(localModels) { model in
                        ModelRow(model: model, isLoading: engine.state == .loading && selectedModel?.id == model.id)
                            .onTapGesture { loadModel(model) }
                            .swipeActions(edge: .leading) {
                                Button {
                                    modelToExport = model
                                } label: {
                                    Label("Move to Files", systemImage: "square.and.arrow.up")
                                }
                                .tint(.blue)
                            }
                            .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                Button(role: .destructive) {
                                    modelToDelete = model
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                    }
                }
            }

            // Download section
            Section("Download from HuggingFace") {
                ForEach(ModelCatalog.models) { entry in
                    let hasActiveDownload = downloadManager.activeDownload?.catalogId == entry.id
                        && downloadManager.activeDownload?.status != .complete
                    ModelDownloadRow(
                        entry: entry,
                        downloadManager: downloadManager,
                        isDownloaded: !hasActiveDownload && downloadManager.isModelDownloaded(entry.id)
                    )
                }
            }

            Section("Expert Settings") {
                Picker("Active Experts (K)", selection: $activeExpertsK) {
                    Text("Model default").tag(0)
                    Text("K=2 (fastest, lowest quality)").tag(2)
                    Text("K=4").tag(4)
                    Text("K=6").tag(6)
                    Text("K=8").tag(8)
                    Text("K=10 (full quality)").tag(10)
                }
                .pickerStyle(.menu)
                if activeExpertsK > 0 {
                    Text("Uses \(activeExpertsK) experts per token instead of the model default. Lower = faster but less accurate. Reload model to apply.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Picker("Expert I/O Fanout", selection: $cacheIOSplit) {
                    Text("Off (single pread)").tag(1)
                    Text("2 chunks").tag(2)
                    Text("4 chunks").tag(4)
                    Text("8 chunks").tag(8)
                }
                .pickerStyle(.menu)
                if cacheIOSplit > 1 {
                    Text("Splits each expert read into \(cacheIOSplit) page-aligned chunks for parallel SSD reads. Reload model to apply.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if let error = downloadManager.error,
               downloadManager.activeDownload == nil {
                Section {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.caption)
                }
            }

            if case .error(let msg) = engine.state {
                Section {
                    Label(msg, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.caption)
                }
            }
        }
        .navigationTitle("Flash-MoE")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    showFilePicker = true
                } label: {
                    Label("Import", systemImage: "folder.badge.plus")
                }
            }
        }
        .sheet(isPresented: $showFilePicker) {
            FolderImportPicker { url in
                pendingImportURL = url
                showImportActionAlert = true
            }
        }
        .alert("Import Model", isPresented: $showImportActionAlert) {
            Button("Link (Bookmark)") {
                if let url = pendingImportURL {
                    handleImportedFolder(url)
                }
                pendingImportURL = nil
            }
            Button("Move to App (Documents)") {
                if let url = pendingImportURL {
                    moveImportedFolderToDocuments(url)
                }
                pendingImportURL = nil
            }
            Button("Cancel", role: .cancel) {
                pendingImportURL = nil
            }
        } message: {
            Text("Link keeps the model in its current location. Move to App copies it into the app's Documents folder for better reliability.")
        }
        .alert("Delete Model", isPresented: Binding(
            get: { modelToDelete != nil },
            set: { if !$0 { modelToDelete = nil } }
        )) {
            Button("Delete", role: .destructive) {
                if let model = modelToDelete {
                    deleteModel(model)
                }
                modelToDelete = nil
            }
            Button("Cancel", role: .cancel) { modelToDelete = nil }
        } message: {
            Text("Delete \"\(modelToDelete?.name ?? "")\" (\(String(format: "%.1f GB", modelToDelete?.sizeGB ?? 0.0)))? This cannot be undone.")
        }
        .sheet(item: $modelToExport) { model in
            FolderExportPicker(sourceURL: URL(fileURLWithPath: model.path)) { destURL in
                // moveToService already moved the files — just refresh the model list
                print("[export] Model moved to: \(destURL.path)")
                scanForModels()
            }
        }
        .overlay {
            if let progress = importProgress {
                VStack(spacing: 12) {
                    ProgressView()
                    Text(progress)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(24)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
        .onAppear { scanForModels(); restoreBookmarks() }
        .refreshable { scanForModels() }
        .onChange(of: downloadManager.activeDownload?.status) { _, newStatus in
            if newStatus == .complete {
                scanForModels()
            }
        }
    }

    private var headerView: some View {
        VStack(spacing: 8) {
            Image(systemName: "bolt.fill")
                .font(.system(size: 48))
                .foregroundStyle(.orange)
            Text("Flash-MoE")
                .font(.largeTitle.bold())
            Text("Run massive MoE models on iPhone")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical)
    }

    private func scanForModels() {
        isScanning = true
        localModels = []

        Task {
            let models = await ModelScanner.scanLocalModels()
            await MainActor.run {
                localModels = models
                isScanning = false
            }
        }
    }

    private func loadModel(_ model: LocalModel) {
        guard engine.state != .loading && engine.state != .generating else { return }
        selectedModel = model

        // Use picker value, or auto-detect for 397B if user hasn't set a preference
        let activeK: Int
        if activeExpertsK > 0 {
            activeK = activeExpertsK  // user selected a value
        } else {
            // Auto-reduce for 397B on constrained devices
            let is397B = model.path.lowercased().contains("397b") || model.name.lowercased().contains("397b")
            let deviceRAM = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
            activeK = (is397B && deviceRAM <= 16) ? 4 : 0
        }

        Task {
            do {
                try await engine.loadModel(
                    at: model.path,
                    useTiered: model.hasTiered,
                    activeExpertsK: activeK,
                    cacheIOSplit: cacheIOSplit,
                    verbose: true
                )
            } catch {
                // Error state is set by the engine
            }
        }
    }

    // MARK: - File Import

    private func handleImportedFolder(_ url: URL) {
        // Save a security-scoped bookmark so we can access this folder across launches
        guard url.startAccessingSecurityScopedResource() else {
            print("ERROR: Failed to access security-scoped resource")
            return
        }

        do {
            let bookmarkData = try url.bookmarkData(
                options: .minimalBookmark,
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            // Save bookmark to UserDefaults
            var bookmarks = UserDefaults.standard.array(forKey: "importedModelBookmarks") as? [Data] ?? []
            bookmarks.append(bookmarkData)
            UserDefaults.standard.set(bookmarks, forKey: "importedModelBookmarks")

            print("[import] Bookmarked external model folder: \(url.path)")
            scanForModels()
        } catch {
            print("ERROR: Failed to create bookmark: \(error)")
        }

        url.stopAccessingSecurityScopedResource()
    }

    private func deleteModel(_ model: LocalModel) {
        do {
            try FileManager.default.removeItem(atPath: model.path)
            print("[delete] Removed \(model.name) at \(model.path)")
            scanForModels()
        } catch {
            print("ERROR: Failed to delete \(model.name): \(error)")
        }
    }

    private func moveImportedFolderToDocuments(_ url: URL) {
        guard url.startAccessingSecurityScopedResource() else {
            print("ERROR: Failed to access security-scoped resource for move")
            return
        }

        let fm = FileManager.default
        guard let docsDir = fm.urls(for: .documentDirectory, in: .userDomainMask).first else {
            url.stopAccessingSecurityScopedResource()
            return
        }

        let destURL = docsDir.appendingPathComponent(url.lastPathComponent)

        importProgress = "Moving model to Documents..."
        print("[import] Moving \(url.path) -> \(destURL.path)")

        Task {
            do {
                // moveItem is instant if same filesystem, otherwise it copies
                try fm.moveItem(at: url, to: destURL)
                print("[import] Move succeeded")
            } catch {
                print("[import] Move failed: \(error). Trying copy...")
                await MainActor.run { importProgress = "Copying model to Documents (this may take a while)..." }
                do {
                    try fm.copyItem(at: url, to: destURL)
                    print("[import] Copy succeeded")
                } catch {
                    print("ERROR: Copy also failed: \(error)")
                }
            }

            url.stopAccessingSecurityScopedResource()

            await MainActor.run {
                importProgress = nil
                scanForModels()
            }
        }
    }

    private func restoreBookmarks() {
        guard let bookmarks = UserDefaults.standard.array(forKey: "importedModelBookmarks") as? [Data] else { return }

        for bookmark in bookmarks {
            var isStale = false
            if let url = try? URL(resolvingBookmarkData: bookmark, bookmarkDataIsStale: &isStale) {
                if !isStale {
                    _ = url.startAccessingSecurityScopedResource()
                }
            }
        }
    }

    private func moveModelToExternal(model: LocalModel, destination: URL) {
        Task {
            let fm = FileManager.default
            let destPath = destination.appendingPathComponent(URL(fileURLWithPath: model.path).lastPathComponent)

            guard destination.startAccessingSecurityScopedResource() else {
                print("ERROR: Cannot access destination")
                return
            }
            defer { destination.stopAccessingSecurityScopedResource() }

            do {
                // Move (not copy) — instant on same filesystem
                try fm.moveItem(at: URL(fileURLWithPath: model.path), to: destPath)
                print("[export] Moved \(model.name) to \(destPath.path)")
                await MainActor.run { scanForModels() }
            } catch {
                print("ERROR: Move failed: \(error). Trying copy instead...")
                // If move fails (cross-volume), this would be slow for 300GB
                // but at least it works
                do {
                    try fm.copyItem(at: URL(fileURLWithPath: model.path), to: destPath)
                    try fm.removeItem(at: URL(fileURLWithPath: model.path))
                    print("[export] Copied + deleted \(model.name) to \(destPath.path)")
                    await MainActor.run { scanForModels() }
                } catch {
                    print("ERROR: Copy also failed: \(error)")
                }
            }
        }
    }
}

// MARK: - Folder Import Picker

struct FolderImportPicker: UIViewControllerRepresentable {
    let onPick: (URL) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
        picker.allowsMultipleSelection = false
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            onPick(url)
        }
    }
}

// MARK: - Folder Export Picker (pick destination to move model to)

struct FolderExportPicker: UIViewControllerRepresentable {
    let sourceURL: URL
    let onPick: (URL) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        // moveToService: shows full Files browser, user picks destination folder.
        // iOS moves the directory to the chosen location.
        let picker = UIDocumentPickerViewController(urls: [sourceURL], in: .moveToService)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            onPick(url)
        }
    }
}

// MARK: - Model Row

struct ModelRow: View {
    let model: LocalModel
    let isLoading: Bool

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.name)
                    .font(.headline)

                HStack(spacing: 8) {
                    if model.hasTiered {
                        QuantBadge(text: "Tiered", color: .green)
                    } else if model.has4bit {
                        QuantBadge(text: "4-bit", color: .blue)
                    } else if model.has2bit {
                        QuantBadge(text: "2-bit", color: .orange)
                    }

                    Text(String(format: "%.1f GB", model.sizeGB))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            if isLoading {
                ProgressView()
            } else {
                Image(systemName: "chevron.right")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

struct QuantBadge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.caption2.bold())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }
}

// MARK: - Model Scanner

enum ModelScanner {
    /// Scan common locations for Flash-MoE model directories
    static func scanLocalModels() async -> [LocalModel] {
        var models: [LocalModel] = []
        let fm = FileManager.default

        // Scan app Documents directory
        if let docsDir = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            await scanDirectory(docsDir.path, into: &models)
        }

        // Scan bookmarked external folders (imported via Files picker)
        if let bookmarks = UserDefaults.standard.array(forKey: "importedModelBookmarks") as? [Data] {
            for bookmark in bookmarks {
                var isStale = false
                if let url = try? URL(resolvingBookmarkData: bookmark, bookmarkDataIsStale: &isStale),
                   !isStale {
                    let accessed = url.startAccessingSecurityScopedResource()
                    // Check if the folder itself is a model
                    if FlashMoEEngine.validateModel(at: url.path) {
                        let size = directorySize(at: url.path)
                        let hasTiered = fm.fileExists(atPath: url.appendingPathComponent("packed_experts_tiered/layer_00.bin").path)
                        let has4bit = fm.fileExists(atPath: url.appendingPathComponent("packed_experts/layer_00.bin").path)
                        let has2bit = fm.fileExists(atPath: url.appendingPathComponent("packed_experts_2bit/layer_00.bin").path)
                        models.append(LocalModel(
                            name: "📁 " + url.lastPathComponent,
                            path: url.path,
                            sizeBytes: size,
                            hasTiered: hasTiered,
                            has4bit: has4bit,
                            has2bit: has2bit
                        ))
                    } else {
                        // Scan subdirectories
                        await scanDirectory(url.path, into: &models)
                    }
                    if accessed { url.stopAccessingSecurityScopedResource() }
                }
            }
        }

        return models.sorted { $0.name < $1.name }
    }

    private static func scanDirectory(_ path: String, into models: inout [LocalModel]) async {
        let fm = FileManager.default

        guard let entries = try? fm.contentsOfDirectory(atPath: path) else { return }

        for entry in entries {
            let fullPath = (path as NSString).appendingPathComponent(entry)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: fullPath, isDirectory: &isDir), isDir.boolValue else { continue }

            // Check if it's a valid model
            if FlashMoEEngine.validateModel(at: fullPath) {
                // Protect model files from iOS storage optimization / purging
                excludeFromBackup(URL(fileURLWithPath: fullPath))
                let size = directorySize(at: fullPath)
                let hasTiered = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts_tiered/layer_00.bin"))
                let has4bit = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts/layer_00.bin"))
                let has2bit = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts_2bit/layer_00.bin"))

                models.append(LocalModel(
                    name: entry,
                    path: fullPath,
                    sizeBytes: size,
                    hasTiered: hasTiered,
                    has4bit: has4bit,
                    has2bit: has2bit
                ))
            }
        }
    }

    private static func directorySize(at path: String) -> UInt64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: path) else { return 0 }
        var total: UInt64 = 0
        while let file = enumerator.nextObject() as? String {
            let fullPath = (path as NSString).appendingPathComponent(file)
            if let attrs = try? fm.attributesOfItem(atPath: fullPath),
               let size = attrs[.size] as? UInt64 {
                total += size
            }
        }
        return total
    }

    /// Mark a directory (and its contents) as excluded from iCloud backup and
    /// iOS storage optimization, preventing the system from purging model files.
    private static func excludeFromBackup(_ url: URL) {
        var url = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)

        // Also mark all files inside
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: nil) else { return }
        while let fileURL = enumerator.nextObject() as? URL {
            var fileURL = fileURL
            try? fileURL.setResourceValues(values)
        }
    }
}
