/*
 * ModelListView.swift — Model discovery and loading
 *
 * Lists locally available models and allows downloading from HuggingFace.
 * For v1, supports loading models already present on device.
 */

import SwiftUI

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
    @AppStorage("cacheIOSplit") private var cacheIOSplit: Int = 4
    @AppStorage("chatTemplateEnabled") private var chatTemplateEnabled: Bool = true
    @AppStorage("lastModelPath") private var lastModelPath: String = ""
    @AppStorage("maxGenerationTokens") private var maxGenerationTokens: Int = 2048
    @AppStorage("gpuCombineEnabled") private var gpuCombineEnabled: Bool = true
    @AppStorage("gpuLinearAttnEnabled") private var gpuLinearAttnEnabled: Bool = true
    @AppStorage("expertPrefetchEnabled") private var expertPrefetchEnabled: Bool = true
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

            Section("I/O Settings") {
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

            Section("Chat Settings") {
                Toggle("Chat Template", isOn: $chatTemplateEnabled)
                Text("Wraps prompts in Qwen chat format (<|im_start|>). Disable for smoke test models or raw text mode.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Generation") {
                Picker("Max Tokens", selection: $maxGenerationTokens) {
                    Text("256").tag(256)
                    Text("512").tag(512)
                    Text("1024").tag(1024)
                    Text("2048").tag(2048)
                    Text("4096").tag(4096)
                }
                .pickerStyle(.menu)
                Text("Maximum tokens per response. Higher values allow longer replies but take more time.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("GPU Optimizations") {
                Toggle("Fused CMD3 (combine+norm)", isOn: $gpuCombineEnabled)
                    .onChange(of: gpuCombineEnabled) { _, val in engine.setGPUCombine(val) }
                Toggle("GPU Linear Attention", isOn: $gpuLinearAttnEnabled)
                    .onChange(of: gpuLinearAttnEnabled) { _, val in engine.setGPULinearAttn(val) }
                Toggle("Expert Prefetch (async pread)", isOn: $expertPrefetchEnabled)
                    .onChange(of: expertPrefetchEnabled) { _, val in engine.setExpertPrefetch(val) }
                Text("Disable to measure per-optimization impact via timing profile.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Profile") {
                Text("Load a model, then use the \(Image(systemName: "ellipsis.circle")) menu in Chat to run a timing profile.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
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
        .onAppear { scanForModels() }
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
        lastModelPath = model.path

        Task {
            do {
                try await engine.loadModel(
                    at: model.path,
                    maxContext: 4096,
                    useTiered: model.hasTiered,
                    use2bit: model.has2bit && !model.hasTiered && !model.has4bit,
                    cacheIOSplit: cacheIOSplit,
                    verbose: true
                )
                // Apply optimization toggles
                engine.setGPUCombine(gpuCombineEnabled)
                engine.setGPULinearAttn(gpuLinearAttnEnabled)
                engine.setExpertPrefetch(expertPrefetchEnabled)
            } catch {
                // Error state is set by the engine
            }
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

        // Note: App Groups container removed — no entitlement configured.
        // Models are discovered from the Documents directory (accessible via Files.app).

        return models.sorted { $0.name < $1.name }
    }

    private static func scanDirectory(_ path: String, into models: inout [LocalModel]) async {
        let fm = FileManager.default

        print("[model-scan] Scanning: \(path)")
        guard let entries = try? fm.contentsOfDirectory(atPath: path) else {
            print("[model-scan] ERROR: cannot list \(path)")
            return
        }
        print("[model-scan] Found \(entries.count) entries: \(entries.sorted().joined(separator: ", "))")

        for entry in entries {
            let fullPath = (path as NSString).appendingPathComponent(entry)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: fullPath, isDirectory: &isDir), isDir.boolValue else { continue }

            let valid = FlashMoEEngine.validateModel(at: fullPath)
            if !valid {
                // Debug: show why validation failed
                let hasConfig = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("config.json"))
                let hasWeights = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("model_weights.bin"))
                let hasManifest = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("model_weights.json"))
                let hasExperts4 = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts/layer_00.bin"))
                let hasExpertsT = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts_tiered/layer_00.bin"))
                let hasExperts2 = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts_2bit/layer_00.bin"))
                print("[model-scan] SKIP '\(entry)': config=\(hasConfig) weights=\(hasWeights) manifest=\(hasManifest) experts(4bit=\(hasExperts4) tiered=\(hasExpertsT) 2bit=\(hasExperts2))")
                continue
            }

            // Protect model files from iOS storage optimization / purging
            excludeFromBackup(URL(fileURLWithPath: fullPath))
            let size = directorySize(at: fullPath)
            let hasTiered = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts_tiered/layer_00.bin"))
            let has4bit = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts/layer_00.bin"))
            let has2bit = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts_2bit/layer_00.bin"))

            print("[model-scan] OK '\(entry)': size=\(size / (1024*1024))MB tiered=\(hasTiered) 4bit=\(has4bit) 2bit=\(has2bit)")
            models.append(LocalModel(
                name: entry,
                path: fullPath,
                sizeBytes: size,
                hasTiered: hasTiered,
                has4bit: has4bit,
                has2bit: has2bit
            ))
        }
        print("[model-scan] Total valid models: \(models.count)")
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

// MARK: - Profile Result Sheet

struct ProfileResultSheet: View {
    let result: String
    @Environment(\.dismiss) private var dismiss

    /// Extract model name from the report "Model:   xxx" line
    private var modelName: String {
        for line in result.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("Model:") {
                return trimmed.replacingOccurrences(of: "Model:", with: "").trimmingCharacters(in: .whitespaces)
            }
        }
        return ""
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(result)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .navigationTitle(modelName.isEmpty ? "Timing Profile" : "Profile: \(modelName)")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button {
                        #if os(iOS)
                        UIPasteboard.general.string = result
                        #else
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(result, forType: .string)
                        #endif
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                }
            }
        }
        #if os(iOS)
        .presentationDetents([.medium, .large])
        #else
        .frame(minWidth: 450, minHeight: 400)
        #endif
    }
}
