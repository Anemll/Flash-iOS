/*
 * DownloadManager.swift — Background download orchestration for HuggingFace models
 *
 * Uses URLSession background downloads that survive app termination.
 * Downloads files sequentially within a model for clean progress tracking.
 * State persisted to downloads.json for resume across app launches.
 */

import Foundation
import Observation

// MARK: - Download State

enum DownloadStatus: String, Codable, Sendable {
    case downloading
    case paused
    case failed
    case complete
}

struct DownloadState: Codable {
    let catalogId: String
    let repoId: String
    var completedFiles: [String]
    var completedBytes: UInt64
    var currentFile: String?
    var status: DownloadStatus
    var errorMessage: String?
}

// MARK: - DownloadManager

@Observable
final class DownloadManager: NSObject, @unchecked Sendable {
    static let shared = DownloadManager()

    // Observable state
    private(set) var activeDownload: DownloadState?
    private(set) var overallProgress: Double = 0
    private(set) var currentFileProgress: Double = 0
    private(set) var bytesDownloaded: UInt64 = 0
    private(set) var totalBytes: UInt64 = 0
    private(set) var error: String?
    private(set) var downloadSpeed: Double = 0 // bytes/sec

    // Background session callback
    var backgroundCompletionHandler: (() -> Void)?

    // Private state
    private var backgroundSession: URLSession!
    private var currentTask: URLSessionDownloadTask?
    private var currentEntry: CatalogEntry?
    private var resumeData: Data?
    private var speedSampleTime: Date?
    private var speedSampleBytes: UInt64 = 0

    private static let sessionIdentifier = "com.flashmoe.model-download"

    // MARK: - Initialization

    override private init() {
        super.init()
        let config = URLSessionConfiguration.background(withIdentifier: Self.sessionIdentifier)
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.allowsCellularAccess = true
        backgroundSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        // Restore persisted state
        loadPersistedState()

        // Reconnect to any in-flight background tasks
        backgroundSession.getTasksWithCompletionHandler { [weak self] _, _, downloadTasks in
            if let task = downloadTasks.first {
                self?.currentTask = task
            }
        }
    }

    // MARK: - Public API

    func startDownload(entry: CatalogEntry) {
        // Allow starting if no active download, or previous one finished/failed
        if let status = activeDownload?.status, status == .downloading || status == .paused {
            if activeDownload?.catalogId != entry.id {
                error = "A different download is already in progress"
                return
            }
            // Same model — resume instead
            resumeDownload()
            return
        }

        // Clear stale state from previous download
        if activeDownload != nil {
            activeDownload = nil
            clearPersistedState()
        }

        // Check disk space
        let available = availableDiskSpace()
        if available < entry.totalSizeBytes {
            let needed = formatBytes(entry.totalSizeBytes)
            let have = formatBytes(available)
            error = "Not enough space: \(needed) needed, \(have) available"
            return
        }

        error = nil
        currentEntry = entry
        totalBytes = entry.totalSizeBytes

        // Create model directory
        let modelDir = modelDirectory(for: entry.id)
        createDirectoryStructure(for: entry, at: modelDir)

        activeDownload = DownloadState(
            catalogId: entry.id,
            repoId: entry.repoId,
            completedFiles: [],
            completedBytes: 0,
            currentFile: nil,
            status: .downloading
        )
        persistState()
        downloadNextFile()
    }

    func pauseDownload() {
        guard activeDownload?.status == .downloading else { return }

        currentTask?.cancel(byProducingResumeData: { [weak self] data in
            guard let self else { return }
            self.resumeData = data
            self.activeDownload?.status = .paused
            self.persistState()
            self.currentTask = nil
        })
    }

    func resumeDownload() {
        guard activeDownload?.status == .paused || activeDownload?.status == .failed else { return }

        // Resolve the catalog entry
        if currentEntry == nil, let catalogId = activeDownload?.catalogId {
            currentEntry = ModelCatalog.models.first { $0.id == catalogId }
        }
        guard currentEntry != nil else {
            error = "Cannot find model in catalog"
            return
        }

        error = nil
        activeDownload?.status = .downloading
        activeDownload?.errorMessage = nil
        totalBytes = currentEntry?.totalSizeBytes ?? 0
        persistState()

        if let resumeData {
            let task = backgroundSession.downloadTask(withResumeData: resumeData)
            task.resume()
            currentTask = task
            self.resumeData = nil
        } else {
            downloadNextFile()
        }
    }

    func cancelDownload() {
        currentTask?.cancel()
        currentTask = nil
        resumeData = nil

        if let catalogId = activeDownload?.catalogId {
            let dir = modelDirectory(for: catalogId)
            try? FileManager.default.removeItem(at: dir)
        }

        activeDownload = nil
        overallProgress = 0
        currentFileProgress = 0
        bytesDownloaded = 0
        totalBytes = 0
        error = nil
        currentEntry = nil
        clearPersistedState()
    }

    func deleteModel(catalogId: String) {
        let dir = modelDirectory(for: catalogId)
        try? FileManager.default.removeItem(at: dir)

        if activeDownload?.catalogId == catalogId {
            activeDownload = nil
            clearPersistedState()
        }
    }

    func isModelDownloaded(_ catalogId: String) -> Bool {
        let dir = modelDirectory(for: catalogId)
        return FlashMoEEngine.validateModel(at: dir.path)
    }

    func modelPath(for catalogId: String) -> String {
        modelDirectory(for: catalogId).path
    }

    // MARK: - File Management

    private func modelDirectory(for catalogId: String) -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent(catalogId)
    }

    private func createDirectoryStructure(for entry: CatalogEntry, at baseURL: URL) {
        let fm = FileManager.default
        try? fm.createDirectory(at: baseURL, withIntermediateDirectories: true)

        // Create subdirectories for expert files
        var subdirs = Set<String>()
        for file in entry.files {
            let url = baseURL.appendingPathComponent(file.filename)
            let parent = url.deletingLastPathComponent()
            if parent != baseURL {
                subdirs.insert(parent.path)
            }
        }
        for dir in subdirs {
            try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        }
    }

    // MARK: - Sequential Download Engine

    private func downloadNextFile() {
        guard var state = activeDownload, let entry = currentEntry else { return }

        // Find next file to download
        let nextFile = entry.files.first { !state.completedFiles.contains($0.filename) }

        guard let file = nextFile else {
            // All files downloaded
            state.status = .complete
            state.currentFile = nil
            activeDownload = state
            overallProgress = 1.0
            persistState()

            // Protect from iOS storage optimization (exclude from backup/purge)
            let dir = modelDirectory(for: entry.id)
            var dirURL = dir
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try? dirURL.setResourceValues(values)
            if let enumerator = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil) {
                while let fileURL = enumerator.nextObject() as? URL {
                    var fURL = fileURL
                    try? fURL.setResourceValues(values)
                }
            }

            // Validate the model
            if !FlashMoEEngine.validateModel(at: dir.path) {
                error = "Download complete but model validation failed"
                state.status = .failed
                state.errorMessage = "Validation failed — some files may be corrupt"
                activeDownload = state
                persistState()
            }
            return
        }

        state.currentFile = file.filename
        activeDownload = state
        persistState()

        let url = entry.downloadURL(for: file)
        let task = backgroundSession.downloadTask(with: url)
        task.taskDescription = file.filename
        task.resume()
        currentTask = task
        currentFileProgress = 0
        speedSampleTime = Date()
        speedSampleBytes = bytesDownloaded
    }

    // MARK: - Disk Space

    private func availableDiskSpace() -> UInt64 {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        guard let values = try? docs.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
              let capacity = values.volumeAvailableCapacityForImportantUsage else {
            return 0
        }
        return UInt64(capacity)
    }

    // MARK: - State Persistence

    private var stateFileURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("downloads.json")
    }

    private func persistState() {
        guard let state = activeDownload else { return }
        if let data = try? JSONEncoder().encode(state) {
            try? data.write(to: stateFileURL)
        }
    }

    private func clearPersistedState() {
        try? FileManager.default.removeItem(at: stateFileURL)
    }

    private func loadPersistedState() {
        guard let data = try? Data(contentsOf: stateFileURL),
              let state = try? JSONDecoder().decode(DownloadState.self, from: data) else {
            return
        }

        activeDownload = state
        currentEntry = ModelCatalog.models.first { $0.id == state.catalogId }

        if let entry = currentEntry {
            totalBytes = entry.totalSizeBytes
            bytesDownloaded = state.completedBytes
            overallProgress = totalBytes > 0 ? Double(bytesDownloaded) / Double(totalBytes) : 0
        }
    }

    // MARK: - Formatting

    private func formatBytes(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / (1024 * 1024 * 1024)
        if gb >= 1 {
            return String(format: "%.1f GB", gb)
        }
        let mb = Double(bytes) / (1024 * 1024)
        return String(format: "%.0f MB", mb)
    }
}

// MARK: - URLSessionDownloadDelegate

extension DownloadManager: URLSessionDownloadDelegate {

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let entry = currentEntry,
              let filename = downloadTask.taskDescription else { return }

        // Check HTTP status code — HuggingFace returns 200 HTML pages for 404s
        if let httpResponse = downloadTask.response as? HTTPURLResponse,
           httpResponse.statusCode != 200 {
            let statusCode = httpResponse.statusCode
            DispatchQueue.main.async { [weak self] in
                guard let self, var state = self.activeDownload else { return }
                self.error = "HTTP \(statusCode) downloading \(filename)"
                state.status = .failed
                state.errorMessage = self.error
                self.activeDownload = state
                self.persistState()
            }
            return
        }

        // Move from temp to model directory (must happen synchronously before this method returns)
        let dest = modelDirectory(for: entry.id).appendingPathComponent(filename)
        let fm = FileManager.default
        try? fm.removeItem(at: dest)

        var moveError: Error?
        do {
            try fm.moveItem(at: location, to: dest)
        } catch {
            moveError = error
        }

        // Compute file size on this thread before dispatching
        let actualSize: UInt64
        if moveError == nil {
            let attrs = try? fm.attributesOfItem(atPath: dest.path)
            actualSize = attrs?[.size] as? UInt64 ?? 0
        } else {
            actualSize = 0
        }

        let expectedSize = entry.files.first(where: { $0.filename == filename })?.sizeBytes ?? 0

        // Check if we got an HTML error page instead of the actual file
        // (HuggingFace sometimes returns 200 with HTML for missing LFS files)
        if actualSize > 0 && actualSize < 10_000 && expectedSize > 100_000 {
            // Downloaded file is suspiciously small — likely an error page
            try? fm.removeItem(at: dest)
            DispatchQueue.main.async { [weak self] in
                guard let self, var state = self.activeDownload else { return }
                self.error = "File \(filename) not found on server (got \(actualSize) bytes, expected \(self.formatBytes(expectedSize)))"
                state.status = .failed
                state.errorMessage = self.error
                self.activeDownload = state
                self.persistState()
            }
            return
        }

        // All @Observable mutations on main thread
        DispatchQueue.main.async { [weak self] in
            guard let self, var state = self.activeDownload else { return }

            if let moveError {
                self.error = "Failed to save \(filename): \(moveError.localizedDescription)"
                state.status = .failed
                state.errorMessage = self.error
                self.activeDownload = state
                self.persistState()
                return
            }

            // Validate file size
            if actualSize > 0 && expectedSize > 0 && actualSize < expectedSize * 9 / 10 {
                self.error = "File \(filename) is too small (\(actualSize) vs expected \(expectedSize))"
                state.status = .failed
                state.errorMessage = self.error
                self.activeDownload = state
                self.persistState()
                return
            }

            state.completedBytes += actualSize > 0 ? actualSize : expectedSize
            state.completedFiles.append(filename)
            state.currentFile = nil
            self.activeDownload = state
            self.bytesDownloaded = state.completedBytes
            self.persistState()

            // Start next file
            self.downloadNextFile()
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        // Compute values on background thread
        let fileProgress = totalBytesExpectedToWrite > 0
            ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite) : 0
        let completed = activeDownload?.completedBytes ?? 0
        let currentTotal = completed + UInt64(totalBytesWritten)
        let total = totalBytes
        let overall = total > 0 ? Double(currentTotal) / Double(total) : 0

        // Speed sampling (non-observable state, safe on background)
        var newSpeed: Double?
        if let sampleTime = speedSampleTime, Date().timeIntervalSince(sampleTime) >= 2 {
            let elapsed = Date().timeIntervalSince(sampleTime)
            let delta = currentTotal - speedSampleBytes
            newSpeed = Double(delta) / elapsed
            speedSampleTime = Date()
            speedSampleBytes = currentTotal
        }

        // All @Observable mutations on main thread
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.currentFileProgress = fileProgress
            self.bytesDownloaded = currentTotal
            self.overallProgress = overall
            if let newSpeed {
                self.downloadSpeed = newSpeed
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        guard let error else { return }

        let nsError = error as NSError
        if nsError.code == NSURLErrorCancelled {
            return
        }

        // Save resume data if available (non-observable)
        if let data = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
            self.resumeData = data
        }

        let errorMsg = error.localizedDescription

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.error = errorMsg
            self.activeDownload?.status = .failed
            self.activeDownload?.errorMessage = errorMsg
            self.persistState()
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async { [weak self] in
            self?.backgroundCompletionHandler?()
            self?.backgroundCompletionHandler = nil
        }
    }
}
