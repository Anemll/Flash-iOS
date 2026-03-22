/*
 * ModelDownloadRow.swift — Download progress UI for catalog entries
 *
 * Shows different states: available for download, downloading with progress,
 * paused, failed with retry, or completed.
 */

import SwiftUI

struct ModelDownloadRow: View {
    let entry: CatalogEntry
    let downloadManager: DownloadManager
    let isDownloaded: Bool

    private var quantColor: Color {
        switch entry.quantization.lowercased() {
        case "4-bit": return .blue
        case "2-bit": return .orange
        case "tiered": return .purple
        default: return .gray
        }
    }

    private var isActiveDownload: Bool {
        downloadManager.activeDownload?.catalogId == entry.id
    }

    private var downloadStatus: DownloadStatus? {
        guard isActiveDownload else { return nil }
        return downloadManager.activeDownload?.status
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header row
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.displayName)
                        .font(.headline)
                    Text(entry.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                Spacer()

                QuantBadge(text: entry.quantization, color: quantColor)
            }

            // Status-specific content
            if isDownloaded {
                downloadedView
            } else if let status = downloadStatus {
                switch status {
                case .downloading:
                    downloadingView
                case .paused:
                    pausedView
                case .failed:
                    failedView
                case .complete:
                    downloadedView
                }
            } else {
                availableView
            }
        }
        .padding(.vertical, 4)
    }

    // MARK: - State Views

    private var availableView: some View {
        HStack {
            Label(formatSize(entry.totalSizeBytes), systemImage: "internaldrive")
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            Button {
                downloadManager.startDownload(entry: entry)
            } label: {
                Label("Download", systemImage: "arrow.down.circle.fill")
                    .font(.subheadline.weight(.medium))
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
        }
    }

    private var downloadingView: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: downloadManager.overallProgress)
                .tint(.blue)

            HStack {
                // Progress text
                Text("\(formatSize(downloadManager.bytesDownloaded)) / \(formatSize(downloadManager.totalBytes))")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if downloadManager.downloadSpeed > 0 {
                    Text("(\(formatSpeed(downloadManager.downloadSpeed)))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                // Current file
                if let currentFile = downloadManager.activeDownload?.currentFile {
                    Text(shortFilename(currentFile))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                // Pause button
                Button {
                    downloadManager.pauseDownload()
                } label: {
                    Image(systemName: "pause.circle.fill")
                        .foregroundStyle(.orange)
                }

                // Cancel button
                Button {
                    downloadManager.cancelDownload()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.red)
                }
            }
        }
    }

    private var pausedView: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: downloadManager.overallProgress)
                .tint(.orange)

            HStack {
                Text("Paused — \(formatSize(downloadManager.bytesDownloaded)) / \(formatSize(downloadManager.totalBytes))")
                    .font(.caption)
                    .foregroundStyle(.orange)

                Spacer()

                Button {
                    downloadManager.resumeDownload()
                } label: {
                    Label("Resume", systemImage: "play.circle.fill")
                        .font(.subheadline.weight(.medium))
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)

                Button {
                    downloadManager.cancelDownload()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.red)
                }
            }
        }
    }

    private var failedView: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let error = downloadManager.activeDownload?.errorMessage ?? downloadManager.error {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            HStack {
                Text("\(formatSize(downloadManager.bytesDownloaded)) downloaded")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                Button {
                    downloadManager.resumeDownload()
                } label: {
                    Label("Retry", systemImage: "arrow.clockwise.circle.fill")
                        .font(.subheadline.weight(.medium))
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)

                Button {
                    downloadManager.cancelDownload()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var downloadedView: some View {
        HStack {
            Label("Downloaded", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)

            Spacer()

            Button(role: .destructive) {
                downloadManager.deleteModel(catalogId: entry.id)
            } label: {
                Image(systemName: "trash")
                    .font(.caption)
            }
        }
    }

    // MARK: - Formatting

    private func formatSize(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / (1024.0 * 1024.0 * 1024.0)
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        let mb = Double(bytes) / (1024.0 * 1024.0)
        return String(format: "%.0f MB", mb)
    }

    private func formatSpeed(_ bytesPerSec: Double) -> String {
        let mbps = bytesPerSec / (1024 * 1024)
        return String(format: "%.1f MB/s", mbps)
    }

    private func shortFilename(_ filename: String) -> String {
        (filename as NSString).lastPathComponent
    }
}

// QuantBadge is defined in ModelListView.swift
