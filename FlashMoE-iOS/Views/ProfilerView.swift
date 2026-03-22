/*
 * ProfilerView.swift — Lightweight resource profiler overlay
 *
 * Displays real-time system metrics during inference:
 * memory (RSS + available), thermal state, CPU usage,
 * and engine stats (tok/s, TTFT).
 *
 * All APIs are public (mach_task_info, os_proc_available_memory,
 * ProcessInfo.thermalState) — no entitlements needed.
 */

import SwiftUI
import Darwin.Mach

// MARK: - System Metrics Sampler

@Observable
final class SystemMetrics: @unchecked Sendable {
    private(set) var residentMemoryMB: Double = 0
    private(set) var availableMemoryMB: Double = 0
    private(set) var cpuUsagePercent: Double = 0
    private(set) var thermalState: ProcessInfo.ThermalState = .nominal

    private var prevCPUTime: Double = 0
    private var prevSampleTime: CFAbsoluteTime = 0

    func sample() {
        residentMemoryMB = Self.getResidentMemory()
        availableMemoryMB = Self.getAvailableMemory()
        cpuUsagePercent = sampleCPU()
        thermalState = ProcessInfo.processInfo.thermalState
    }

    // MARK: - Memory

    private static func getResidentMemory() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / (1024 * 1024)
    }

    private static func getAvailableMemory() -> Double {
        #if os(iOS)
        // iOS: use os_proc_available_memory()
        return Double(os_proc_available_memory()) / (1024 * 1024)
        #elseif os(macOS)
        // macOS: estimate available memory using host_statistics64 and page counts
        var vmStats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        var size: vm_size_t = 0
        let host = mach_host_self()

        // Get page size
        let kerrPage = host_page_size(host, &size)
        guard kerrPage == KERN_SUCCESS else { return 0 }

        // Fetch VM statistics
        let result: kern_return_t = withUnsafeMutablePointer(to: &vmStats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                host_statistics64(host, HOST_VM_INFO64, intPtr, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }

        // Consider free + inactive pages as "available"
        let freePages = UInt64(vmStats.free_count)
        let inactivePages = UInt64(vmStats.inactive_count)
        let speculativePages = UInt64(vmStats.speculative_count)

        let availableBytes = (freePages + inactivePages + speculativePages) * UInt64(size)
        return Double(availableBytes) / (1024 * 1024)
        #else
        // Other platforms: not available — return 0
        return 0
        #endif
    }

    // MARK: - CPU

    private func sampleCPU() -> Double {
        let now = CFAbsoluteTimeGetCurrent()
        let totalCPU = Self.getThreadCPUTime()

        defer {
            prevCPUTime = totalCPU
            prevSampleTime = now
        }

        guard prevSampleTime > 0 else { return 0 }
        let elapsed = now - prevSampleTime
        guard elapsed > 0 else { return 0 }

        let cpuDelta = totalCPU - prevCPUTime
        // Normalize to percentage (cpuDelta is in seconds of CPU time)
        return min((cpuDelta / elapsed) * 100.0, 999.0)
    }

    private static func getThreadCPUTime() -> Double {
        var threadList: thread_act_array_t?
        var threadCount: mach_msg_type_number_t = 0
        let result = task_threads(mach_task_self_, &threadList, &threadCount)
        guard result == KERN_SUCCESS, let threads = threadList else { return 0 }
        defer {
            vm_deallocate(mach_task_self_, vm_address_t(bitPattern: threads), vm_size_t(Int(threadCount) * MemoryLayout<thread_t>.size))
        }

        var total: Double = 0
        for i in 0..<Int(threadCount) {
            var info = thread_basic_info()
            var infoCount = mach_msg_type_number_t(MemoryLayout<thread_basic_info_data_t>.size / MemoryLayout<natural_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(infoCount)) {
                    thread_info(threads[i], thread_flavor_t(THREAD_BASIC_INFO), $0, &infoCount)
                }
            }
            if kr == KERN_SUCCESS {
                total += Double(info.user_time.seconds) + Double(info.user_time.microseconds) / 1_000_000
                total += Double(info.system_time.seconds) + Double(info.system_time.microseconds) / 1_000_000
            }
        }
        return total
    }
}

// MARK: - Profiler View

struct ProfilerView: View {
    let engine: FlashMoEEngine
    @State private var metrics = SystemMetrics()
    @State private var timer: Timer?

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: "gauge.with.dots.needle.50percent")
                    .foregroundStyle(.orange)
                Text("Profiler")
                    .font(.caption.bold())
                Spacer()
                thermalBadge
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 4)

            Divider().opacity(0.3)

            // Metrics grid
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
            ], spacing: 6) {
                metricCell(
                    icon: "memorychip",
                    label: "RSS",
                    value: String(format: "%.0f MB", metrics.residentMemoryMB)
                )
                metricCell(
                    icon: "memorychip.fill",
                    label: "Free",
                    value: String(format: "%.0f MB", metrics.availableMemoryMB)
                )
                metricCell(
                    icon: "cpu",
                    label: "CPU",
                    value: String(format: "%.0f%%", metrics.cpuUsagePercent)
                )
                metricCell(
                    icon: "speedometer",
                    label: "tok/s",
                    value: String(format: "%.1f", engine.tokensPerSecond)
                )
                metricCell(
                    icon: "number",
                    label: "Tokens",
                    value: "\(engine.tokensGenerated)"
                )
                metricCell(
                    icon: "timer",
                    label: "TTFT",
                    value: engine.timeToFirstToken > 0
                        ? String(format: "%.0f ms", engine.timeToFirstToken)
                        : "--"
                )
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .shadow(color: .black.opacity(0.15), radius: 8, y: 2)
        .padding(.horizontal)
        .onAppear { startSampling() }
        .onDisappear { stopSampling() }
    }

    // MARK: - Subviews

    private func metricCell(icon: String, label: String, value: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 14)
            VStack(alignment: .leading, spacing: 0) {
                Text(label)
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
                Text(value)
                    .font(.caption.monospacedDigit().bold())
            }
            Spacer()
        }
    }

    private var thermalBadge: some View {
        HStack(spacing: 3) {
            Circle()
                .fill(thermalColor)
                .frame(width: 6, height: 6)
            Text(thermalLabel)
                .font(.system(size: 9).bold())
                .foregroundStyle(thermalColor)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(thermalColor.opacity(0.12))
        .clipShape(Capsule())
    }

    private var thermalColor: Color {
        switch metrics.thermalState {
        case .nominal: return .green
        case .fair: return .yellow
        case .serious: return .orange
        case .critical: return .red
        @unknown default: return .gray
        }
    }

    private var thermalLabel: String {
        switch metrics.thermalState {
        case .nominal: return "Cool"
        case .fair: return "Warm"
        case .serious: return "Hot"
        case .critical: return "Critical"
        @unknown default: return "?"
        }
    }

    // MARK: - Sampling

    private func startSampling() {
        metrics.sample() // initial
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            metrics.sample()
        }
    }

    private func stopSampling() {
        timer?.invalidate()
        timer = nil
    }
}

