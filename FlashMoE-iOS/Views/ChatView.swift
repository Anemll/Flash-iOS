/*
 * ChatView.swift — Chat interface for Flash-MoE inference
 *
 * Streaming token display, stats overlay, conversation history.
 */

import SwiftUI

// MARK: - Chat Message Model

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var text: String
    let timestamp: Date

    enum Role {
        case user
        case assistant
    }
}

// MARK: - Chat View

struct ChatView: View {
    @Environment(FlashMoEEngine.self) private var engine
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var isGenerating = false
    @State private var showStats = false
    @AppStorage("chatTemplateEnabled") private var chatTemplateEnabled: Bool = true
    @AppStorage("maxGenerationTokens") private var maxGenerationTokens: Int = 2048
    @State private var showModelInfo = false
    @State private var showProfiler = false
    @State private var isProfileRunning = false
    @State private var profileResult: String?
    @State private var showProfileResult = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                    .padding()
                }
                .onTapGesture { inputFocused = false }
                .onChange(of: messages.count) {
                    if let last = messages.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            // Stats bar
            if isGenerating || engine.tokensGenerated > 0 {
                StatsBar(
                    tokensPerSecond: engine.tokensPerSecond,
                    tokensGenerated: engine.tokensGenerated,
                    isGenerating: isGenerating
                )
            }

            // Profiler panel (between messages and input)
            if showProfiler {
                Divider()
                ProfilerView(engine: engine)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            Divider()

            // Input bar
            HStack(spacing: 12) {
                TextField("Message...", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...5)
                    .focused($inputFocused)
                    .onSubmit { sendMessage() }
                    .disabled(isGenerating)

                if isGenerating {
                    Button(action: { engine.cancel() }) {
                        Image(systemName: "stop.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.red)
                    }
                } else {
                    Button(action: sendMessage) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                            .foregroundStyle(inputText.isEmpty ? .gray : .blue)
                    }
                    .disabled(inputText.isEmpty)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
        }
        .animation(.easeInOut(duration: 0.25), value: showProfiler)
        .navigationTitle("Flash-MoE")
#if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button(action: { showModelInfo = true }) {
                    Image(systemName: "cpu")
                }
            }
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button("New Chat", systemImage: "plus.message") {
                        messages.removeAll()
                        engine.reset()
                    }
                    Button(showProfiler ? "Hide Profiler" : "Profiler", systemImage: "gauge.with.dots.needle.50percent") {
                        showProfiler.toggle()
                    }
                    Button("Show Stats", systemImage: "chart.bar") {
                        showStats.toggle()
                    }
                    Divider()
                    Button {
                        runTimingProfile()
                    } label: {
                        Label(isProfileRunning ? "Profiling..." : "Run Timing Profile", systemImage: "timer")
                    }
                    .disabled(isGenerating || isProfileRunning)
                    Divider()
                    Button("Models & Settings", systemImage: "gearshape") {
                        messages.removeAll()
                        engine.reset()
                        engine.unloadModel()
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
#else
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Button(action: { showModelInfo = true }) {
                    Image(systemName: "cpu")
                }
            }
            ToolbarItem(placement: .automatic) {
                Menu {
                    Button("New Chat", systemImage: "plus.message") {
                        messages.removeAll()
                        engine.reset()
                    }
                    Button(showProfiler ? "Hide Profiler" : "Profiler", systemImage: "gauge.with.dots.needle.50percent") {
                        showProfiler.toggle()
                    }
                    Button("Show Stats", systemImage: "chart.bar") {
                        showStats.toggle()
                    }
                    Divider()
                    Button {
                        runTimingProfile()
                    } label: {
                        Label(isProfileRunning ? "Profiling..." : "Run Timing Profile", systemImage: "timer")
                    }
                    .disabled(isGenerating || isProfileRunning)
                    Divider()
                    Button("Models & Settings", systemImage: "gearshape") {
                        messages.removeAll()
                        engine.reset()
                        engine.unloadModel()
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
#endif
        .sheet(isPresented: $showModelInfo) {
            ModelInfoSheet(info: engine.modelInfo)
        }
        .sheet(isPresented: $showProfileResult) {
            ProfileResultSheet(result: profileResult ?? "")
        }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        inputText = ""
        let userMessage = ChatMessage(role: .user, text: text, timestamp: Date())
        messages.append(userMessage)

        // Start generation
        isGenerating = true
        let assistantMessage = ChatMessage(role: .assistant, text: "", timestamp: Date())
        messages.append(assistantMessage)
        let assistantIndex = messages.count - 1

        Task {
            let stream: AsyncStream<GenerationToken>

            if engine.canContinue {
                // Reuse KV cache — only process the new user turn
                stream = engine.generateContinuation(userMessage: text, maxTokens: maxGenerationTokens)
            } else {
                // First message — full chat template with system prompt
                let formattedPrompt = buildChatPrompt(userMessage: text)
                stream = engine.generate(prompt: formattedPrompt, maxTokens: maxGenerationTokens)
            }

            var gotTokens = false
            for await token in stream {
                // Skip prefill progress tokens (negative tokensGenerated)
                if token.tokensGenerated < 0 { continue }
                gotTokens = true
                // Strip special tokens that leak through
                let clean = token.text
                    .replacingOccurrences(of: "<|im_end|>", with: "")
                    .replacingOccurrences(of: "<|im_start|>", with: "")
                    .replacingOccurrences(of: "<|endoftext|>", with: "")
                if !clean.isEmpty {
                    messages[assistantIndex].text += clean
                }
            }

            // If continuation returned empty (context full), fall back to full generate
            if !gotTokens && engine.canContinue {
                engine.reset()
                let formattedPrompt = buildChatPrompt(userMessage: text)
                let fallbackStream = engine.generate(prompt: formattedPrompt, maxTokens: maxGenerationTokens)
                for await token in fallbackStream {
                    if token.tokensGenerated < 0 { continue }
                    let clean = token.text
                        .replacingOccurrences(of: "<|im_end|>", with: "")
                        .replacingOccurrences(of: "<|im_start|>", with: "")
                        .replacingOccurrences(of: "<|endoftext|>", with: "")
                    if !clean.isEmpty {
                        messages[assistantIndex].text += clean
                    }
                }
            }

            isGenerating = false
        }
    }

    private func runTimingProfile() {
        isProfileRunning = true
        isGenerating = true
        showProfiler = true  // auto-show profiler panel during run

        // Add profile prompt as a user message in chat
        let profilePrompt = "What is Apple Neural Engine?"
        let userMsg = ChatMessage(role: .user, text: "[Profile] \(profilePrompt)", timestamp: Date())
        messages.append(userMsg)

        // Add assistant message for streaming tokens
        let assistantMsg = ChatMessage(role: .assistant, text: "", timestamp: Date())
        messages.append(assistantMsg)
        let assistantIndex = messages.count - 1

        Task {
            // Enable timing, reset state for clean run
            engine.reset()
            engine.enableTiming()

            // Stream tokens — profiler panel updates live via engine.tokensPerSecond
            let prompt = buildChatPrompt(userMessage: profilePrompt)
            let stream = engine.generate(prompt: prompt, maxTokens: 40)

            for await token in stream {
                if token.tokensGenerated < 0 { continue }
                let clean = token.text
                    .replacingOccurrences(of: "<|im_end|>", with: "")
                    .replacingOccurrences(of: "<|im_start|>", with: "")
                    .replacingOccurrences(of: "<|endoftext|>", with: "")
                if !clean.isEmpty {
                    messages[assistantIndex].text += clean
                }
            }

            // Build report after generation completes
            let result = engine.buildTimingReport()
            profileResult = result
            isProfileRunning = false
            isGenerating = false
            showProfileResult = true
        }
    }

    /// Format conversation as Qwen chat template
    private func buildChatPrompt(userMessage: String) -> String {
        // Chat template can be disabled in settings (e.g. for smoke test models)
        if !chatTemplateEnabled {
            NSLog("[chat] chat template disabled — sending raw prompt")
            return userMessage
        }

        var prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"

        // Include conversation history (skip the empty assistant message we just appended)
        for msg in messages.dropLast() {
            switch msg.role {
            case .user:
                prompt += "<|im_start|>user\n\(msg.text)<|im_end|>\n"
            case .assistant:
                if !msg.text.isEmpty {
                    prompt += "<|im_start|>assistant\n\(msg.text)<|im_end|>\n"
                }
            }
        }

        prompt += "<|im_start|>assistant\n"
        return prompt
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage
    @State private var showThinking = false

    /// Split text into visible reply and thinking content
    private var parsedContent: (think: String?, reply: String) {
        let text = message.text
        // Match <think>...</think> blocks
        guard let thinkStart = text.range(of: "<think>"),
              let thinkEnd = text.range(of: "</think>") else {
            // No complete think block — check if still streaming thinking
            if text.hasPrefix("<think>") {
                let thinkBody = String(text.dropFirst("<think>".count))
                return (think: thinkBody, reply: "")
            }
            return (think: nil, reply: text)
        }
        let thinkBody = String(text[thinkStart.upperBound..<thinkEnd.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        let reply = String(text[thinkEnd.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        return (think: thinkBody, reply: reply)
    }

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                // Thinking disclosure (assistant only)
                if message.role == .assistant, let thinkText = parsedContent.think, !thinkText.isEmpty {
                    DisclosureGroup(isExpanded: $showThinking) {
                        Text(thinkText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                    } label: {
                        Label("Thinking...", systemImage: "brain")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 6)
                    #if os(iOS)
                    .background(Color(.systemGray6))
                    #else
                    .background(.thinMaterial)
                    #endif
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                }

                // Main message text
                let displayText = message.role == .assistant ? parsedContent.reply : message.text
                if !displayText.isEmpty {
                    Text(displayText)
                        .textSelection(.enabled)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        #if os(iOS)
                        .background(message.role == .user ? Color.blue : Color(.systemGray5))
                        #else
                        .background(message.role == .user ? Color.blue : Color.secondary.opacity(0.3))
                        #endif
                        .foregroundStyle(message.role == .user ? .white : .primary)
                        .clipShape(RoundedRectangle(cornerRadius: 18))
                }

                if message.text.isEmpty && message.role == .assistant {
                    ThinkingIndicator()
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        #if os(iOS)
                        .background(Color(.systemGray5))
                        #else
                        .background(Color.secondary.opacity(0.3))
                        #endif
                        .clipShape(RoundedRectangle(cornerRadius: 18))
                }
            }

            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }
}

// MARK: - Thinking Indicator

struct ThinkingIndicator: View {
    @State private var dotCount = 0
    private let timer = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()
    private let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    var body: some View {
        HStack(spacing: 6) {
            Text(frames[dotCount % frames.count])
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
            Text("Thinking")
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .onReceive(timer) { _ in
            dotCount += 1
        }
    }
}

// MARK: - Stats Bar

struct StatsBar: View {
    let tokensPerSecond: Double
    let tokensGenerated: Int
    let isGenerating: Bool

    var body: some View {
        HStack(spacing: 16) {
            Label(
                tokensGenerated < 0
                    ? String(format: "prefill %.1f tok/s", tokensPerSecond)
                    : String(format: "%.1f tok/s", tokensPerSecond),
                systemImage: "speedometer"
            )
                .font(.caption)
                .foregroundStyle(.secondary)

            Label(
                tokensGenerated < 0
                    ? "prefill \(-tokensGenerated) tok"
                    : "\(tokensGenerated) tokens",
                systemImage: "number"
            )
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            if isGenerating {
                ProgressView()
                    .scaleEffect(0.7)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Model Info Sheet

struct ModelInfoSheet: View {
    let info: ModelInfo?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            if let info {
                List {
                    Section("Model") {
                        let folderName = (info.name as NSString).lastPathComponent
                        InfoRow(label: "Name", value: folderName)
                        InfoRow(label: "Parameters", value: info.estimatedParams)
                        InfoRow(label: "Routed Experts", value: info.quantLabel)
                        InfoRow(label: "Dense/Shared", value: info.denseQuantLabel)
                        if info.isSmokeTest {
                            InfoRow(label: "Mode", value: "Smoke Test (\(info.numExperts)/512)")
                        }
                    }

                    Section("Architecture") {
                        InfoRow(label: "Layers", value: "\(info.numLayers) (\(info.numLinearLayers) linear + \(info.numFullAttnLayers) full attn)")
                        InfoRow(label: "Experts", value: "\(info.numExperts) total, K=\(info.activeExpertsK) active/layer")
                        InfoRow(label: "Hidden Dim", value: "\(info.hiddenDim)")
                        InfoRow(label: "Attn Heads", value: "\(info.numAttnHeads) Q / \(info.numKVHeads) KV (dim \(info.headDim))")
                        InfoRow(label: "MoE FFN Dim", value: "\(info.moeIntermediate)")
                        InfoRow(label: "Vocab", value: String(format: "%,d", info.vocabSize))
                    }

                    Section("Storage") {
                        InfoRow(label: "Dense Weights", value: String(format: "%.2f GB", info.weightFileMB / 1024))
                        InfoRow(label: "Expert Data", value: String(format: "%.1f GB", info.expertFileMB / 1024))
                        InfoRow(label: "Per Expert", value: String(format: "%.2f MB", info.expertSizeEachMB))
                        InfoRow(label: "Total on Disk", value: String(format: "%.1f GB", info.totalSizeGB))
                    }

                    Section("Runtime") {
                        InfoRow(label: "GPU Buffers", value: String(format: "%.0f MB", Double(info.metalBufferBytes) / 1_048_576))
                        InfoRow(label: "I/O per Token", value: String(format: "%.2f GB", info.expertSizeEachMB * Double(info.activeExpertsK) * Double(info.numLayers) / 1024))
                    }
                }
                .navigationTitle("Model Info")
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "cpu")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No model loaded")
                        .foregroundStyle(.secondary)
                }
            }
        }
#if os(iOS)
        .presentationDetents([.medium, .large])
#else
        .frame(minWidth: 400, minHeight: 450)
#endif
    }
}

struct InfoRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .fontDesign(.monospaced)
        }
    }
}

