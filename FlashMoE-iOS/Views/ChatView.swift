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
    @State private var showModelInfo = false
    @State private var showProfiler = false
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
                    isGenerating: isGenerating,
                    ttftMs: engine.timeToFirstToken
                )
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
        .overlay(alignment: .bottom) {
            if showProfiler {
                ProfilerView(engine: engine)
                    .padding(.bottom, 80)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
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
                stream = engine.generateContinuation(userMessage: text, maxTokens: 500)
            } else {
                // First message — full chat template with system prompt
                let formattedPrompt = buildChatPrompt(userMessage: text)
                stream = engine.generate(prompt: formattedPrompt, maxTokens: 500)
            }

            var gotTokens = false
            for await token in stream {
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
                let fallbackStream = engine.generate(prompt: formattedPrompt, maxTokens: 500)
                for await token in fallbackStream {
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

    /// Format conversation as Qwen chat template
    private func buildChatPrompt(userMessage: String) -> String {
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
                        .background(message.role == .user ? Color.blue : Color.secondary)
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
                        .background(.quaternary)
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
    var ttftMs: Double = 0

    private var ttftText: String {
        if ttftMs <= 0 { return "" }
        if ttftMs > 500_000 {
            return String(format: "%.1f min", ttftMs / 60_000)
        } else if ttftMs > 1000 {
            return String(format: "%.1fs", ttftMs / 1000)
        } else {
            return String(format: "%.0fms", ttftMs)
        }
    }

    private var thermalLabel: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:  return "\u{1F7E2} Cool"
        case .fair:     return "\u{1F7E1} Warm"
        case .serious:  return "\u{1F7E0} Hot"
        case .critical: return "\u{1F534} Critical"
        @unknown default: return "\u{2753} Unknown"
        }
    }

    var body: some View {
        HStack(spacing: 12) {
            Label(String(format: "%.1f tok/s", tokensPerSecond), systemImage: "speedometer")
                .font(.caption)
                .foregroundStyle(.secondary)

            Label("\(tokensGenerated) tokens", systemImage: "number")
                .font(.caption)
                .foregroundStyle(.secondary)

            if !ttftText.isEmpty {
                Label(ttftText, systemImage: "clock")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text(thermalLabel)
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
                    Section("Architecture") {
                        InfoRow(label: "Layers", value: "\(info.numLayers)")
                        InfoRow(label: "Experts", value: "\(info.numExperts) (K=\(info.activeExpertsK))")
                        InfoRow(label: "Hidden Dim", value: "\(info.hiddenDim)")
                        InfoRow(label: "Vocab Size", value: "\(info.vocabSize)")
                    }
                    Section("Storage") {
                        InfoRow(label: "Weights", value: String(format: "%.1f MB", info.weightFileMB))
                        InfoRow(label: "Experts", value: String(format: "%.1f MB", info.expertFileMB))
                        InfoRow(label: "Total", value: String(format: "%.1f GB", info.totalSizeMB / 1024))
                    }
                }
                .navigationTitle("Model Info")
            } else {
                Text("No model loaded")
            }
        }
#if os(iOS)
        .presentationDetents([.medium])
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

