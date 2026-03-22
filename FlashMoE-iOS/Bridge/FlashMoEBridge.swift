/*
 * FlashMoEBridge.swift — Async Swift wrapper for the FlashMoE C engine
 *
 * Provides a Swift-native interface with:
 * - AsyncStream for token-by-token generation
 * - Observable properties for SwiftUI integration
 * - Automatic background thread management
 */

import Foundation
import Observation

// MARK: - Data Types

/// Generation result with streaming tokens
struct GenerationToken {
    let text: String
    let tokenId: Int
    let tokensGenerated: Int
    let tokensPerSecond: Double
}

/// Model information after loading
struct ModelInfo {
    let name: String
    let numLayers: Int
    let numExperts: Int
    let activeExpertsK: Int
    let hiddenDim: Int
    let vocabSize: Int
    let weightFileBytes: UInt64
    let expertFileBytes: UInt64
    let metalBufferBytes: UInt64

    var weightFileMB: Double { Double(weightFileBytes) / 1_048_576 }
    var expertFileMB: Double { Double(expertFileBytes) / 1_048_576 }
    var totalSizeMB: Double { weightFileMB + expertFileMB }
}

/// Engine state for UI binding
enum EngineState: Equatable {
    case idle
    case loading
    case ready
    case generating
    case error(String)
}

// MARK: - FlashMoEEngine (Observable)

@Observable
final class FlashMoEEngine: @unchecked Sendable {
    // Observable state for SwiftUI
    private(set) var state: EngineState = .idle
    private(set) var modelInfo: ModelInfo?
    private(set) var tokensPerSecond: Double = 0
    private(set) var tokensGenerated: Int = 0
    private(set) var timeToFirstToken: Double = 0

    // Private engine state
    private var context: OpaquePointer?  // FlashMoEContext*
    private let engineQueue = DispatchQueue(label: "com.flashmoe.engine", qos: .userInitiated)
    private var isGenerating = false

    init() {}

    deinit {
        if let ctx = context {
            flashmoe_destroy(ctx)
        }
    }

    // MARK: - Model Loading

    /// Load a model from the given path. Runs on a background thread.
    /// Load a model. Set `activeExpertsK` to reduce expert count for large models on small devices.
    /// For example, K=4 on a K=10 model cuts I/O by 60%.
    func loadModel(at path: String, maxContext: Int = 0, thinkBudget: Int = 2048,
                   useTiered: Bool = false, activeExpertsK: Int = 0, cacheIOSplit: Int = 1,
                   verbose: Bool = false) async throws {
        guard state != .loading && state != .generating else {
            throw FlashMoEError.busy
        }

        await MainActor.run { state = .loading }

        return try await withCheckedThrowingContinuation { continuation in
            engineQueue.async { [weak self] in
                guard let self else {
                    continuation.resume(throwing: FlashMoEError.engineDestroyed)
                    return
                }

                // Create context if needed
                if self.context == nil {
                    self.context = flashmoe_create()
                }
                guard let ctx = self.context else {
                    DispatchQueue.main.async { self.state = .error("Failed to create engine context") }
                    continuation.resume(throwing: FlashMoEError.initFailed)
                    return
                }

                // Configure
                var config = FlashMoEConfig()
                let pathCStr = (path as NSString).utf8String
                config.model_path = pathCStr
                config.max_context = Int32(maxContext)
                config.think_budget = Int32(thinkBudget)
                config.use_tiered = useTiered ? 1 : 0
                config.active_experts_k = Int32(activeExpertsK)
                config.cache_io_split = Int32(cacheIOSplit)
                config.verbose = verbose ? 1 : 0

                // Load
                let result = flashmoe_load(ctx, &config)
                if result != 0 {
                    let error = String(cString: flashmoe_last_error(ctx))
                    DispatchQueue.main.async { self.state = .error(error) }
                    continuation.resume(throwing: FlashMoEError.loadFailed(error))
                    return
                }

                // Get stats for model info
                var stats = FlashMoEStats()
                flashmoe_get_stats(ctx, &stats)

                let modelName = withUnsafePointer(to: &stats.model_name) {
                    $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) }
                }
                let info = ModelInfo(
                    name: modelName,
                    numLayers: Int(stats.num_layers),
                    numExperts: Int(stats.num_experts),
                    activeExpertsK: Int(stats.active_experts_k),
                    hiddenDim: Int(stats.hidden_dim),
                    vocabSize: Int(stats.vocab_size),
                    weightFileBytes: UInt64(stats.weight_file_bytes),
                    expertFileBytes: UInt64(stats.expert_file_bytes),
                    metalBufferBytes: UInt64(stats.metal_buffer_bytes)
                )

                DispatchQueue.main.async {
                    self.modelInfo = info
                    self.state = .ready
                }
                continuation.resume()
            }
        }
    }

    /// Unload the current model
    func unloadModel() {
        guard let ctx = context else { return }
        engineQueue.sync {
            flashmoe_unload(ctx)
        }
        modelInfo = nil
        state = .idle
    }

    // MARK: - Generation

    /// Generate tokens from a prompt, returning an AsyncStream of tokens
    func generate(prompt: String, maxTokens: Int = 200) -> AsyncStream<GenerationToken> {
        AsyncStream { continuation in
            guard let ctx = context, state == .ready else {
                continuation.finish()
                return
            }

            DispatchQueue.main.async {
                self.state = .generating
                self.tokensGenerated = 0
                self.tokensPerSecond = 0
                self.isGenerating = true
            }

            // Set up cancellation
            nonisolated(unsafe) let ctxForCancel = ctx
            continuation.onTermination = { @Sendable _ in
                flashmoe_cancel(ctxForCancel)
            }

            engineQueue.async { [weak self] in
                // C callback bridge: userdata points to the Swift continuation
                let userDataPtr = Unmanaged.passRetained(
                    TokenCallbackContext(continuation: continuation, engine: self)
                ).toOpaque()

                let result = flashmoe_generate(
                    ctx,
                    prompt,
                    Int32(maxTokens),
                    { tokenText, tokenId, tokensGenerated, tokensPerSecond, userData -> Int32 in
                        guard let userData else { return 1 }
                        let context = Unmanaged<TokenCallbackContext>.fromOpaque(userData)
                            .takeUnretainedValue()

                        guard let text = tokenText else { return 0 }
                        let token = GenerationToken(
                            text: String(cString: text),
                            tokenId: Int(tokenId),
                            tokensGenerated: Int(tokensGenerated),
                            tokensPerSecond: tokensPerSecond
                        )

                        // Update engine stats on main thread
                        if let engine = context.engine {
                            DispatchQueue.main.async {
                                engine.tokensGenerated = Int(tokensGenerated)
                                engine.tokensPerSecond = tokensPerSecond
                            }
                        }

                        context.continuation.yield(token)
                        return 0
                    },
                    userDataPtr
                )

                // Clean up
                Unmanaged<TokenCallbackContext>.fromOpaque(userDataPtr).release()

                // Get final stats
                var stats = FlashMoEStats()
                flashmoe_get_stats(ctx, &stats)

                DispatchQueue.main.async {
                    self?.timeToFirstToken = stats.ttft_ms
                    self?.tokensPerSecond = stats.tokens_per_second
                    self?.tokensGenerated = Int(stats.tokens_generated)
                    self?.state = .ready
                    self?.isGenerating = false
                }

                continuation.finish()
            }
        }
    }

    /// Generate continuation — reuses KV cache from previous turns.
    /// Returns nil if context is full (caller should reset and use generate instead).
    func generateContinuation(userMessage: String, maxTokens: Int = 200) -> AsyncStream<GenerationToken> {
        AsyncStream { continuation in
            guard let ctx = context, state == .ready else {
                continuation.finish()
                return
            }

            DispatchQueue.main.async {
                self.state = .generating
                self.tokensGenerated = 0
                self.tokensPerSecond = 0
                self.isGenerating = true
            }

            nonisolated(unsafe) let ctxForCancel = ctx
            continuation.onTermination = { @Sendable _ in
                flashmoe_cancel(ctxForCancel)
            }

            engineQueue.async { [weak self] in
                let userDataPtr = Unmanaged.passRetained(
                    TokenCallbackContext(continuation: continuation, engine: self)
                ).toOpaque()

                let result = flashmoe_generate_continuation(
                    ctx,
                    userMessage,
                    Int32(maxTokens),
                    { tokenText, tokenId, tokensGenerated, tokensPerSecond, userData -> Int32 in
                        guard let userData else { return 1 }
                        let context = Unmanaged<TokenCallbackContext>.fromOpaque(userData)
                            .takeUnretainedValue()

                        guard let text = tokenText else { return 0 }
                        let token = GenerationToken(
                            text: String(cString: text),
                            tokenId: Int(tokenId),
                            tokensGenerated: Int(tokensGenerated),
                            tokensPerSecond: tokensPerSecond
                        )

                        if let engine = context.engine {
                            DispatchQueue.main.async {
                                engine.tokensGenerated = Int(tokensGenerated)
                                engine.tokensPerSecond = tokensPerSecond
                            }
                        }

                        context.continuation.yield(token)
                        return 0
                    },
                    userDataPtr
                )

                Unmanaged<TokenCallbackContext>.fromOpaque(userDataPtr).release()

                // -2 = context full, signal via empty stream (caller handles reset)
                if result == -2 {
                    DispatchQueue.main.async {
                        self?.state = .ready
                        self?.isGenerating = false
                    }
                    continuation.finish()
                    return
                }

                var stats = FlashMoEStats()
                flashmoe_get_stats(ctx, &stats)

                DispatchQueue.main.async {
                    self?.timeToFirstToken = stats.ttft_ms
                    self?.tokensPerSecond = stats.tokens_per_second
                    self?.tokensGenerated = Int(stats.tokens_generated)
                    self?.state = .ready
                    self?.isGenerating = false
                }

                continuation.finish()
            }
        }
    }

    /// Whether the engine has conversation state that can be continued
    var canContinue: Bool {
        guard let ctx = context else { return false }
        return flashmoe_turn_count(ctx) > 0
    }

    /// Cancel an in-progress generation
    func cancel() {
        guard let ctx = context, isGenerating else { return }
        flashmoe_cancel(ctx)
    }

    /// Reset conversation state (KV cache, attention state)
    func reset() {
        guard let ctx = context else { return }
        engineQueue.async {
            flashmoe_reset(ctx)
        }
    }

    // MARK: - Model Validation

    /// Check if a model directory contains a valid Flash-MoE model
    static func validateModel(at path: String) -> Bool {
        return flashmoe_validate_model(path) == 0
    }
}

// MARK: - Helper Types

/// Bridging class to pass Swift state through C void* callback
private final class TokenCallbackContext {
    let continuation: AsyncStream<GenerationToken>.Continuation
    weak var engine: FlashMoEEngine?

    init(continuation: AsyncStream<GenerationToken>.Continuation, engine: FlashMoEEngine?) {
        self.continuation = continuation
        self.engine = engine
    }
}

/// Errors from the Flash-MoE engine
enum FlashMoEError: LocalizedError {
    case busy
    case engineDestroyed
    case initFailed
    case loadFailed(String)
    case generateFailed(String)
    case notLoaded

    var errorDescription: String? {
        switch self {
        case .busy: return "Engine is busy"
        case .engineDestroyed: return "Engine was destroyed"
        case .initFailed: return "Failed to initialize engine"
        case .loadFailed(let msg): return "Failed to load model: \(msg)"
        case .generateFailed(let msg): return "Generation failed: \(msg)"
        case .notLoaded: return "No model loaded"
        }
    }
}
