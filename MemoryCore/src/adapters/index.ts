/**
 * TDAI Adapters — barrel re-export for all host adapter implementations.
 *
 * Each adapter translates a specific host environment's API into
 * the host-neutral HostAdapter interface consumed by TdaiCore.
 *
 * Directory structure:
 *   adapters/
 *   ├── openclaw/      — OpenClaw plugin host (in-process, runEmbeddedPiAgent)
 *   ├── standalone/    — Gateway / Hermes sidecar (HTTP, OpenAI-compatible API)
 *   └── bridge/        — Gateway / Hermes sidecar (local `claude` CLI subprocess,
 *                        no API key; see bridge/README.md)
 */

// OpenClaw adapter
export { OpenClawHostAdapter, OpenClawLLMRunner, OpenClawLLMRunnerFactory } from "./openclaw/index.js";
export type { OpenClawHostAdapterOptions, OpenClawLLMRunnerFactoryOptions } from "./openclaw/index.js";

// Standalone adapter
export { StandaloneHostAdapter, StandaloneLLMRunner, StandaloneLLMRunnerFactory } from "./standalone/index.js";
export type { StandaloneHostAdapterOptions, StandaloneLLMConfig, StandaloneLLMRunnerFactoryOptions } from "./standalone/index.js";

// Bridge adapter
export { BridgeHostAdapter, BridgeLLMRunner, BridgeLLMRunnerFactory } from "./bridge/index.js";
export type { BridgeHostAdapterOptions, BridgeLLMConfig, BridgeLLMRunnerFactoryOptions } from "./bridge/index.js";
