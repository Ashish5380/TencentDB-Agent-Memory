/**
 * BridgeHostAdapter — HostAdapter that drives the local `claude` CLI.
 *
 * Identical to StandaloneHostAdapter in every respect except how it answers
 * "how do I call an LLM?": no `llmConfig.baseUrl`, no `llmConfig.apiKey`, and
 * therefore no credential to provision, rotate, or leak.
 *
 * `hostType` stays "standalone" because the HostAdapter contract enumerates
 * exactly "openclaw" | "hermes" | "standalone", and downstream code branches on
 * it. The bridge is a standalone sidecar by every behavioural measure — only
 * its transport to the model differs — so widening that union would force
 * changes in call sites that have no business knowing about the bridge.
 */

import { BridgeLLMRunnerFactory } from "./llm-runner.js";
import type { BridgeLLMConfig } from "./llm-runner.js";
import type {
  HostAdapter,
  RuntimeContext,
  Logger,
  LLMRunnerFactory,
} from "../../core/types.js";

// ============================
// Options
// ============================

export interface BridgeHostAdapterOptions {
  /** Base data directory for TDAI storage. */
  dataDir: string;
  /** Bridge configuration. Optional — defaults to `claude` on PATH. */
  llmConfig?: BridgeLLMConfig;
  /** Logger instance. */
  logger: Logger;
  /** Default user ID (can be overridden per-request). */
  defaultUserId?: string;
  /** Platform identifier. */
  platform?: string;
}

// ============================
// BridgeHostAdapter
// ============================

export class BridgeHostAdapter implements HostAdapter {
  readonly hostType = "standalone" as const;

  private dataDir: string;
  private logger: Logger;
  private runnerFactory: BridgeLLMRunnerFactory;
  private defaultUserId: string;
  private platform: string;

  constructor(opts: BridgeHostAdapterOptions) {
    this.dataDir = opts.dataDir;
    this.logger = opts.logger;
    this.defaultUserId = opts.defaultUserId ?? "default_user";
    this.platform = opts.platform ?? "bridge";

    this.runnerFactory = new BridgeLLMRunnerFactory({
      // Tool-enabled runs default to the data dir, matching the standalone
      // adapter's workspaceDir so file paths resolve the same way.
      config: { cwd: opts.dataDir, ...opts.llmConfig },
      logger: opts.logger,
    });
  }

  getRuntimeContext(): RuntimeContext {
    return {
      userId: this.defaultUserId,
      sessionId: "",
      sessionKey: "",
      platform: this.platform,
      workspaceDir: this.dataDir,
      dataDir: this.dataDir,
    };
  }

  /**
   * Build a RuntimeContext for a specific request.
   * Mirrors StandaloneHostAdapter so route handlers are interchangeable.
   */
  buildRuntimeContextForRequest(params: {
    userId?: string;
    sessionId?: string;
    sessionKey?: string;
    platform?: string;
  }): RuntimeContext {
    return {
      userId: params.userId ?? this.defaultUserId,
      sessionId: params.sessionId ?? "",
      sessionKey: params.sessionKey ?? params.sessionId ?? "",
      platform: params.platform ?? this.platform,
      workspaceDir: this.dataDir,
      dataDir: this.dataDir,
    };
  }

  getLogger(): Logger {
    return this.logger;
  }

  getLLMRunnerFactory(): LLMRunnerFactory {
    return this.runnerFactory;
  }
}
