/**
 * BridgeLLMRunner — runs prompts through the locally-installed `claude` CLI.
 *
 * This is the API-key-free counterpart to StandaloneLLMRunner. Where the
 * standalone runner opens an HTTPS connection to an OpenAI-compatible endpoint
 * and authenticates with `apiKey`, this runner spawns the same `claude` binary
 * you run by hand and reads its stdout. The binary authenticates itself with
 * whatever credentials it already holds, so TDAI never sees, stores, or
 * forwards a token.
 *
 * Consequences worth knowing before you wire this in:
 *
 *  - There is no `baseUrl` and no `apiKey`. Config that carried them
 *    (TDAI_LLM_API_KEY, MEMORY_LLM_API_KEY, LLM_API_KEY) is unused here.
 *  - Model selection goes through the CLI's own `--model`, so `modelRef` must
 *    name a Claude model or alias ("opus", "sonnet", "claude-opus-5").
 *    An OpenAI-style ref like "openai/gpt-4o" cannot be honoured.
 *  - Cost is billed to the CLI's own account, not per-token to an API key.
 *    `usage` is still reported so MetricTrackingRunner keeps working.
 *  - Caller-supplied `params.tools` CANNOT cross the process boundary — see
 *    the note in `run()`.
 */

import { spawn } from "node:child_process";
import { report } from "../../core/report/reporter.js";
import type {
  LLMRunner,
  LLMRunParams,
  LLMRunnerFactory,
  LLMRunnerCreateOptions,
  Logger,
} from "../../core/types.js";
import type { LLMUsage } from "../../core/report/metric-tracking-runner.js";

const TAG = "[memory-tdai] [bridge-runner]";

/** Cap iterations of the CLI's internal agent loop. Mirrors the standalone runner. */
const MAX_TOOL_ITERATIONS = 20;

/**
 * Hard deny-list used for pure-text runs.
 *
 * `--permission-mode dontAsk` alone is not enough: with no `--allowedTools`
 * the CLI falls back to its built-in defaults, which still permit reads. Since
 * `--disallowedTools` overrides any allowlist, naming every tool here is the
 * only reliable way to guarantee a text-only turn — matching the standalone
 * runner's `tools = undefined` branch, where the model is given no tool at all.
 *
 * Passing `--allowedTools ""` does NOT work; the CLI rejects the empty value.
 */
const ALL_TOOLS = [
  "Bash", "Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep",
  "WebFetch", "WebSearch", "Task", "TodoWrite",
];

/** Tools exposed when `enableTools` is on, aligned with the standalone sandbox. */
const FILE_TOOLS = ["Read", "Write", "Edit"];

// ============================
// Configuration
// ============================

export interface BridgeLLMConfig {
  /**
   * Path to the `claude` executable. Defaults to `claude` on PATH.
   * Set explicitly when TDAI runs as a daemon with a minimal PATH.
   */
  claudeBin?: string;
  /** Default model alias or id (e.g. "opus", "claude-opus-5"). CLI default when omitted. */
  model?: string;
  /** Request timeout in milliseconds (default: 120_000). */
  timeoutMs?: number;
  /**
   * Working directory for the spawned process. Tool-enabled runs resolve file
   * paths against `params.workspaceDir`, falling back to this.
   */
  cwd?: string;
  /**
   * Extra CLI flags appended verbatim. Escape hatch for flags this adapter
   * does not model (e.g. "--settings", "--add-dir").
   */
  extraArgs?: string[];
}

// ============================
// CLI result envelope
// ============================

/** The subset of `claude -p --output-format json` this runner depends on. */
interface ClaudeCliResult {
  type?: string;
  subtype?: string;
  is_error?: boolean;
  result?: string;
  session_id?: string;
  total_cost_usd?: number;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
  };
}

// ============================
// BridgeLLMRunner
// ============================

export class BridgeLLMRunner implements LLMRunner {
  private config: BridgeLLMConfig;
  private model?: string;
  private enableTools: boolean;
  private logger?: Logger;

  /** Side-channel read by MetricTrackingRunner; same contract as the standalone runner. */
  lastUsage?: LLMUsage;

  /** Cost in USD reported by the CLI for the last run. No standalone equivalent. */
  lastCostUsd?: number;

  constructor(opts: {
    config: BridgeLLMConfig;
    model?: string;
    enableTools?: boolean;
    logger?: Logger;
  }) {
    this.config = opts.config;
    this.model = opts.model ?? opts.config.model;
    this.enableTools = opts.enableTools ?? false;
    this.logger = opts.logger;
  }

  async run(params: LLMRunParams): Promise<string> {
    const runStartMs = Date.now();
    const timeoutMs = params.timeoutMs ?? this.config.timeoutMs ?? 120_000;
    const cwd = params.workspaceDir ?? this.config.cwd ?? process.cwd();
    const effectiveEnableTools = params.enableTools ?? this.enableTools;
    const maxIterations = params.maxIterations ?? MAX_TOOL_ITERATIONS;

    // The standalone runner accepts a caller-provided tool dict (Vercel AI SDK
    // shape) and drives the tool loop in-process — SkillExtractor relies on
    // this to inject skill_list / skill_view / skill_manage. Those are live JS
    // closures over TDAI's own state; they cannot be serialized across a
    // subprocess boundary into the CLI's tool registry.
    //
    // Failing loudly is deliberate. Dropping the tools silently would let the
    // model answer without ever calling them, and the caller would parse a
    // confidently-wrong response as a successful extraction.
    if (params.tools && Object.keys(params.tools).length > 0 && effectiveEnableTools) {
      throw new Error(
        `${TAG} caller-provided tools are not supported by the bridge runner ` +
        `(taskId=${params.taskId}, tools=[${Object.keys(params.tools).join(", ")}]). ` +
        `These must be re-exposed as an MCP server the CLI can connect to, or the ` +
        `call routed to the standalone runner.`,
      );
    }

    // `params.storage` selects storage-backed tools in the standalone runner
    // (COS service mode). The CLI only reaches the local filesystem, so a
    // storage-backed run would silently read the wrong files.
    if (effectiveEnableTools && params.storage) {
      throw new Error(
        `${TAG} storage-backed tools are not supported by the bridge runner ` +
        `(taskId=${params.taskId}). The spawned CLI sees the local filesystem, ` +
        `not the StorageAdapter.`,
      );
    }

    const args = this.buildArgs(params, { effectiveEnableTools, maxIterations, cwd });

    this.logger?.debug?.(
      `${TAG} run() start: taskId=${params.taskId}, model=${this.model ?? "cli-default"}, ` +
      `tools=${effectiveEnableTools}, timeout=${timeoutMs}ms, cwd=${cwd}`,
    );

    try {
      const raw = await this.spawnClaude(args, { cwd, timeoutMs, abortSignal: params.abortSignal });
      const parsed = this.parseResult(raw);
      const text = (parsed.result ?? "").trim();
      const totalMs = Date.now() - runStartMs;

      if (parsed.is_error) {
        throw new Error(`${TAG} CLI reported failure (subtype=${parsed.subtype}): ${text || "no detail"}`);
      }

      if (parsed.usage) {
        const promptTokens =
          (parsed.usage.input_tokens ?? 0) +
          // Cache reads/writes are real input the model processed. Counting
          // them keeps credit reporting comparable to the standalone runner,
          // which sees a single undifferentiated promptTokens figure.
          (parsed.usage.cache_read_input_tokens ?? 0) +
          (parsed.usage.cache_creation_input_tokens ?? 0);
        const completionTokens = parsed.usage.output_tokens ?? 0;
        this.lastUsage = {
          promptTokens,
          completionTokens,
          totalTokens: promptTokens + completionTokens,
        };
      } else {
        this.lastUsage = undefined;
      }
      this.lastCostUsd = parsed.total_cost_usd;

      this.logger?.debug?.(
        `${TAG} run() completed: ${totalMs}ms, output=${text.length} chars, ` +
        `cost=$${parsed.total_cost_usd ?? 0}`,
      );

      if (params.instanceId) {
        report("llm_call", {
          taskId: params.taskId,
          provider: "bridge",
          model: this.model ?? "cli-default",
          inputLength: params.prompt.length,
          outputLength: text.length,
          totalDurationMs: totalMs,
          success: true,
          error: null,
        });
      }

      return text;
    } catch (err) {
      const totalMs = Date.now() - runStartMs;
      const errMsg = err instanceof Error ? err.message : String(err);
      this.logger?.error(`${TAG} run() failed after ${totalMs}ms: ${errMsg}`);

      if (params.instanceId) {
        report("llm_call", {
          taskId: params.taskId,
          provider: "bridge",
          model: this.model ?? "cli-default",
          inputLength: params.prompt.length,
          outputLength: 0,
          totalDurationMs: totalMs,
          success: false,
          error: errMsg,
        });
      }

      throw err;
    }
  }

  // ----------------------------------------------------------------

  private buildArgs(
    params: LLMRunParams,
    opts: { effectiveEnableTools: boolean; maxIterations: number; cwd: string },
  ): string[] {
    const args = ["-p", params.prompt, "--output-format", "json"];

    if (opts.effectiveEnableTools) {
      // acceptEdits, not bypassPermissions: headless runs have no approval
      // callback, and bypass would also unlock Bash — far beyond the
      // read/write/edit sandbox the standalone runner grants.
      args.push("--permission-mode", "acceptEdits");
      args.push("--allowedTools", FILE_TOOLS.join(","));
      args.push("--max-turns", String(opts.maxIterations));
    } else {
      args.push("--permission-mode", "dontAsk");
      args.push("--disallowedTools", ALL_TOOLS.join(","));
      // A pure-text task is one turn by definition.
      args.push("--max-turns", "1");
    }

    // LLMRunParams.systemPrompt maps to --append-system-prompt rather than
    // --system-prompt: the CLI's own system prompt is what makes it a working
    // agent, and replacing it wholesale breaks tool use.
    if (params.systemPrompt) {
      args.push("--append-system-prompt", params.systemPrompt);
    }
    if (this.model) {
      args.push("--model", this.model);
    }
    if (this.config.extraArgs?.length) {
      args.push(...this.config.extraArgs);
    }

    // Deliberately unmapped: `params.maxTokens` has no CLI equivalent, so
    // output length is governed by the model's own limit. Callers that depend
    // on a hard cap must post-truncate.
    return args;
  }

  private spawnClaude(
    args: string[],
    opts: { cwd: string; timeoutMs: number; abortSignal?: AbortSignal },
  ): Promise<string> {
    const bin = this.config.claudeBin ?? "claude";

    return new Promise<string>((resolve, reject) => {
      const child = spawn(bin, args, {
        cwd: opts.cwd,
        stdio: ["ignore", "pipe", "pipe"],
        env: process.env,
      });

      let stdout = "";
      let stderr = "";
      let settled = false;

      const finish = (fn: () => void) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        opts.abortSignal?.removeEventListener("abort", onAbort);
        fn();
      };

      // SIGTERM first so the CLI can flush; the process is killed regardless
      // once the promise settles, so no SIGKILL escalation is needed here.
      const timer = setTimeout(() => {
        child.kill("SIGTERM");
        finish(() => reject(new Error(`${TAG} timed out after ${opts.timeoutMs}ms`)));
      }, opts.timeoutMs);

      const onAbort = () => {
        child.kill("SIGTERM");
        finish(() => reject(new Error(`${TAG} aborted by caller`)));
      };
      opts.abortSignal?.addEventListener("abort", onAbort, { once: true });

      child.stdout.on("data", (d) => { stdout += d.toString(); });
      child.stderr.on("data", (d) => { stderr += d.toString(); });

      child.on("error", (err) => {
        finish(() => reject(new Error(
          `${TAG} failed to spawn "${bin}": ${err.message}. ` +
          `Is Claude Code installed and on PATH?`,
        )));
      });

      child.on("close", (code) => {
        finish(() => {
          if (code === 0) return resolve(stdout);
          reject(new Error(
            `${TAG} exited ${code}: ${stderr.trim().slice(0, 500) || "no stderr"}`,
          ));
        });
      });
    });
  }

  private parseResult(raw: string): ClaudeCliResult {
    const trimmed = raw.trim();
    if (!trimmed) {
      throw new Error(`${TAG} CLI produced no output`);
    }
    try {
      return JSON.parse(trimmed) as ClaudeCliResult;
    } catch {
      throw new Error(
        `${TAG} could not parse CLI output as JSON: ${trimmed.slice(0, 300)}`,
      );
    }
  }
}

// ============================
// BridgeLLMRunnerFactory
// ============================

export interface BridgeLLMRunnerFactoryOptions {
  /** Bridge configuration (binary path, default model, timeout). */
  config?: BridgeLLMConfig;
  /** Logger instance. */
  logger?: Logger;
}

/**
 * Factory that creates BridgeLLMRunner instances.
 *
 * Drop-in for StandaloneLLMRunnerFactory wherever a HostAdapter is built —
 * the only difference is that `config` carries no credentials.
 */
export class BridgeLLMRunnerFactory implements LLMRunnerFactory {
  private config: BridgeLLMConfig;
  private logger?: Logger;

  constructor(opts: BridgeLLMRunnerFactoryOptions = {}) {
    this.config = opts.config ?? {};
    this.logger = opts.logger;
  }

  createRunner(opts?: LLMRunnerCreateOptions): LLMRunner {
    const enableTools = opts?.enableTools ?? false;
    const modelRef = opts?.modelRef;

    let model = this.config.model;
    if (modelRef) {
      // Callers pass "provider/model". The CLI has no notion of a provider
      // prefix, so strip it — but warn, because a non-Anthropic prefix means
      // the caller expected a model this runner cannot reach.
      const slashIdx = modelRef.indexOf("/");
      if (slashIdx > 0) {
        const provider = modelRef.slice(0, slashIdx);
        model = modelRef.slice(slashIdx + 1);
        if (provider !== "anthropic" && provider !== "claude") {
          this.logger?.warn(
            `${TAG} modelRef "${modelRef}" names provider "${provider}", ` +
            `but the bridge only reaches models the local CLI serves. ` +
            `Passing "${model}" through; expect the CLI to reject it if unknown.`,
          );
        }
      } else {
        model = modelRef;
      }
    }

    this.logger?.debug?.(
      `${TAG} Creating BridgeLLMRunner: model=${model ?? "cli-default"}, tools=${enableTools}`,
    );

    return new BridgeLLMRunner({
      config: this.config,
      model,
      enableTools,
      logger: this.logger,
    });
  }
}
