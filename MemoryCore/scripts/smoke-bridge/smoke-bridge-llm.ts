/**
 * Smoke test for the bridge LLM runner.
 *
 *   npx tsx scripts/smoke-bridge/smoke-bridge-llm.ts
 *
 * Requires the `claude` CLI on PATH and already logged in. Makes two real
 * calls, so it costs a little; it is deliberately not part of `npm test`.
 *
 * Proves the three things the conversion has to get right:
 *   1. a pure-text run returns text and reports usage, with no API key anywhere
 *   2. caller-provided tools fail loudly rather than being silently dropped
 *   3. an unreachable binary produces a diagnosable error, not a hang
 */

import { BridgeLLMRunnerFactory } from "../../src/adapters/bridge/index.js";
import type { Logger } from "../../src/core/types.js";

const logger: Logger = {
  debug: (m) => console.log(`  · ${m}`),
  info: (m) => console.log(`  i ${m}`),
  warn: (m) => console.warn(`  ! ${m}`),
  error: (m) => console.error(`  x ${m}`),
};

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures++;
}

async function main() {
  const factory = new BridgeLLMRunnerFactory({ logger });

  // --- 1. pure text ---------------------------------------------------------
  console.log("\n1. pure-text run (no tools, no API key)");
  const runner = factory.createRunner({ enableTools: false });
  const text = await runner.run({
    prompt: "Reply with exactly the word: pong",
    taskId: "smoke-text",
    timeoutMs: 120_000,
  });
  check("returns text", text.toLowerCase().includes("pong"), JSON.stringify(text.slice(0, 60)));

  const usage = (runner as { lastUsage?: { totalTokens: number } }).lastUsage;
  check("reports usage for credit tracking", (usage?.totalTokens ?? 0) > 0, `totalTokens=${usage?.totalTokens}`);

  // --- 2. system prompt is honoured ----------------------------------------
  console.log("\n2. systemPrompt reaches the model");
  const sysText = await runner.run({
    prompt: "What is the codeword?",
    systemPrompt: "The codeword is BRIDGE42. When asked, reply with only the codeword.",
    taskId: "smoke-system",
    timeoutMs: 120_000,
  });
  check("append-system-prompt applied", sysText.includes("BRIDGE42"), JSON.stringify(sysText.slice(0, 60)));

  // --- 3. caller tools rejected --------------------------------------------
  console.log("\n3. caller-provided tools fail loudly");
  let threw = "";
  try {
    await runner.run({
      prompt: "list the skills",
      taskId: "smoke-tools",
      enableTools: true,
      tools: { skill_list: {} },
    });
  } catch (err) {
    threw = err instanceof Error ? err.message : String(err);
  }
  check("throws instead of dropping tools", threw.includes("caller-provided tools are not supported"), threw.slice(0, 90));

  // --- 4. missing binary is diagnosable ------------------------------------
  console.log("\n4. missing CLI binary");
  const broken = new BridgeLLMRunnerFactory({
    config: { claudeBin: "/nonexistent/claude" },
    logger,
  }).createRunner();
  let spawnErr = "";
  try {
    await broken.run({ prompt: "hi", taskId: "smoke-missing", timeoutMs: 10_000 });
  } catch (err) {
    spawnErr = err instanceof Error ? err.message : String(err);
  }
  check("names the binary and the fix", spawnErr.includes("failed to spawn") && spawnErr.includes("PATH"), spawnErr.slice(0, 90));

  console.log(`\n${failures === 0 ? "PASS" : `FAIL (${failures})`}`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
