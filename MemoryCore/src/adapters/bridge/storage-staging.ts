/**
 * Storage staging — lets the spawned `claude` CLI operate on StorageAdapter
 * content using its own filesystem tools.
 *
 * The standalone runner hands the model in-process `read`/`write`/`edit` tools
 * that call StorageAdapter directly. A subprocess cannot be given those
 * closures, and the CLI's Read/Write/Edit only reach the real filesystem. So
 * instead of bridging the tools, we bridge the *data*:
 *
 *   1. stage    — copy every object under `prefix` into a fresh temp dir
 *   2. (caller runs the CLI with cwd = that dir)
 *   3. syncBack — diff the dir against what we staged, and apply creates,
 *                 modifications, and deletions back through StorageAdapter
 *
 * This is backend-agnostic: it behaves identically whether StorageAdapter is
 * backed by local disk or COS. The cost is one read and one write per changed
 * object, which is proportional to what the model touched — scene blocks are
 * small markdown files, so this stays cheap.
 *
 * Known limits, deliberate:
 *   - Not atomic. A crash between the first and last writeFile leaves storage
 *     partially updated. The scene extractor already restores from its own
 *     backup when the runner throws, which covers the case that matters.
 *   - Text only. Objects are staged via readFile (string); binary content
 *     would be corrupted by the round-trip. Scene blocks and persona files are
 *     text, and StorageAdapter's own tools are equally text-only.
 *   - Files the model creates in subdirectories are synced back under the same
 *     relative key, which matches resolveStorageKey's flat-prefix semantics.
 */

import fsPromises from "node:fs/promises";
import path from "node:path";
import { tmpdir } from "node:os";
import type { StorageAdapter } from "../../core/storage/adapter.js";
import type { Logger } from "../../core/types.js";

const TAG = "[memory-tdai] [bridge-staging]";

export interface StagedWorkspace {
  /** Absolute path to the temp dir the CLI should run in. */
  dir: string;
  /** Relative path → content, as staged. Used to detect what changed. */
  staged: Map<string, string>;
}

/**
 * Copy everything under `prefix` into a fresh temp directory.
 *
 * An empty prefix is normal (first run) and yields an empty dir rather than an
 * error — the model is expected to create the first scene block itself.
 */
export async function stageFromStorage(
  storage: StorageAdapter,
  prefix: string,
  logger?: Logger,
): Promise<StagedWorkspace> {
  const dir = await fsPromises.mkdtemp(path.join(tmpdir(), "tdai-bridge-"));
  const staged = new Map<string, string>();

  const names = await storage.readdirNames(prefix);
  for (const name of names) {
    const content = await storage.readFile(`${prefix}${name}`);
    // readFile returns null for a key that vanished between list and read.
    // Skipping is correct: it is not part of the workspace we are staging.
    if (content === null) continue;
    const dest = path.join(dir, name);
    await fsPromises.mkdir(path.dirname(dest), { recursive: true });
    await fsPromises.writeFile(dest, content, "utf8");
    staged.set(name, content);
  }

  logger?.debug?.(`${TAG} staged ${staged.size} object(s) from "${prefix}" into ${dir}`);
  return { dir, staged };
}

/** Recursively list files in `dir`, as paths relative to it. */
async function listRelative(dir: string, base = dir): Promise<string[]> {
  const out: string[] = [];
  const entries = await fsPromises.readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      out.push(...await listRelative(full, base));
    } else if (e.isFile()) {
      out.push(path.relative(base, full));
    }
  }
  return out;
}

export interface SyncResult {
  written: number;
  deleted: number;
  unchanged: number;
}

/**
 * Apply the temp dir's current state back to storage, then remove the dir.
 *
 * Unchanged files are skipped so a run where the model only read costs no
 * writes. Files present at stage time but gone afterwards are deleted from
 * storage — the scene extractor merges and prunes blocks, so a disappearance
 * is meaningful, not an accident.
 */
export async function syncBackToStorage(
  storage: StorageAdapter,
  prefix: string,
  ws: StagedWorkspace,
  logger?: Logger,
): Promise<SyncResult> {
  const result: SyncResult = { written: 0, deleted: 0, unchanged: 0 };

  try {
    const present = await listRelative(ws.dir);
    const presentSet = new Set(present);

    for (const rel of present) {
      // Normalize to forward slashes: storage keys are POSIX-style even when
      // staged on a platform whose path.join emits backslashes.
      const key = rel.split(path.sep).join("/");
      const content = await fsPromises.readFile(path.join(ws.dir, rel), "utf8");
      if (ws.staged.get(key) === content) {
        result.unchanged++;
        continue;
      }
      await storage.writeFile(`${prefix}${key}`, content);
      result.written++;
    }

    for (const key of ws.staged.keys()) {
      const asPath = key.split("/").join(path.sep);
      if (presentSet.has(asPath) || presentSet.has(key)) continue;
      await storage.unlink(`${prefix}${key}`);
      result.deleted++;
    }

    logger?.debug?.(
      `${TAG} synced back to "${prefix}": ${result.written} written, ` +
      `${result.deleted} deleted, ${result.unchanged} unchanged`,
    );
    return result;
  } finally {
    // Always clean up, even if a write failed partway — leaving temp dirs
    // behind would accumulate one per L2 cycle.
    await fsPromises.rm(ws.dir, { recursive: true, force: true }).catch(() => {});
  }
}
