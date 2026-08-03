/**
 * Tests for bridge storage staging.
 *
 * These run against a real StorageAdapter over LocalStorageBackend — no CLI,
 * no network, no model. That is the point: the staging round-trip is where a
 * bug silently destroys user data, and it is pure logic that can be pinned
 * down deterministically.
 *
 * The delete case matters most. L2 merges scene blocks by removing the files
 * it consolidated, so "file gone from the temp dir" must mean "delete from
 * storage" — and must NOT mean that when the model merely didn't touch it.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fsPromises from "node:fs/promises";
import path from "node:path";
import { tmpdir } from "node:os";
import { LocalStorageBackend } from "../../core/storage/local-backend.js";
import { StorageAdapter } from "../../core/storage/adapter.js";
import { stageFromStorage, syncBackToStorage } from "./storage-staging.js";

let rootDir: string;
let storage: StorageAdapter;
const PREFIX = "scene_blocks/";

beforeEach(async () => {
  rootDir = await fsPromises.mkdtemp(path.join(tmpdir(), "staging-test-"));
  storage = new StorageAdapter(new LocalStorageBackend({ rootDir }));
});

afterEach(async () => {
  await fsPromises.rm(rootDir, { recursive: true, force: true });
});

describe("stageFromStorage", () => {
  it("copies every object under the prefix into the temp dir", async () => {
    await storage.writeFile(`${PREFIX}a.md`, "alpha");
    await storage.writeFile(`${PREFIX}b.md`, "beta");

    const ws = await stageFromStorage(storage, PREFIX);
    try {
      expect(ws.staged.size).toBe(2);
      expect(await fsPromises.readFile(path.join(ws.dir, "a.md"), "utf8")).toBe("alpha");
      expect(await fsPromises.readFile(path.join(ws.dir, "b.md"), "utf8")).toBe("beta");
    } finally {
      await fsPromises.rm(ws.dir, { recursive: true, force: true });
    }
  });

  it("yields an empty workspace for an empty prefix rather than throwing", async () => {
    // First L2 run on a fresh install: no scene blocks exist yet and the model
    // is expected to create the first one.
    const ws = await stageFromStorage(storage, PREFIX);
    try {
      expect(ws.staged.size).toBe(0);
      expect(await fsPromises.readdir(ws.dir)).toEqual([]);
    } finally {
      await fsPromises.rm(ws.dir, { recursive: true, force: true });
    }
  });

  it("does not stage objects outside the prefix", async () => {
    await storage.writeFile(`${PREFIX}in.md`, "inside");
    await storage.writeFile("other/out.md", "outside");

    const ws = await stageFromStorage(storage, PREFIX);
    try {
      expect([...ws.staged.keys()]).toEqual(["in.md"]);
    } finally {
      await fsPromises.rm(ws.dir, { recursive: true, force: true });
    }
  });
});

describe("syncBackToStorage", () => {
  it("writes files the model created", async () => {
    const ws = await stageFromStorage(storage, PREFIX);
    await fsPromises.writeFile(path.join(ws.dir, "new.md"), "created", "utf8");

    const result = await syncBackToStorage(storage, PREFIX, ws);

    expect(result).toEqual({ written: 1, deleted: 0, unchanged: 0 });
    expect(await storage.readFile(`${PREFIX}new.md`)).toBe("created");
  });

  it("writes files the model modified", async () => {
    await storage.writeFile(`${PREFIX}a.md`, "before");
    const ws = await stageFromStorage(storage, PREFIX);
    await fsPromises.writeFile(path.join(ws.dir, "a.md"), "after", "utf8");

    const result = await syncBackToStorage(storage, PREFIX, ws);

    expect(result).toEqual({ written: 1, deleted: 0, unchanged: 0 });
    expect(await storage.readFile(`${PREFIX}a.md`)).toBe("after");
  });

  it("deletes files the model removed", async () => {
    // The L2 merge path: two blocks are consolidated into one and the
    // originals are removed.
    await storage.writeFile(`${PREFIX}old-1.md`, "one");
    await storage.writeFile(`${PREFIX}old-2.md`, "two");
    const ws = await stageFromStorage(storage, PREFIX);
    await fsPromises.rm(path.join(ws.dir, "old-1.md"));
    await fsPromises.rm(path.join(ws.dir, "old-2.md"));
    await fsPromises.writeFile(path.join(ws.dir, "merged.md"), "one+two", "utf8");

    const result = await syncBackToStorage(storage, PREFIX, ws);

    expect(result).toEqual({ written: 1, deleted: 2, unchanged: 0 });
    expect(await storage.readFile(`${PREFIX}merged.md`)).toBe("one+two");
    expect(await storage.readFile(`${PREFIX}old-1.md`)).toBeNull();
    expect(await storage.readFile(`${PREFIX}old-2.md`)).toBeNull();
  });

  it("does NOT delete or rewrite files the model left alone", async () => {
    // The regression that would quietly destroy a user's scene library: a
    // read-only run must be a no-op, not a wipe.
    await storage.writeFile(`${PREFIX}keep-1.md`, "one");
    await storage.writeFile(`${PREFIX}keep-2.md`, "two");
    const ws = await stageFromStorage(storage, PREFIX);

    const result = await syncBackToStorage(storage, PREFIX, ws);

    expect(result).toEqual({ written: 0, deleted: 0, unchanged: 2 });
    expect(await storage.readFile(`${PREFIX}keep-1.md`)).toBe("one");
    expect(await storage.readFile(`${PREFIX}keep-2.md`)).toBe("two");
  });

  it("handles a mixed create + modify + delete + untouched run", async () => {
    await storage.writeFile(`${PREFIX}untouched.md`, "same");
    await storage.writeFile(`${PREFIX}edited.md`, "before");
    await storage.writeFile(`${PREFIX}removed.md`, "bye");
    const ws = await stageFromStorage(storage, PREFIX);

    await fsPromises.writeFile(path.join(ws.dir, "edited.md"), "after", "utf8");
    await fsPromises.rm(path.join(ws.dir, "removed.md"));
    await fsPromises.writeFile(path.join(ws.dir, "fresh.md"), "new", "utf8");

    const result = await syncBackToStorage(storage, PREFIX, ws);

    expect(result).toEqual({ written: 2, deleted: 1, unchanged: 1 });
    expect(await storage.readFile(`${PREFIX}untouched.md`)).toBe("same");
    expect(await storage.readFile(`${PREFIX}edited.md`)).toBe("after");
    expect(await storage.readFile(`${PREFIX}removed.md`)).toBeNull();
    expect(await storage.readFile(`${PREFIX}fresh.md`)).toBe("new");
  });

  it("syncs files the model created in a subdirectory", async () => {
    const ws = await stageFromStorage(storage, PREFIX);
    await fsPromises.mkdir(path.join(ws.dir, "sub"), { recursive: true });
    await fsPromises.writeFile(path.join(ws.dir, "sub", "nested.md"), "deep", "utf8");

    const result = await syncBackToStorage(storage, PREFIX, ws);

    expect(result.written).toBe(1);
    expect(await storage.readFile(`${PREFIX}sub/nested.md`)).toBe("deep");
  });

  it("removes the temp dir even when a write fails", async () => {
    const ws = await stageFromStorage(storage, PREFIX);
    await fsPromises.writeFile(path.join(ws.dir, "x.md"), "boom", "utf8");

    const exploding = {
      writeFile: async () => { throw new Error("storage down"); },
      unlink: async () => {},
      readdirNames: async () => [],
      readFile: async () => null,
    } as unknown as StorageAdapter;

    await expect(syncBackToStorage(exploding, PREFIX, ws)).rejects.toThrow("storage down");
    await expect(fsPromises.access(ws.dir)).rejects.toThrow();
  });

  it("round-trips content with newlines and unicode intact", async () => {
    // Scene blocks are markdown with a META header and Chinese prose.
    const body = "-----META-START-----\nheat: 1\n-----META-END-----\n\n## 场景\n多行内容\n";
    await storage.writeFile(`${PREFIX}scene.md`, body);
    const ws = await stageFromStorage(storage, PREFIX);

    const result = await syncBackToStorage(storage, PREFIX, ws);

    expect(result.unchanged).toBe(1);
    expect(await storage.readFile(`${PREFIX}scene.md`)).toBe(body);
  });
});
