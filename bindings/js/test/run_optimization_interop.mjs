// SPDX-License-Identifier: CECILL-2.1
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { loadModule, Optimizer } from "../dist/index.js";

await loadModule();
const root = process.argv[2];
if (!root) throw new Error("Expected checkpoint directory");
const space = {
    x: { kind: "int", low: 1, high: 5 },
    a: { kind: "float", low: 0.1, high: 0.9 },
};
const expected = JSON.parse(fs.readFileSync(path.join(root, "expected.json"), "utf8"));
const read = Optimizer.load(fs.readFileSync(path.join(root, "python.n4mopt")), space);
try {
    const next = read.ask();
    assert.deepEqual({ id: Number(next.id), x: Number(next.parameters.x),
                       a: next.parameters.a }, expected);
} finally {
    read.dispose();
}

const write = Optimizer.create(space, { seed: 42 });
try {
    for (let index = 0; index < 3; index += 1) {
        const trial = write.ask();
        write.tell(trial, "completed", Number(trial.parameters.x) + trial.parameters.a);
    }
    fs.writeFileSync(path.join(root, "js.n4mopt"), write.save());
    const next = write.ask();
    assert.deepEqual({ id: Number(next.id), x: Number(next.parameters.x),
                       a: next.parameters.a }, expected);
} finally {
    write.dispose();
}
console.log("JS/WASM resumed Python N4MOPT checkpoint exactly");
