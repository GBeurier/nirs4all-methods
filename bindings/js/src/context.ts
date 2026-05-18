// SPDX-License-Identifier: CECILL-2.1

import { checkStatus, getModule } from "./ffi.js";

/** RAII wrapper around p4a_context_t. */
export class Context {
    private _ptr: number;

    private constructor(ptr: number) {
        this._ptr = ptr;
    }

    /** Create a new context. Throws Pls4allError on failure. */
    static create(): Context {
        const m = getModule();
        const out = m._malloc(4);
        try {
            const status = m.ccall(
                "p4a_context_create", "number", ["number"], [out]) as number;
            checkStatus(status);
            const ptr = m.getValue(out, "i32");
            return new Context(ptr);
        } finally {
            m._free(out);
        }
    }

    /** Returns the raw `p4a_context_t*` pointer (handle). */
    get handle(): number {
        return this._ptr;
    }

    /** Free the context. Safe to call multiple times. */
    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall(
            "p4a_context_destroy", null, ["number"], [this._ptr]);
        this._ptr = 0;
    }

    /** Set the RNG seed used by stochastic algorithms. */
    setSeed(seed: bigint | number): void {
        const status = getModule().ccall(
            "p4a_context_set_seed", "number",
            ["number", "number"],
            [this._ptr,
             typeof seed === "bigint" ? Number(seed) : seed]) as number;
        checkStatus(status, this._ptr);
    }
}
