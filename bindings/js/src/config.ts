// SPDX-License-Identifier: CeCILL-2.1

import { Algorithm, Deflation, Solver } from "./types.js";
import { checkStatus, getModule } from "./ffi.js";

/** RAII wrapper around p4a_config_t. */
export class Config {
    private _ptr: number;

    private constructor(ptr: number) {
        this._ptr = ptr;
    }

    static create(): Config {
        const m = getModule();
        const out = m._malloc(4);
        try {
            const status = m.ccall(
                "p4a_config_create", "number", ["number"], [out]) as number;
            checkStatus(status);
            return new Config(m.getValue(out, "i32"));
        } finally {
            m._free(out);
        }
    }

    get handle(): number {
        return this._ptr;
    }

    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall(
            "p4a_config_destroy", null, ["number"], [this._ptr]);
        this._ptr = 0;
    }

    setNComponents(k: number): void {
        const s = getModule().ccall(
            "p4a_config_set_n_components", "number",
            ["number", "number"], [this._ptr, k]) as number;
        checkStatus(s);
    }

    setAlgorithm(a: Algorithm): void {
        const s = getModule().ccall(
            "p4a_config_set_algorithm", "number",
            ["number", "number"], [this._ptr, a]) as number;
        checkStatus(s);
    }

    setSolver(solver: Solver): void {
        const s = getModule().ccall(
            "p4a_config_set_solver", "number",
            ["number", "number"], [this._ptr, solver]) as number;
        checkStatus(s);
    }

    setDeflation(d: Deflation): void {
        const s = getModule().ccall(
            "p4a_config_set_deflation", "number",
            ["number", "number"], [this._ptr, d]) as number;
        checkStatus(s);
    }

    setCenterX(on: boolean): void {
        const s = getModule().ccall(
            "p4a_config_set_center_x", "number",
            ["number", "number"], [this._ptr, on ? 1 : 0]) as number;
        checkStatus(s);
    }

    setCenterY(on: boolean): void {
        const s = getModule().ccall(
            "p4a_config_set_center_y", "number",
            ["number", "number"], [this._ptr, on ? 1 : 0]) as number;
        checkStatus(s);
    }

    setStoreScores(on: boolean): void {
        const s = getModule().ccall(
            "p4a_config_set_store_scores", "number",
            ["number", "number"], [this._ptr, on ? 1 : 0]) as number;
        checkStatus(s);
    }
}
