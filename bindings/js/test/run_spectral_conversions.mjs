// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Real WASM: the spectral conversions replay their external-reference fixture
// (parity/spectral_conversions), so the WASM build is checked against
// photobiology, mdatools, astropy, scipy, specdal, spectrolab, prospectr and asdreader.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as n4m from "../dist/index.js";

const FIXTURE = new URL("../../../parity/fixtures/spectral_conversions_external_v1.json", import.meta.url);

// Exact C99 / Python float.hex() parsing ("-0x1.8p+1").
function fromHex(text) {
    const m = /^(-?)0x([0-9a-f]+)(?:\.([0-9a-f]*))?p([+-]\d+)$/i.exec(text);
    if (!m) throw new Error(`not a hexadecimal float: ${text}`);
    const fraction = m[3] ?? "";
    const mantissa = BigInt(`0x${m[2]}${fraction}`);
    const value = Number(mantissa) * 2 ** (Number(m[4]) - 4 * fraction.length);
    return m[1] ? -value : value;
}
const vector = values => values.map(fromHex);
const matrix = rows => {
    const data = Array.isArray(rows[0]) ? rows.map(vector) : [vector(rows)];
    return { data: Float64Array.from(data.flat()), rows: data.length, cols: data[0].length };
};
const params = raw => Object.fromEntries(Object.entries(raw).map(
    ([key, value]) => [key, Array.isArray(value) ? vector(value) : value]));

export function checkSpectralConversions() {
    const doc = JSON.parse(readFileSync(FIXTURE, "utf8"));
    const classes = new Map(Object.values(n4m)
        .filter(c => typeof c === "function" && c.prototype instanceof n4m.NativeEstimator)
        .map(c => [new c().methodId, c]));
    for (const c of doc.cases) {
        const est = new (classes.get(c.method_id))(params(c.params));
        try {
            const X = matrix(c.X);
            est.fit(X, undefined, c.axis ? { axis: Float64Array.from(vector(c.axis)) } : {});
            const got = est.transform(X).data;
            const want = matrix(c.expected).data;
            assert.equal(got.length, want.length, c.name);
            want.forEach((w, i) => assert.ok(Math.abs(got[i] - w) <= c.atol + c.rtol * Math.abs(w),
                `${c.name} [${i}]: ${got[i]} vs ${w} (${c.reference})`));
            if (c.expected_axis) {
                const axis = est.fitResult().axis.data;
                vector(c.expected_axis).forEach((w, i) => assert.ok(
                    Math.abs(axis[i] - w) <= c.axis_rtol * w, `${c.name} axis [${i}]`));
            }
        } finally { est.dispose(); }
    }
    assert.ok(doc.cases.length >= 35);
}
