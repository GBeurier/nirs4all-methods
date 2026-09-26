// SPDX-License-Identifier: CECILL-2.1

import { Config } from "./config.js";
import { Context } from "./context.js";
import { MethodResult } from "./methodResult.js";
import { Matrix } from "./types.js";

/** Run native SPA on training rows and return its ordered selected indices. */
export function selectSpa(X: Matrix, Y: Matrix, topK: number,
                          nComponents: number = 2): BigInt64Array {
    if (!Number.isSafeInteger(topK) || topK < 1 || topK > X.cols) {
        throw new RangeError("SPA topK must be between 1 and X.cols.");
    }
    if (!Number.isSafeInteger(nComponents) || nComponents < 1 ||
        nComponents > Math.min(X.cols, X.rows - 1)) {
        throw new RangeError("SPA nComponents exceed the available training rank.");
    }
    if (X.rows !== Y.rows || Y.cols !== 1) {
        throw new RangeError("SPA requires matching X/Y rows and one target column.");
    }
    const context = Context.create();
    try {
        const config = Config.create();
        try {
            config.setNComponents(nComponents);
            const result = MethodResult.run(
                "n4m_feature_selection_spa_select",
                context.handle, config.handle, [X, Y],
                [{ kind: "int", value: topK }],
            );
            try {
                return result.vectorInt64("selected_indices");
            } finally {
                result.destroy();
            }
        } finally {
            config.destroy();
        }
    } finally {
        context.destroy();
    }
}
