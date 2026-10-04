// SPDX-License-Identifier: CECILL-2.1
//
// Public TypeScript API for the @nirs4all/methods binding — a
// non-idiomatic function library over libn4m (raw typed arrays in/out). See
// INPUT_CONTRACT.md and examples/consume.mjs.
//
// Example:
//   import * as n4m from "@nirs4all/methods";
//   await n4m.loadModule();
//   const model = n4m.fitPls({ data: X, rows, cols }, { data: y, rows, cols: 1 }, 3);
//   const preds = n4m.predictPls(model, { data: X, rows, cols });

import { getModule } from "./ffi.js";
export { SpectralEncoder, type SpectralEncodingOptions } from "./spectralEncoding.js";
export { Optimizer, type OptimizerOptions, type OptimizerTrial, type BatchResult,
         type OptimizerTrialRecord, type SearchAxis, type SearchConstraint,
         type TrialValue } from "./optimization.js";

export { loadModule, getModule, makeMatrixView, readArrayView } from "./ffi.js";
export { Context } from "./context.js";
export { Config } from "./config.js";
export { Model, fitPls, predictPls, fitModel, predictModel, fitAom, fitAomChain, fitPop, fitAomRidge, fitAomStack, computeSplit, computeSplitIndices, type PlsModel, type FittedModel, type AomModel, type AomChainDescriptor, type AomChainModel, type PopModel, type AomRidgeOptions, type AomStackOptions, type SplitKind, type SplitOptions, type SplitIndices } from "./model.js";
export {
    ppCreate,
    ppFit,
    ppTransform,
    ppGetState,
    ppSetState,
    ppDestroy,
    type PpOperator,
} from "./preprocessing.js";
export { MethodResult } from "./methodResult.js";
export { NativeModel } from "./nativeModel.js";
export { NativeEstimator, NativeMethod, NativeProcedure, manifest, methodClass, type Augmenter, type Classifier, type FitInputs, type Fold, type ParamType, type ParamValue, type ProbabilisticClassifier, type Procedure, type ProcedureOutput, type Regressor, type SampleFilter, type Selector, type Splitter, type TargetMixingAugmenter, type Transformer } from "./estimatorRoles.js";
export * from "./estimatorRolesGenerated.js";
export { RolePipeline, type ClassLabel, type PipelineRole, type RolePipelineState, type RolePipelineStepInfo, type RoleStep } from "./rolePipeline.js";
export { MultimodalPipeline, MultimodalClassifierPipeline, type MultimodalClassLabel, type MultimodalRecipe, type MultimodalSourceSchema, type MultimodalSourceSchemas, type MultimodalBlocks, type RawTensor } from "./multimodalPipeline.js";
export { splitNative, type NativeSplitterKind,
         type NativeSplitterOptions, type NativeSplitIndices } from "./nativeSplitter.js";
export { augmentNative, type NativeAugmentationKind } from "./nativeAugmentation.js";
export {
    NativePreprocessingPipeline,
    PipelineOperatorKind,
    type PipelineStep,
} from "./nativePreprocessingPipeline.js";
export { selectSpa, selectVariables, selectorMethods } from "./selection.js";
export {
    inspectN4mm,
    SERIALIZED_MODEL_INFO_SCHEMA_V1,
    SERIALIZED_MODEL_CAPABILITY_PREDICT,
    SERIALIZED_MODEL_CAPABILITY_TRANSFORM,
    SERIALIZED_MODEL_CAPABILITY_AFFINE,
    SERIALIZED_MODEL_CAPABILITY_PIPELINE,
    PipelineFingerprintAlgorithm,
    PipelineSemanticProfile,
    SerializedSavitzkyGolayMode,
    SerializedPipelineOperatorKind,
    type SerializedModelInfo,
    type SerializedPipelineInfo,
} from "./serialization.js";
export {
    Status,
    Dtype,
    Algorithm,
    Solver,
    Deflation,
    N4mError,
    type Matrix,
} from "./types.js";

/** ABI / project version reported by the loaded WASM module. */
export function version(): string {
    const m = getModule();
    const ptr = m.ccall("n4m_get_version_string", "number",
                         [], []) as number;
    return ptr === 0 ? "" : m.UTF8ToString(ptr);
}

/** ABI MAJOR.MINOR.PATCH triple. */
export function abiVersion(): readonly [number, number, number] {
    const m = getModule();
    return [
        m.ccall("n4m_get_abi_version_major", "number", [], []) as number,
        m.ccall("n4m_get_abi_version_minor", "number", [], []) as number,
        m.ccall("n4m_get_abi_version_patch", "number", [], []) as number,
    ];
}
