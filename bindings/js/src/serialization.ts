// SPDX-License-Identifier: CECILL-2.1
//
// Authoritative inspection of complete N4MM fitted-model payloads.

import { checkStatus, getModule } from "./ffi.js";
import { Algorithm, Deflation, Solver } from "./types.js";

export const SERIALIZED_MODEL_INFO_SCHEMA_V1 = 1;
export const SERIALIZED_MODEL_CAPABILITY_PREDICT = 1n << 0n;
export const SERIALIZED_MODEL_CAPABILITY_TRANSFORM = 1n << 1n;
export const SERIALIZED_MODEL_CAPABILITY_AFFINE = 1n << 2n;
export const SERIALIZED_MODEL_CAPABILITY_PIPELINE = 1n << 3n;

const SERIALIZED_MODEL_INFO_V1_SIZE = 64;
const SERIALIZED_PIPELINE_INFO_V1_SIZE = 96;

export enum SerializedPipelineOperatorKind {
    Snv = 4,
    SavitzkyGolaySmooth = 8,
}

export enum PipelineFingerprintAlgorithm {
    Fnv1a64V1 = 1,
}

export interface SerializedPipelineInfo {
    schemaVersion: number;
    operatorCount: number;
    operators: readonly [SerializedPipelineOperatorKind, SerializedPipelineOperatorKind];
    savgolWindow: number;
    savgolPolyDegree: number;
    savgolDerivative: number;
    savgolDelta: number;
    rawNFeatures: number;
    modelNFeatures: number;
    fingerprintAlgorithm: PipelineFingerprintAlgorithm;
    fingerprint: bigint;
}

export interface SerializedModelInfo {
    schemaVersion: number;
    formatVersion: number;
    writerAbi: readonly [number, number, number];
    algorithm: Algorithm;
    solver: Solver;
    deflation: Deflation;
    trainingSamples: bigint;
    nFeatures: number;
    nTargets: number;
    nComponents: number;
    capabilities: bigint;
    pipeline: SerializedPipelineInfo | null;
}

/** Validate and inspect a complete N4MM payload without importing model state.
 *
 * The capability mask is copied from libn4m's validated result and is never
 * inferred from host-side JSON or other declarative metadata.
 */
export function inspectN4mm(payload: Uint8Array): SerializedModelInfo {
    const m = getModule();
    const payloadPtr = m._malloc(Math.max(1, payload.byteLength));
    const infoPtr = m._malloc(SERIALIZED_MODEL_INFO_V1_SIZE);
    const pipelinePtr = m._malloc(SERIALIZED_PIPELINE_INFO_V1_SIZE);
    try {
        if (payload.byteLength > 0) {
            m.HEAPU8.set(payload, payloadPtr);
        }
        m.HEAPU8.fill(0xa5, infoPtr, infoPtr + SERIALIZED_MODEL_INFO_V1_SIZE);
        const status = m.ccall(
            "n4m_serialization_inspect_model_v1",
            "number",
            ["number", "number", "number"],
            [payloadPtr, payload.byteLength, infoPtr],
        ) as number;
        checkStatus(status);
        m.HEAPU8.fill(0xa5, pipelinePtr, pipelinePtr + SERIALIZED_PIPELINE_INFO_V1_SIZE);
        const pipelineStatus = m.ccall(
            "n4m_serialization_inspect_pipeline_v1",
            "number",
            ["number", "number", "number", "number"],
            [payloadPtr, payload.byteLength, pipelinePtr, SERIALIZED_PIPELINE_INFO_V1_SIZE],
        ) as number;
        checkStatus(pipelineStatus);

        const view = new DataView(
            m.HEAPU8.buffer,
            infoPtr,
            SERIALIZED_MODEL_INFO_V1_SIZE,
        );
        const pipelineView = new DataView(
            m.HEAPU8.buffer,
            pipelinePtr,
            SERIALIZED_PIPELINE_INFO_V1_SIZE,
        );
        const pipeline = pipelineView.getUint32(8, true) === 0 ? null : {
            schemaVersion: pipelineView.getUint32(0, true),
            operatorCount: pipelineView.getUint32(12, true),
            operators: [
                pipelineView.getInt32(16, true) as SerializedPipelineOperatorKind,
                pipelineView.getInt32(20, true) as SerializedPipelineOperatorKind,
            ] as const,
            savgolWindow: pipelineView.getInt32(24, true),
            savgolPolyDegree: pipelineView.getInt32(28, true),
            savgolDerivative: pipelineView.getInt32(32, true),
            savgolDelta: pipelineView.getFloat64(40, true),
            rawNFeatures: pipelineView.getInt32(48, true),
            modelNFeatures: pipelineView.getInt32(52, true),
            fingerprintAlgorithm: pipelineView.getUint32(56, true) as PipelineFingerprintAlgorithm,
            fingerprint: pipelineView.getBigUint64(64, true),
        };
        return {
            schemaVersion: view.getUint32(0, true),
            formatVersion: view.getUint32(4, true),
            writerAbi: [
                view.getUint32(8, true),
                view.getUint32(12, true),
                view.getUint32(16, true),
            ],
            algorithm: view.getInt32(20, true) as Algorithm,
            solver: view.getInt32(24, true) as Solver,
            deflation: view.getInt32(28, true) as Deflation,
            trainingSamples: view.getBigInt64(32, true),
            nFeatures: view.getInt32(40, true),
            nTargets: view.getInt32(44, true),
            nComponents: view.getInt32(48, true),
            capabilities: view.getBigUint64(56, true),
            pipeline,
        };
    } finally {
        m._free(pipelinePtr);
        m._free(infoPtr);
        m._free(payloadPtr);
    }
}
