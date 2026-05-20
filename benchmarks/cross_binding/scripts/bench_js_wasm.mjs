#!/usr/bin/env node
// SPDX-License-Identifier: CECILL-2.1

import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import * as p4a from "../../../bindings/js/dist/index.js";

const WARMUP_ABORT_MS = 5 * 60 * 1000;
const PROBE_ABORT_MS = 60 * 1000;
const DEFAULT_MAX_RUNS = 40;

function usage() {
  console.error("usage: bench_js_wasm.mjs algo csv_dir n p n_components max_total_runs seed_base predictions_path");
  process.exit(2);
}

if (process.argv.length !== 10) usage();
const [, , algo, csvDir, nRaw, pRaw, ncRaw, runsRaw, seedBaseRaw, predPath] = process.argv;
const n = Number(nRaw);
const p = Number(pRaw);
const nComponentsArg = Number(ncRaw);
const maxRuns = Number(runsRaw);
const seedBase = Number(seedBaseRaw);
const params = JSON.parse(process.env.BENCH_JS_PARAMS_JSON || "{}");
const predictionKey = process.env.BENCH_PREDICTION_KEY || "predictions";

await p4a.loadModule();

function loadDataset(seed) {
  const file = path.join(csvDir, `data_${n}x${p}_seed${seed}.csv`);
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const X = new Float64Array(n * p);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const parts = lines[i + 1].split(",").map(Number);
    for (let j = 0; j < p; j += 1) X[i * p + j] = parts[j];
    y[i] = parts[p];
  }
  return { X: { rows: n, cols: p, data: X }, y };
}

function loadXTarget(seed, X) {
  if (process.env.BENCH_JS_NEEDS_X_TARGET !== "1") return X;
  const dir = process.env.BENCH_JS_X_TARGET_DIR || "";
  const file = path.join(dir, `xtarget_${n}x${p}_seed${seed}.csv`);
  if (!fs.existsSync(file)) return X;
  const rows = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const data = new Float64Array(n * p);
  for (let i = 0; i < n; i += 1) {
    const parts = rows[i].split(",").map(Number);
    for (let j = 0; j < p; j += 1) data[i * p + j] = parts[j];
  }
  return { rows: n, cols: p, data };
}

function buildInputs(X, y, seed) {
  const nTargets = Number(params.n_targets || 1);
  const Y = new Float64Array(n * nTargets);
  for (let i = 0; i < n; i += 1) {
    Y[i * nTargets] = y[i];
    for (let j = 1; j < nTargets; j += 1) {
      Y[i * nTargets + j] = 0.5 * y[i] + 0.1 * X.data[i * p + ((j - 1) % p)];
    }
  }

  const nClasses = Number(params.n_classes || 0);
  const labels = new Int32Array(n);
  if (nClasses >= 2) {
    const order = Array.from({ length: n }, (_, i) => i)
      .sort((a, b) => y[a] - y[b]);
    for (let rank = 0; rank < n; rank += 1) {
      labels[order[rank]] = rank % nClasses;
    }
  } else {
    const sorted = Array.from(y).sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    for (let i = 0; i < n; i += 1) labels[i] = y[i] > median ? 1 : 0;
  }

  const sampleWeights = new Float64Array(n);
  for (let i = 0; i < n; i += 1) sampleWeights[i] = Math.abs(y[i]) + 0.5;

  const nGroups = Math.max(1, Math.min(5, p));
  const width = Math.max(1, Math.ceil(p / nGroups));
  const groupAssignment = new Int32Array(p);
  for (let j = 0; j < p; j += 1) {
    groupAssignment[j] = Math.min(Math.floor(j / width), nGroups - 1);
  }

  return {
    Y: { rows: n, cols: nTargets, data: Y },
    XTarget: loadXTarget(seed, X),
    labels,
    sampleWeights,
    groupAssignment,
  };
}

function cfg(nComponents = params.n_components || nComponentsArg) {
  const c = p4a.Config.create();
  c.setAlgorithm(p4a.Algorithm.PLS_REGRESSION);
  c.setSolver(p4a.Solver.SIMPLS);
  c.setDeflation(p4a.Deflation.REGRESSION);
  c.setNComponents(Math.max(1, Number(nComponents)));
  c.setCenterX(true);
  c.setScaleX(false);
  c.setCenterY(true);
  c.setScaleY(false);
  c.setStoreScores(true);
  return c;
}

function makePlan(nSamples, nFolds = 3) {
  const plan = new p4a.ValidationPlan(nSamples);
  const foldSize = Math.max(1, Math.floor(nSamples / nFolds));
  for (let f = 0; f < nFolds; f += 1) {
    const start = f * foldSize;
    const end = f < nFolds - 1 ? (f + 1) * foldSize : nSamples;
    const train = [];
    const test = [];
    for (let i = 0; i < nSamples; i += 1) {
      (i >= start && i < end ? test : train).push(BigInt(i));
    }
    plan.addFold(train, test);
  }
  return plan;
}

function splitBlocks(X, nBlocks) {
  const blocks = [];
  const colsPerBlock = Math.floor(X.cols / nBlocks);
  for (let b = 0; b < nBlocks; b += 1) {
    const start = b * colsPerBlock;
    const end = b < nBlocks - 1 ? (b + 1) * colsPerBlock : X.cols;
    const cols = end - start;
    const data = new Float64Array(X.rows * cols);
    for (let i = 0; i < X.rows; i += 1) {
      for (let j = 0; j < cols; j += 1) {
        data[i * cols + j] = X.data[i * X.cols + start + j];
      }
    }
    blocks.push({ rows: X.rows, cols, data });
  }
  return blocks;
}

function selectorMask(result, nFeatures) {
  const out = new Float64Array(nFeatures);
  for (const idx of result.vectorInt64("selected_indices")) {
    const j = Number(idx);
    if (j >= 0 && j < nFeatures) out[j] = 1;
  }
  return { rows: 1, cols: nFeatures, data: out };
}

function extract(result, key, nFeatures) {
  if (key === "mask") return selectorMask(result, nFeatures);
  return result.matrix(key);
}

function finish(result, key, nFeatures) {
  try {
    return extract(result, key, nFeatures);
  } finally {
    result.destroy();
  }
}

function compactBank(nOperators) {
  const specs = [
    [p4a.OperatorKind.IDENTITY, []],
    [p4a.OperatorKind.SAVGOL_SMOOTH, [11, 2]],
    [p4a.OperatorKind.SAVGOL_SMOOTH, [21, 3]],
    [p4a.OperatorKind.SAVGOL_DERIVATIVE, [11, 2, 1]],
    [p4a.OperatorKind.SAVGOL_DERIVATIVE, [21, 3, 1]],
    [p4a.OperatorKind.SAVGOL_DERIVATIVE, [11, 2, 2]],
    [p4a.OperatorKind.DETREND_POLY, [1]],
    [p4a.OperatorKind.DETREND_POLY, [2]],
    [p4a.OperatorKind.FINITE_DIFFERENCE, [1]],
  ];
  const bank = new p4a.OperatorBank();
  for (const [kind, opParams] of specs.slice(0, Math.max(1, nOperators))) {
    bank.add(kind, Float64Array.from(opParams));
  }
  return bank;
}

function preprocessBank(nOperators) {
  const kinds = [
    p4a.OperatorKind.IDENTITY,
    p4a.OperatorKind.CENTER,
    p4a.OperatorKind.PARETO_SCALE,
    p4a.OperatorKind.AUTOSCALE,
    p4a.OperatorKind.SNV,
  ];
  const bank = new p4a.OperatorBank();
  for (const kind of kinds.slice(0, Math.max(1, nOperators))) bank.add(kind);
  return bank;
}

function oneSeMatrix(maxComponents, nFolds) {
  const data = new Float64Array(maxComponents * nFolds);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = 0.5 + 0.5 * (((i * 37 + 11) % 997) / 996);
  }
  return { rows: maxComponents, cols: nFolds, data };
}

function corePredict(X, Y, algorithm, solver, deflation) {
  return p4a.fitPredictCore(X, Y, {
    algorithm,
    solver,
    deflation,
    nComponents: params.n_components || nComponentsArg,
    centerX: true,
    scaleX: false,
    centerY: true,
    scaleY: false,
  });
}

function fitPredict(seed) {
  const { X, y } = loadDataset(seed);
  const input = buildInputs(X, y, seed);
  const Y = input.Y;
  const ctx = p4a.Context.create();
  const c = cfg();
  let plan = null;
  let model = null;
  let bank = null;
  let gate = null;
  try {
    switch (algo) {
      case "pls":
        return corePredict(X, Y, p4a.Algorithm.PLS_REGRESSION,
          p4a.Solver.SIMPLS, p4a.Deflation.REGRESSION);
      case "pcr":
        return corePredict(X, Y, p4a.Algorithm.PCR,
          p4a.Solver.SVD, p4a.Deflation.REGRESSION);
      case "opls":
        return corePredict(X, Y, p4a.Algorithm.OPLS,
          p4a.Solver.NIPALS, p4a.Deflation.ORTHOGONAL);
      case "sparse_simpls":
        return finish(p4a.sparse_simpls_fit(ctx, c, X, Y,
          params.sparsity_lambda || 0.01), predictionKey, p);
      case "di_pls":
        return finish(p4a.di_pls_fit(ctx, c, X, Y, input.XTarget,
          params.di_lambda || 0.1), predictionKey, p);
      case "recursive_pls":
        return finish(p4a.recursive_pls_run(ctx, c, X, Y,
          params.window_size || Math.max(5, Math.floor(n / 4))), predictionKey, p);
      case "cppls":
        return finish(p4a.cppls_fit(ctx, c, X, Y, params.gamma || 0.5), predictionKey, p);
      case "weighted_pls":
        return finish(p4a.weighted_pls_fit(ctx, c, X, Y,
          input.sampleWeights), predictionKey, p);
      case "robust_pls":
        return finish(p4a.robust_pls_fit(ctx, c, X, Y,
          params.huber_k || 1.35, params.max_irls_iter || 5), predictionKey, p);
      case "ridge_pls":
        return finish(p4a.ridge_pls_fit(ctx, c, X, Y,
          params.ridge_lambda || 0.01), predictionKey, p);
      case "continuum_regression":
        return finish(p4a.continuum_regression_fit(ctx, c, X, Y,
          params.tau || 0.5), predictionKey, p);
      case "n_pls":
        return finish(p4a.n_pls_fit(ctx, c, X, params.mode_j || 1,
          params.mode_k || p, Y), predictionKey, p);
      case "kernel_pls_rbf":
        return finish(p4a.kernel_pls_fit(ctx, c, X, Y,
          params.kernel_type ?? 1, params.gamma ?? 0.1,
          params.coef0 ?? 1.0, params.degree ?? 3), predictionKey, p);
      case "o2pls":
        return finish(p4a.o2pls_fit(ctx, c, X, Y,
          params.n_predictive || 1, params.n_x_orthogonal || 0,
          params.n_y_orthogonal || 0), predictionKey, p);
      case "approximate_press":
        return finish(p4a.approximate_press_compute(ctx, c, X, Y,
          params.max_components || params.n_components || nComponentsArg),
          predictionKey, p);
      case "pls_diagnostic_t2":
      case "pls_diagnostic_q":
      case "pls_diagnostic_dmodx":
        model = p4a.NativeModel.fitPls(X, Y, params.n_components || nComponentsArg, true);
        return finish(p4a.pls_diagnostics_compute(ctx, model, X, null), predictionKey, p);
      case "pls_monitoring": {
        const split = Math.floor(n / 2);
        const Xref = { rows: split, cols: p, data: X.data.slice(0, split * p) };
        const Xmon = { rows: n - split, cols: p, data: X.data.slice(split * p) };
        const Yref = { rows: split, cols: Y.cols, data: Y.data.slice(0, split * Y.cols) };
        model = p4a.NativeModel.fitPls(Xref, Yref, params.n_components || nComponentsArg, true);
        return finish(p4a.pls_monitoring_run(ctx, model, Xref, Xmon,
          params.alpha || 0.05), predictionKey, p);
      }
      case "one_se_rule":
        return finish(p4a.one_se_rule_compute(ctx,
          oneSeMatrix(params.max_components || 3, params.n_folds || 3)),
          predictionKey, p);
      case "so_pls": {
        const blocks = splitBlocks(X, params.n_blocks || 3);
        return finish(p4a.so_pls_fit(ctx, c, blocks, Y,
          Int32Array.from(params.n_components_per_block || blocks.map(() => 1))),
          predictionKey, p);
      }
      case "on_pls": {
        const blocks = splitBlocks(X, params.n_blocks || 3);
        return finish(p4a.on_pls_fit(ctx, c, blocks, params.n_joint || 1,
          Int32Array.from(params.n_unique_per_block || blocks.map(() => 1))),
          predictionKey, p);
      }
      case "rosa": {
        const blocks = splitBlocks(X, params.n_blocks || 3);
        return finish(p4a.rosa_fit(ctx, c, blocks, Y,
          params.n_components || nComponentsArg), predictionKey, p);
      }
      case "gpr_pls":
        c.setNComponents(params.n_components || nComponentsArg);
        return finish(p4a.gpr_pls_fit(ctx, c, X, Y,
          params.length_scale || 1.0, params.noise_level || 0.1,
          BigInt(params.seed || seed)), predictionKey, p);
      case "bagging_pls":
        return finish(p4a.bagging_pls_fit(ctx, c, X, Y,
          params.n_estimators || 5, BigInt(params.seed || seed)), predictionKey, p);
      case "boosting_pls":
        return finish(p4a.boosting_pls_fit(ctx, c, X, Y,
          params.n_estimators || 5, params.learning_rate || 0.1), predictionKey, p);
      case "random_subspace_pls":
        return finish(p4a.random_subspace_pls_fit(ctx, c, X, Y,
          params.n_estimators || 5, params.features_per_subspace || Math.max(1, Math.floor(p / 2)),
          BigInt(params.seed || seed)), predictionKey, p);
      case "pls_glm":
        return finish(p4a.pls_glm_fit(ctx, c, X, Y, Boolean(params.poisson)),
          predictionKey, p);
      case "pls_qda":
        return finish(p4a.pls_qda_fit(ctx, c, X, input.labels), predictionKey, p);
      case "pls_cox": {
        const times = input.sampleWeights;
        const events = Int32Array.from(input.labels, (label) => label > 0 ? 1 : 0);
        return finish(p4a.pls_cox_fit(ctx, c, X, times, events), predictionKey, p);
      }
      case "pds":
        return finish(p4a.pds_fit(ctx, X, input.XTarget,
          params.window_half_width || 1), predictionKey, p);
      case "ds":
        return finish(p4a.ds_fit(ctx, X, input.XTarget), predictionKey, p);
      case "mir_pls":
        return finish(p4a.mir_pls_fit(ctx, c, X, Y), predictionKey, p);
      case "missing_aware_nipals":
        return finish(p4a.missing_aware_nipals_fit(ctx, c, X, Y), predictionKey, p);
      case "sparse_pls_da":
        return finish(p4a.sparse_pls_da_fit(ctx, c, X, input.labels), predictionKey, p);
      case "group_sparse_pls":
        return finish(p4a.group_sparse_pls_fit(ctx, c, X, Y,
          input.groupAssignment, params.group_lambda || 0.05), predictionKey, p);
      case "fused_sparse_pls":
        return finish(p4a.fused_sparse_pls_fit(ctx, c, X, Y,
          params.l1_lambda || 0.01, params.fusion_lambda || 0.01), predictionKey, p);
      case "mb_pls": {
        const blocks = splitBlocks(X, params.n_blocks || 3);
        const sizes = BigInt64Array.from(blocks.map((b) => BigInt(b.cols)));
        return finish(p4a.mb_pls_fit(ctx, c, X, Y, sizes), predictionKey, p);
      }
      case "lw_pls":
        return finish(p4a.lw_pls_fit(ctx, c, X, Y, params.n_neighbors || 30),
          predictionKey, p);
      case "pls_lda":
        return finish(p4a.pls_lda_fit(ctx, c, X, input.labels,
          params.n_classes || 2), predictionKey, p);
      case "pls_logistic":
        return finish(p4a.pls_logistic_fit(ctx, c, X, input.labels,
          params.n_classes || 2), predictionKey, p);
      case "aom_preprocess":
        bank = preprocessBank(params.n_operators || 3);
        gate = new p4a.GatingStrategy(params.gating_mode || 0);
        return finish(p4a.aom_preprocess_fit(ctx, bank, gate, X, Y), predictionKey, p);
      case "aom_pls":
      case "pop_pls": {
        bank = compactBank(params.n_operators || 9);
        plan = makePlan(n, params.cv || 3);
        const result = algo === "aom_pls"
          ? p4a.aom_global_select(ctx, c, bank, X, Y, plan, params.max_components || 3)
          : p4a.aom_per_component_select(ctx, c, bank, X, Y, plan, params.max_components || 3);
        try {
          return result.predictions();
        } finally {
          result.destroy();
        }
      }
      default:
        return selectorDispatch(ctx, c, X, Y, input, seed);
    }
  } finally {
    model?.destroy();
    gate?.destroy();
    bank?.destroy();
    plan?.destroy();
    c.destroy();
    ctx.destroy();
  }
}

function selectorDispatch(ctx, c, X, Y, input, seed) {
  const plan = makePlan(X.rows, params.cv || 3);
  try {
    switch (algo) {
      case "vissa_select":
        return finish(p4a.vissa_select(ctx, c, X, Y, plan,
          params.n_iterations || 20, params.n_submodels || 100,
          params.ratio_kept || 0.1, params.threshold || 0.5,
          params.floor_probability || 0.01, BigInt(params.seed || seed)), "mask", p);
      case "pso_select":
        return finish(p4a.pso_select(ctx, c, X, Y, plan,
          params.n_swarm || 10, params.n_iterations || 10,
          params.w || 0.729, params.c1 || 1.494, params.c2 || 1.494,
          params.v_max || 4.0, BigInt(params.seed || seed)), "mask", p);
      case "variable_select_vip":
      case "variable_select_coef":
      case "variable_select_sr": {
        const method = algo === "variable_select_vip" ? 0 :
          algo === "variable_select_coef" ? 1 : 2;
        const model = p4a.NativeModel.fitPls(X, Y, params.n_components || nComponentsArg, true);
        try {
          return finish(p4a.variable_select_rank(ctx, model, X, method,
            params.top_k || 10), "mask", p);
        } finally {
          model.destroy();
        }
      }
      case "interval_select":
        return finish(p4a.interval_select(ctx, c, X, Y, plan,
          params.interval_width || 5, params.interval_step || params.step || 1), "mask", p);
      case "bipls_select":
        return finish(p4a.bipls_select(ctx, c, X, Y, plan,
          params.interval_width || 5, params.min_intervals || 1), "mask", p);
      case "sipls_select":
        return finish(p4a.sipls_select(ctx, c, X, Y, plan,
          params.interval_width || 5, params.combination_size || 2), "mask", p);
      case "stability_select":
        return finish(p4a.stability_select(ctx, c, X, Y, plan,
          params.top_k || 10), "mask", p);
      case "uve_select":
        return finish(p4a.uve_select(ctx, c, X, Y, plan,
          params.noise_features || 5, BigInt(params.noise_seed || seed)), "mask", p);
      case "spa_select":
        return finish(p4a.spa_select(ctx, c, X, Y, params.top_k || 10), "mask", p);
      case "cars_select":
        return finish(p4a.cars_select(ctx, c, X, Y, plan,
          params.n_iterations || 10, params.min_features || 5), "mask", p);
      case "random_frog_select":
        return finish(p4a.random_frog_select(ctx, c, X, Y, plan,
          params.n_iterations || 10, params.initial_size || 5,
          params.min_size || 3, params.max_size || 10,
          params.top_k || 10, BigInt(params.seed || seed)), "mask", p);
      case "scars_select":
        return finish(p4a.scars_select(ctx, c, X, Y, plan,
          params.n_iterations || 10, params.min_features || 5,
          params.sample_fraction || 0.8, BigInt(params.seed || seed)), "mask", p);
      case "ga_select":
        return finish(p4a.ga_select(ctx, c, X, Y, plan,
          params.n_generations || 10, params.population_size || 12,
          params.min_features || 3, params.max_features || 10,
          params.mutation_rate || 0.1, BigInt(params.seed || seed)), "mask", p);
      case "shaving_select":
        return finish(p4a.shaving_select(ctx, c, X, Y, plan,
          params.n_steps || 5, params.min_features || 5,
          params.shave_fraction || 0.2), "mask", p);
      case "bve_select":
        return finish(p4a.bve_select(ctx, c, X, Y, plan,
          params.n_steps || 5, params.min_features || 5), "mask", p);
      case "t2_select":
        return finish(p4a.t2_select(ctx, c, X, Y, plan,
          Float64Array.from(params.alpha_thresholds || [0.05, 0.1]),
          params.min_selected || 2), "mask", p);
      case "wvc_select":
        return finish(p4a.wvc_select(ctx, X, Y,
          params.n_components || nComponentsArg, params.top_k || 10,
          params.normalize ?? true), "mask", p);
      case "wvc_threshold_select":
        return finish(p4a.wvc_threshold_select(ctx, X, Y,
          params.n_components || nComponentsArg, params.normalize ?? true,
          params.score_threshold || 0.0, params.threshold_factor || 1.0,
          params.min_selected || 2), "mask", p);
      case "emcuve_select":
        return finish(p4a.emcuve_select(ctx, c, X, Y, plan,
          params.noise_features || 5, BigInt(params.noise_seed || seed),
          params.n_ensembles || 5, params.vote_threshold || 0.5), "mask", p);
      case "randomization_select":
        return finish(p4a.randomization_select(ctx, c, X, Y,
          params.n_permutations || 10, BigInt(params.randomization_seed || seed),
          params.alpha || 0.05), "mask", p);
      case "rep_select":
        return finish(p4a.rep_select(ctx, c, X, Y, plan,
          params.n_steps || 5, params.min_features || 5,
          params.remove_count || 1), "mask", p);
      case "ipw_select":
        return finish(p4a.ipw_select(ctx, c, X, Y, plan,
          params.n_iterations || 5, params.top_k || 10,
          params.damping || 0.5, params.weight_floor || 0.01), "mask", p);
      case "st_select":
        return finish(p4a.st_select(ctx, c, X, Y, plan,
          Float64Array.from(params.thresholds || [0.2, 0.4, 0.6]),
          params.min_selected || 2), "mask", p);
      case "ecr":
        return finish(p4a.ecr_fit(ctx, c, X, Y, params.alpha || 0.5), predictionKey, p);
      case "iriv_select":
        return finish(p4a.iriv_select(ctx, c, X, Y, plan,
          params.max_rounds || 2, BigInt(params.seed || seed)), "mask", p);
      case "irf_select":
        return finish(p4a.irf_select(ctx, c, X, Y, plan,
          params.n_iterations || 10, params.window_size || 5,
          params.initial_intervals || 3, params.top_k || 3,
          BigInt(params.seed || seed)), "mask", p);
      case "vip_spa_select":
        return finish(p4a.vip_spa_select(ctx, c, X, Y,
          params.top_k || 10, params.vip_threshold || 0.3), "mask", p);
      default:
        throw new Error(`unsupported JS/WASM benchmark algorithm: ${algo}`);
    }
  } finally {
    plan.destroy();
  }
}

function flatten(M) {
  return M.data instanceof Float64Array ? M.data : Float64Array.from(M.data);
}

function adaptiveTarget(probeMs, maxTotal) {
  const capped = Math.max(2, maxTotal);
  if (probeMs > PROBE_ABORT_MS) return [2, "single", "probe_gt_60s"];
  if (probeMs > 30000) return [Math.min(capped, 2), "single", "probe_gt_30s"];
  if (probeMs > 5000) return [Math.min(capped, 3), "mean", "probe_gt_5s"];
  if (probeMs > 1000) return [Math.min(capped, 10), "median", "probe_gt_1s"];
  if (probeMs > 100) return [Math.min(capped, 20), "median", "probe_gt_100ms"];
  return [Math.min(capped, DEFAULT_MAX_RUNS), "median", "probe_le_100ms"];
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function stats(samples, statistic, warmupMs, decision, totalRuns, warmupIncluded = false) {
  if (samples.length === 1) statistic = "single";
  const reported = statistic === "mean" ? mean(samples) : median(samples);
  return {
    ok: true,
    n_runs: samples.length,
    median_ms: reported,
    reported_ms: reported,
    sample_median_ms: median(samples),
    mean_ms: mean(samples),
    min_ms: Math.min(...samples),
    max_ms: Math.max(...samples),
    warmup_ms: warmupMs,
    warmup_included: warmupIncluded,
    timing_statistic: statistic,
    timing_decision: decision,
    max_runs: Math.max(1, maxRuns),
    total_runs: totalRuns,
  };
}

function timeRunsSeeded() {
  let t0 = performance.now();
  let warmupPreds = flatten(fitPredict(seedBase));
  const warmupMs = performance.now() - t0;
  if (warmupMs > WARMUP_ABORT_MS || maxRuns < 2) {
    return [
      stats([warmupMs], "single", warmupMs,
        warmupMs > WARMUP_ABORT_MS ? "warmup_gt_5min" : "max_runs_1_warmup_only",
        1, true),
      warmupPreds,
    ];
  }
  const samples = [];
  t0 = performance.now();
  let lastPreds = flatten(fitPredict(seedBase));
  const probeMs = performance.now() - t0;
  samples.push(probeMs);
  const [targetTotal, statistic, decision] = adaptiveTarget(probeMs, maxRuns);
  const targetSamples = Math.max(1, targetTotal - 1);
  for (let i = 1; i < targetSamples; i += 1) {
    t0 = performance.now();
    lastPreds = flatten(fitPredict(seedBase + i));
    samples.push(performance.now() - t0);
  }
  return [stats(samples, statistic, warmupMs, decision, 1 + samples.length), lastPreds];
}

function writeNpy(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  let header = `{'descr': '<f8', 'fortran_order': False, 'shape': (${data.length},), }`;
  const preamble = 10;
  const pad = (16 - ((preamble + header.length + 1) % 16)) % 16;
  header += " ".repeat(pad) + "\n";
  const headerBytes = Buffer.from(header, "latin1");
  const magic = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00]);
  const len = Buffer.alloc(2);
  len.writeUInt16LE(headerBytes.length, 0);
  const payload = Buffer.alloc(data.length * 8);
  for (let i = 0; i < data.length; i += 1) {
    payload.writeDoubleLE(data[i], i * 8);
  }
  fs.writeFileSync(file, Buffer.concat([magic, len, headerBytes, payload]));
}

const [timing, predictions] = timeRunsSeeded();
writeNpy(predPath, predictions);
console.log(JSON.stringify({
  ...timing,
  predictions_path: predPath,
  versions: {
    language: `Node ${process.version}`,
    pls4all: p4a.version(),
    registry_method: algo,
    wasm: "emscripten",
  },
}));
