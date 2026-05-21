// SPDX-License-Identifier: CECILL-2.1

import assert from "node:assert/strict";
import * as p4a from "../dist/index.js";

const requireAll = process.env.P4A_REQUIRE_ALL_METHODS === "1";

await p4a.loadModule();

function matrix(rows, cols, fn) {
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      data[i * cols + j] = fn(i, j);
    }
  }
  return { rows, cols, data };
}

function subsetColumns(M, start, count) {
  return matrix(M.rows, count, (i, j) => M.data[i * M.cols + start + j]);
}

const X = matrix(10, 6, (i, j) =>
  Math.sin((i + 1) * (j + 2) * 0.37) + 0.18 * i + 0.07 * j);
const Y = matrix(10, 1, (i) =>
  0.4 * X.data[i * X.cols] - 0.2 * X.data[i * X.cols + 2] + 0.1 * i);
const Y2 = matrix(10, 2, (i, j) => Y.data[i] + 0.2 * (j + 1));
const XTarget = matrix(10, 6, (i, j) =>
  X.data[i * X.cols + j] * 1.03 + 0.05 * Math.cos(i + j));
const labels = Int32Array.from({ length: 10 }, (_, i) => i % 3);
const binaryLabels = Int32Array.from({ length: 10 }, (_, i) => i % 2);
const weights = Float64Array.from({ length: 10 }, (_, i) => 1 + 0.05 * i);
const groups = Int32Array.from([0, 0, 1, 1, 2, 2]);
const blockSizes = BigInt64Array.from([3n, 3n]);
const blocks = [subsetColumns(X, 0, 3), subsetColumns(X, 3, 3)];
const oneSe = matrix(3, 3, (i, j) => 0.4 + 0.1 * i + 0.03 * j);
const alphaThresholds = Float64Array.from([0.05, 0.10, 0.20]);
const thresholds = Float64Array.from([0.2, 0.4, 0.6]);

function makeConfig() {
  const cfg = p4a.Config.create();
  cfg.setAlgorithm(p4a.Algorithm.PLS_REGRESSION);
  cfg.setSolver(p4a.Solver.SIMPLS);
  cfg.setDeflation(p4a.Deflation.REGRESSION);
  cfg.setNComponents(2);
  cfg.setCenterX(true);
  cfg.setCenterY(true);
  cfg.setScaleX(true);
  cfg.setScaleY(true);
  cfg.setStoreScores(true);
  return cfg;
}

function makePlan() {
  const plan = new p4a.ValidationPlan(X.rows);
  plan.addFold([0n, 1n, 2n, 3n, 4n], [5n, 6n, 7n, 8n, 9n]);
  plan.addFold([5n, 6n, 7n, 8n, 9n], [0n, 1n, 2n, 3n, 4n]);
  return plan;
}

function makeAomBank() {
  const bank = new p4a.OperatorBank();
  bank.add(p4a.OperatorKind.IDENTITY);
  bank.add(p4a.OperatorKind.FINITE_DIFFERENCE);
  return bank;
}

function makePreprocessBank() {
  const bank = new p4a.OperatorBank();
  bank.add(p4a.OperatorKind.IDENTITY);
  bank.add(p4a.OperatorKind.CENTER);
  return bank;
}

function assertMatrix(M, label) {
  assert.ok(M.rows > 0, `${label}.rows`);
  assert.ok(M.cols > 0, `${label}.cols`);
  assert.equal(M.data.length, M.rows * M.cols, `${label}.data length`);
  assert.ok(Number.isFinite(M.data[0]), `${label}.data[0]`);
}

function methodResultProbe(key) {
  return (result) => assertMatrix(result.matrix(key), key);
}

function int64Probe(key) {
  return (result) => assert.ok(result.vectorInt64(key).length >= 0, key);
}

const ctx = p4a.Context.create();
const cfg = makeConfig();
let plan = null;
let aomBank = null;
let preprocessBank = null;
let gate = null;
let nativeModel = null;

const tests = [
  ["p4a_sparse_simpls_fit", "sparse_simpls_fit",
    () => p4a.sparse_simpls_fit(ctx, cfg, X, Y, 0.01),
    methodResultProbe("predictions")],
  ["p4a_di_pls_fit", "di_pls_fit",
    () => p4a.di_pls_fit(ctx, cfg, X, Y, XTarget, 0.1),
    methodResultProbe("predictions")],
  ["p4a_recursive_pls_run", "recursive_pls_run",
    () => p4a.recursive_pls_run(ctx, cfg, X, Y, 5),
    methodResultProbe("predictions")],
  ["p4a_cppls_fit", "cppls_fit",
    () => p4a.cppls_fit(ctx, cfg, X, Y, 0.5),
    methodResultProbe("predictions")],
  ["p4a_weighted_pls_fit", "weighted_pls_fit",
    () => p4a.weighted_pls_fit(ctx, cfg, X, Y, weights),
    methodResultProbe("predictions")],
  ["p4a_robust_pls_fit", "robust_pls_fit",
    () => p4a.robust_pls_fit(ctx, cfg, X, Y, 1.35, 3),
    methodResultProbe("predictions")],
  ["p4a_ridge_pls_fit", "ridge_pls_fit",
    () => p4a.ridge_pls_fit(ctx, cfg, X, Y, 0.01),
    methodResultProbe("predictions")],
  ["p4a_continuum_regression_fit", "continuum_regression_fit",
    () => p4a.continuum_regression_fit(ctx, cfg, X, Y, 0.4),
    methodResultProbe("predictions")],
  ["p4a_n_pls_fit", "n_pls_fit",
    () => p4a.n_pls_fit(ctx, cfg, X, 2, 3, Y),
    methodResultProbe("predictions")],
  ["p4a_kernel_pls_fit", "kernel_pls_fit",
    () => p4a.kernel_pls_fit(ctx, cfg, X, Y, 0),
    methodResultProbe("predictions")],
  ["p4a_o2pls_fit", "o2pls_fit",
    () => p4a.o2pls_fit(ctx, cfg, X, Y2, 1, 0, 0),
    methodResultProbe("predictions")],
  ["p4a_approximate_press_compute", "approximate_press_compute",
    () => p4a.approximate_press_compute(ctx, cfg, X, Y, 3),
    methodResultProbe("press_per_component")],
  ["p4a_sparse_pls_da_fit", "sparse_pls_da_fit",
    () => p4a.sparse_pls_da_fit(ctx, cfg, X, labels),
    methodResultProbe("predictions")],
  ["p4a_group_sparse_pls_fit", "group_sparse_pls_fit",
    () => p4a.group_sparse_pls_fit(ctx, cfg, X, Y, groups, 0.05),
    methodResultProbe("predictions")],
  ["p4a_fused_sparse_pls_fit", "fused_sparse_pls_fit",
    () => p4a.fused_sparse_pls_fit(ctx, cfg, X, Y, 0.01, 0.01),
    methodResultProbe("predictions")],
  [["p4a_pls_monitoring_run", "p4a_wasm_pls_model_fit"], "pls_monitoring_run",
    () => p4a.pls_monitoring_run(ctx, nativeModel, X, XTarget, 0.05),
    methodResultProbe("t2")],
  ["p4a_one_se_rule_compute", "one_se_rule_compute",
    () => p4a.one_se_rule_compute(ctx, oneSe),
    (r) => assert.ok(r.vectorInt("best_n_components").length > 0)],
  ["p4a_so_pls_fit", "so_pls_fit",
    () => p4a.so_pls_fit(ctx, cfg, blocks, Y, Int32Array.from([1, 1])),
    methodResultProbe("predictions")],
  ["p4a_on_pls_fit", "on_pls_fit",
    () => p4a.on_pls_fit(ctx, cfg, blocks, 1, Int32Array.from([1, 1])),
    (r) => assert.ok(Number.isFinite(r.scalar("n_blocks")))],
  ["p4a_rosa_fit", "rosa_fit",
    () => p4a.rosa_fit(ctx, cfg, blocks, Y, 2),
    methodResultProbe("predictions")],
  ["p4a_bagging_pls_fit", "bagging_pls_fit",
    () => p4a.bagging_pls_fit(ctx, cfg, X, Y, 3, 123n),
    methodResultProbe("predictions")],
  ["p4a_gpr_pls_fit", "gpr_pls_fit",
    () => p4a.gpr_pls_fit(ctx, cfg, X, Y, 1.0, 0.1, 123n),
    methodResultProbe("predictions")],
  ["p4a_boosting_pls_fit", "boosting_pls_fit",
    () => p4a.boosting_pls_fit(ctx, cfg, X, Y, 3, 0.2),
    methodResultProbe("predictions")],
  ["p4a_random_subspace_pls_fit", "random_subspace_pls_fit",
    () => p4a.random_subspace_pls_fit(ctx, cfg, X, Y, 3, 3, 123n),
    methodResultProbe("predictions")],
  ["p4a_pls_glm_fit", "pls_glm_fit",
    () => p4a.pls_glm_fit(ctx, cfg, X, Y2, false),
    methodResultProbe("predictions")],
  ["p4a_pls_qda_fit", "pls_qda_fit",
    () => p4a.pls_qda_fit(ctx, cfg, X, labels),
    methodResultProbe("predictions")],
  ["p4a_pls_cox_fit", "pls_cox_fit",
    () => p4a.pls_cox_fit(ctx, cfg, X,
      Float64Array.from({ length: 10 }, (_, i) => i + 1),
      binaryLabels),
    methodResultProbe("predictions")],
  ["p4a_pds_fit", "pds_fit",
    () => p4a.pds_fit(ctx, X, XTarget, 1),
    methodResultProbe("predictions")],
  ["p4a_ds_fit", "ds_fit",
    () => p4a.ds_fit(ctx, X, XTarget),
    methodResultProbe("predictions")],
  ["p4a_mir_pls_fit", "mir_pls_fit",
    () => p4a.mir_pls_fit(ctx, cfg, X, Y),
    methodResultProbe("predictions")],
  ["p4a_missing_aware_nipals_fit", "missing_aware_nipals_fit",
    () => p4a.missing_aware_nipals_fit(ctx, cfg, X, Y),
    methodResultProbe("predictions")],
  [["p4a_pls_diagnostics_compute", "p4a_wasm_pls_model_fit"], "pls_diagnostics_compute",
    () => p4a.pls_diagnostics_compute(ctx, nativeModel, X, X),
    methodResultProbe("t2")],
  [["p4a_aom_global_select", "p4a_operator_bank_create", "p4a_validation_plan_create"],
    "aom_global_select",
    () => p4a.aom_global_select(ctx, cfg, aomBank, X, Y, plan, 2),
    (r) => assertMatrix(r.predictions(), "predictions")],
  [["p4a_aom_per_component_select", "p4a_operator_bank_create", "p4a_validation_plan_create"],
    "aom_per_component_select",
    () => p4a.aom_per_component_select(ctx, cfg, aomBank, X, Y, plan, 2),
    (r) => assertMatrix(r.predictions(), "predictions")],
  [["p4a_aom_preprocess_fit", "p4a_operator_bank_create", "p4a_gating_strategy_create",
    "p4a_method_result_get_int64_vector"],
    "aom_preprocess_fit",
    () => p4a.aom_preprocess_fit(ctx, preprocessBank, gate, X, Y),
    methodResultProbe("transformed")],
  ["p4a_mb_pls_fit", "mb_pls_fit",
    () => p4a.mb_pls_fit(ctx, cfg, X, Y, blockSizes),
    methodResultProbe("predictions")],
  [["p4a_lw_pls_fit", "p4a_method_result_get_int64_vector"], "lw_pls_fit",
    () => p4a.lw_pls_fit(ctx, cfg, X, Y, 3),
    int64Probe("neighbor_indices_i64")],
  ["p4a_pls_lda_fit", "pls_lda_fit",
    () => p4a.pls_lda_fit(ctx, cfg, X, labels, 3),
    methodResultProbe("decision_scores")],
  ["p4a_pls_logistic_fit", "pls_logistic_fit",
    () => p4a.pls_logistic_fit(ctx, cfg, X, labels, 3),
    methodResultProbe("probabilities")],
  [["p4a_variable_select_rank", "p4a_wasm_pls_model_fit",
    "p4a_method_result_get_int64_vector"],
    "variable_select_rank",
    () => p4a.variable_select_rank(ctx, nativeModel, X, 0, 3),
    int64Probe("selected_indices")],
  ["p4a_interval_select", "interval_select",
    () => p4a.interval_select(ctx, cfg, X, Y, plan, 2, 1),
    int64Probe("selected_indices")],
  ["p4a_stability_select", "stability_select",
    () => p4a.stability_select(ctx, cfg, X, Y, plan, 3),
    int64Probe("selected_indices")],
  ["p4a_uve_select", "uve_select",
    () => p4a.uve_select(ctx, cfg, X, Y, plan, 2, 123n),
    int64Probe("selected_indices")],
  ["p4a_spa_select", "spa_select",
    () => p4a.spa_select(ctx, cfg, X, Y, 3),
    int64Probe("selected_indices")],
  ["p4a_cars_select", "cars_select",
    () => p4a.cars_select(ctx, cfg, X, Y, plan, 3, 2),
    int64Probe("selected_indices")],
  ["p4a_random_frog_select", "random_frog_select",
    () => p4a.random_frog_select(ctx, cfg, X, Y, plan, 5, 3, 2, 4, 3, 123n),
    int64Probe("selected_indices")],
  ["p4a_scars_select", "scars_select",
    () => p4a.scars_select(ctx, cfg, X, Y, plan, 3, 2, 0.7, 123n),
    int64Probe("selected_indices")],
  ["p4a_ga_select", "ga_select",
    () => p4a.ga_select(ctx, cfg, X, Y, plan, 2, 4, 2, 4, 0.1, 123n),
    int64Probe("selected_indices")],
  ["p4a_pso_select", "pso_select",
    () => p4a.pso_select(ctx, cfg, X, Y, plan, 4, 2, 0.729, 1.494, 1.494, 4.0, 123n),
    int64Probe("selected_indices")],
  ["p4a_vissa_select", "vissa_select",
    () => p4a.vissa_select(ctx, cfg, X, Y, plan, 2, 8, 0.25, 0.5, 0.01, 123n),
    int64Probe("selected_indices")],
  ["p4a_shaving_select", "shaving_select",
    () => p4a.shaving_select(ctx, cfg, X, Y, plan, 2, 2, 0.25),
    int64Probe("selected_indices")],
  ["p4a_bve_select", "bve_select",
    () => p4a.bve_select(ctx, cfg, X, Y, plan, 2, 2),
    int64Probe("selected_indices")],
  ["p4a_t2_select", "t2_select",
    () => p4a.t2_select(ctx, cfg, X, Y, plan, alphaThresholds, 2),
    int64Probe("selected_indices")],
  ["p4a_wvc_select", "wvc_select",
    () => p4a.wvc_select(ctx, X, Y, 2, 3, true),
    int64Probe("selected_indices")],
  ["p4a_wvc_threshold_select", "wvc_threshold_select",
    () => p4a.wvc_threshold_select(ctx, X, Y, 2, true, 0.0, 1.0, 2),
    int64Probe("selected_indices")],
  ["p4a_emcuve_select", "emcuve_select",
    () => p4a.emcuve_select(ctx, cfg, X, Y, plan, 2, 123n, 3, 0.5),
    int64Probe("selected_indices")],
  ["p4a_randomization_select", "randomization_select",
    () => p4a.randomization_select(ctx, cfg, X, Y, 5, 123n, 0.2),
    int64Probe("selected_indices")],
  ["p4a_bipls_select", "bipls_select",
    () => p4a.bipls_select(ctx, cfg, X, Y, plan, 2, 1),
    int64Probe("selected_indices")],
  ["p4a_sipls_select", "sipls_select",
    () => p4a.sipls_select(ctx, cfg, X, Y, plan, 2, 2),
    int64Probe("selected_indices")],
  ["p4a_rep_select", "rep_select",
    () => p4a.rep_select(ctx, cfg, X, Y, plan, 2, 2, 1),
    int64Probe("selected_indices")],
  ["p4a_ipw_select", "ipw_select",
    () => p4a.ipw_select(ctx, cfg, X, Y, plan, 2, 3, 0.5, 0.01),
    int64Probe("selected_indices")],
  ["p4a_st_select", "st_select",
    () => p4a.st_select(ctx, cfg, X, Y, plan, thresholds, 2),
    int64Probe("selected_indices")],
  ["p4a_ecr_fit", "ecr_fit",
    () => p4a.ecr_fit(ctx, cfg, X, Y, 0.5),
    methodResultProbe("predictions")],
  ["p4a_iriv_select", "iriv_select",
    () => p4a.iriv_select(ctx, cfg, X, Y, plan, 1, 123n),
    int64Probe("selected_indices")],
  ["p4a_irf_select", "irf_select",
    () => p4a.irf_select(ctx, cfg, X, Y, plan, 5, 2, 2, 1, 123n),
    int64Probe("selected_indices")],
  ["p4a_vip_spa_select", "vip_spa_select",
    () => p4a.vip_spa_select(ctx, cfg, X, Y, 3, 0.3),
    int64Probe("selected_indices")],
];

function requiredSymbols(spec) {
  return Array.isArray(spec) ? spec : [spec];
}

let passed = 0;
let skipped = 0;
const failures = [];

try {
  if (p4a.hasExportedFunction("p4a_wasm_pls_model_fit")) {
    nativeModel = p4a.NativeModel.fitPls(X, Y, 2, true);
  }
  if (p4a.hasExportedFunction("p4a_validation_plan_create")) {
    plan = makePlan();
  }
  if (p4a.hasExportedFunction("p4a_operator_bank_create")) {
    aomBank = makeAomBank();
    preprocessBank = makePreprocessBank();
  }
  if (p4a.hasExportedFunction("p4a_gating_strategy_create")) {
    gate = new p4a.GatingStrategy(p4a.GatingMode.HARD);
  }

  for (const [symbolsSpec, label, run, probe] of tests) {
    const missing = requiredSymbols(symbolsSpec)
      .filter((symbol) => !p4a.hasExportedFunction(symbol));
    if (missing.length > 0) {
      skipped += 1;
      if (requireAll) {
        failures.push(`${label}: missing ${missing.join(", ")}`);
      }
      continue;
    }
    let result = null;
    try {
      result = run();
      probe(result);
      passed += 1;
    } catch (error) {
      failures.push(`${label}: ${error?.message ?? String(error)}`);
    } finally {
      result?.destroy?.();
    }
  }
} finally {
  nativeModel?.destroy();
  gate?.destroy();
  preprocessBank?.destroy();
  aomBank?.destroy();
  plan?.destroy();
  cfg.destroy();
  ctx.destroy();
}

if (failures.length > 0) {
  console.error(`method smoke failed for WASM ${p4a.version()}`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(
  `method smoke passed (${passed} run, ${skipped} skipped, ` +
  `WASM ${p4a.version()})`,
);
