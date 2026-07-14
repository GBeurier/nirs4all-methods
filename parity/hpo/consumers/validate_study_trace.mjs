#!/usr/bin/env node
// SPDX-License-Identifier: CECILL-2.1
// Independent, dependency-free JavaScript consumer for StudyTrace v1 goldens.

import { readFileSync } from "node:fs";

const terminal = new Set(["COMPLETED", "PRUNED", "FAILED", "CANCELLED"]);
const withError = new Set(["FAILED", "CANCELLED"]);

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function validateStudyTrace(trace) {
  invariant(trace?.format === "n4m.hpo.study_trace", "unsupported StudyTrace format");
  invariant(trace?.schema_version === 1, "unsupported StudyTrace version");
  invariant(Array.isArray(trace.trials) && trace.trials.length > 0, "empty trials");
  invariant(Array.isArray(trace.events), "events must be an array");

  const trials = new Map();
  const intermediateEvents = new Map();
  for (const trial of trace.trials) {
    invariant(Number.isInteger(trial.id) && trial.id >= 0, "invalid trial id");
    invariant(!trials.has(trial.id), `duplicate trial ${trial.id}`);
    invariant(terminal.has(trial.status), `invalid status for trial ${trial.id}`);
    invariant(Array.isArray(trial.intermediates), `missing intermediates for ${trial.id}`);
    invariant(trial.params && typeof trial.params === "object", `missing params for ${trial.id}`);
    if (trial.status === "COMPLETED") {
      invariant(finiteNumber(trial.score), `COMPLETED ${trial.id} needs a finite score`);
    } else {
      invariant(!(Object.hasOwn(trial, "score")), `${trial.status} ${trial.id} has a score`);
    }
    if (withError.has(trial.status)) {
      const error = trial.error;
      invariant(error && typeof error === "object", `${trial.status} ${trial.id} needs error`);
      invariant(/^[A-Z][A-Z0-9_]{1,63}$/.test(error.code), `invalid error code ${trial.id}`);
      invariant(typeof error.retryable === "boolean", `invalid retryable flag ${trial.id}`);
    } else {
      invariant(!(Object.hasOwn(trial, "error")), `${trial.status} ${trial.id} has error`);
    }

    let priorStep = -1;
    const prune = [];
    for (const item of trial.intermediates) {
      invariant(Number.isInteger(item.step) && item.step > priorStep, `steps not ordered ${trial.id}`);
      priorStep = item.step;
      invariant(Number.isInteger(item.sequence), `invalid intermediate sequence ${trial.id}`);
      invariant(!intermediateEvents.has(item.sequence), `duplicate intermediate ${item.sequence}`);
      invariant(finiteNumber(item.score), `non-finite intermediate score ${trial.id}`);
      invariant(typeof item.should_prune === "boolean", `invalid prune flag ${trial.id}`);
      intermediateEvents.set(item.sequence, { trialId: trial.id, item });
      if (item.should_prune) prune.push(item);
    }
    if (trial.status === "PRUNED") {
      invariant(prune.length === 1, `PRUNED ${trial.id} needs one decision`);
      invariant(prune[0] === trial.intermediates.at(-1), `PRUNED ${trial.id} decision not final`);
      invariant(trial.pruned_at === prune[0].step, `pruned_at mismatch ${trial.id}`);
    } else {
      invariant(prune.length === 0, `${trial.status} ${trial.id} has prune decision`);
    }
    trials.set(trial.id, trial);
  }

  const states = new Map([...trials.keys()].map((id) => [id, "NOT_ASKED"]));
  const seenIntermediate = new Set();
  let inFlight = 0;
  let observedMax = 0;
  const configuredMax = trace.study?.execution?.max_in_flight;
  invariant(Number.isInteger(configuredMax) && configuredMax >= 1, "invalid max_in_flight");

  trace.events.forEach((event, sequence) => {
    invariant(event.sequence === sequence, `event sequence gap at ${sequence}`);
    invariant(trials.has(event.trial_id), `unknown event trial ${event.trial_id}`);
    const state = states.get(event.trial_id);
    if (event.kind === "ASK") {
      invariant(state === "NOT_ASKED", `duplicate ASK ${event.trial_id}`);
      states.set(event.trial_id, "RUNNING");
      inFlight += 1;
      observedMax = Math.max(observedMax, inFlight);
      invariant(inFlight <= configuredMax, `max_in_flight exceeded at ${sequence}`);
    } else if (event.kind === "INTERMEDIATE") {
      invariant(state === "RUNNING", `intermediate after terminal ${event.trial_id}`);
      const expected = intermediateEvents.get(sequence);
      invariant(expected?.trialId === event.trial_id, `unmatched intermediate ${sequence}`);
      invariant(expected.item.step === event.step, `intermediate step mismatch ${sequence}`);
      invariant(expected.item.score === event.score, `intermediate score mismatch ${sequence}`);
      invariant(expected.item.should_prune === event.should_prune, `prune mismatch ${sequence}`);
      seenIntermediate.add(sequence);
    } else if (event.kind === "TERMINAL") {
      invariant(state === "RUNNING", `duplicate terminal ${event.trial_id}`);
      const trial = trials.get(event.trial_id);
      invariant(trial.status === event.status, `terminal status mismatch ${event.trial_id}`);
      if (withError.has(event.status)) {
        invariant(event.error_code === trial.error.code, `terminal error mismatch ${event.trial_id}`);
      }
      states.set(event.trial_id, event.status);
      inFlight -= 1;
    } else {
      throw new Error(`unknown event kind ${String(event.kind)}`);
    }
  });

  invariant([...states.values()].every((status) => terminal.has(status)), "open trial remains");
  invariant(seenIntermediate.size === intermediateEvents.size, "missing intermediate event");
  const counts = Object.create(null);
  for (const trial of trials.values()) counts[trial.status] = (counts[trial.status] ?? 0) + 1;
  invariant(trace.summary?.n_trials === trials.size, "summary n_trials mismatch");
  invariant(trace.summary?.n_intermediates === intermediateEvents.size, "summary intermediate mismatch");
  invariant(trace.summary?.max_observed_in_flight === observedMax, "summary concurrency mismatch");
  const summaryCounts = trace.summary?.status_counts;
  invariant(summaryCounts && typeof summaryCounts === "object", "missing summary status counts");
  invariant(Object.keys(summaryCounts).length === Object.keys(counts).length, "summary status mismatch");
  invariant(
    Object.entries(counts).every(([status, count]) => summaryCounts[status] === count),
    "summary status mismatch",
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  invariant(process.argv.length > 2, "usage: validate_study_trace.mjs TRACE.json [...]");
  for (const filename of process.argv.slice(2)) {
    const trace = JSON.parse(readFileSync(filename, "utf8"));
    validateStudyTrace(trace);
    process.stdout.write(`OK ${filename}\n`);
  }
}
