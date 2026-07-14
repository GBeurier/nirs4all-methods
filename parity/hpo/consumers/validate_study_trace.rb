#!/usr/bin/env ruby
# SPDX-License-Identifier: CECILL-2.1
# Dependency-free Ruby consumer for the binding-neutral StudyTrace v1 fixture.

require "json"

TERMINAL = %w[COMPLETED PRUNED FAILED CANCELLED].freeze
WITH_ERROR = %w[FAILED CANCELLED].freeze

def assert_trace(condition, message)
  raise message unless condition
end

def sphere(params)
  params.sum do |name, value|
    if value.is_a?(String)
      value.empty? ? 0.0 : [value[0].ord - "a".ord, 0].max.to_f
    elsif value == true || value == false
      value ? 0.0 : 1.0
    else
      target = name.start_with?("x") ? 2.0 : 0.0
      (value.to_f - target)**2
    end
  end
end

def validate_native_trace(records)
  assert_trace(!records.empty?, "empty native trace")
  assert_trace(records.map { |trial| trial["id"] } == (0...records.length).to_a,
               "native trial ids are not ordered")
  event_kind = {}
  records.each do |trial|
    status = trial["status"]
    assert_trace([1, 2, 3, 4].include?(status), "unknown native trial status")
    if status == 1
      assert_trace(trial["score"].is_a?(Numeric) && trial["score"].finite?,
                   "completed native trial needs score")
      assert_trace(trial["score"] == sphere(trial["params"]), "objective score mismatch")
    elsif status == 2
      assert_trace(!trial.key?("score"), "pruned native trial has score")
      assert_trace(trial["pruned_at"].is_a?(Integer), "pruned_at missing")
    else
      assert_trace(!trial.key?("score") && trial["error"].is_a?(Hash),
                   "failed native trial needs structured host error")
      code = trial.dig("error", "code")
      assert_trace(code.is_a?(String) && /^[A-Z][A-Z0-9_]{1,63}$/.match?(code),
                   "failed/cancelled native trial error code mismatch")
    end

    intermediates = trial.fetch("intermediates", [])
    intermediates.each do |item|
      sequence = item["sequence"]
      assert_trace(sequence.is_a?(Integer) && !event_kind.key?(sequence),
                   "duplicate native event sequence")
      event_kind[sequence] = :intermediate
    end
    if status == 2
      assert_trace(intermediates[-1]["should_prune"] == true &&
                   intermediates[-1]["step"] == trial["pruned_at"],
                   "native prune decision mismatch")
    else
      assert_trace(intermediates.none? { |item| item["should_prune"] },
                   "non-pruned native trial has prune decision")
    end

    next unless trial.key?("asked_at")

    assert_trace(trial["terminal_at"].is_a?(Integer), "terminal_at missing")
    assert_trace(!event_kind.key?(trial["asked_at"]) && !event_kind.key?(trial["terminal_at"]),
                 "duplicate native event sequence")
    event_kind[trial["asked_at"]] = :ask
    event_kind[trial["terminal_at"]] = :terminal
  end

  sequenced = records.any? { |trial| trial.key?("asked_at") }
  return unless sequenced

  assert_trace(records.all? { |trial| trial.key?("asked_at") }, "partial concurrency metadata")
  assert_trace(event_kind.keys.sort == (0...event_kind.length).to_a, "native event sequence gap")
  active = 0
  observed_max = 0
  event_kind.keys.sort.each do |sequence|
    case event_kind[sequence]
    when :ask
      active += 1
      observed_max = [observed_max, active].max
    when :terminal
      active -= 1
      assert_trace(active >= 0, "terminal before ASK")
    end
  end
  assert_trace(active.zero?, "native in-flight trial remains")
  assert_trace(observed_max >= 1, "native trace never opened a trial")
end

def validate_study_trace(trace)
  assert_trace(trace["format"] == "n4m.hpo.study_trace", "unsupported format")
  assert_trace(trace["schema_version"] == 1, "unsupported version")
  trials = trace.fetch("trials")
  events = trace.fetch("events")
  assert_trace(trials.is_a?(Array) && !trials.empty?, "empty trials")
  assert_trace(events.is_a?(Array), "events must be an array")

  by_id = {}
  intermediate_by_sequence = {}
  trials.each do |trial|
    id = trial["id"]
    status = trial["status"]
    assert_trace(id.is_a?(Integer) && id >= 0 && !by_id.key?(id), "invalid trial id")
    assert_trace(TERMINAL.include?(status), "invalid terminal status")
    assert_trace(trial["params"].is_a?(Hash), "invalid params")
    assert_trace(trial["intermediates"].is_a?(Array), "invalid intermediates")
    assert_trace(status == "COMPLETED" ? trial["score"].is_a?(Numeric) : !trial.key?("score"),
                 "terminal score invariant")
    assert_trace(WITH_ERROR.include?(status) ? trial["error"].is_a?(Hash) : !trial.key?("error"),
                 "structured error invariant")

    previous_step = -1
    prune_items = []
    trial["intermediates"].each do |item|
      assert_trace(item["step"].is_a?(Integer) && item["step"] > previous_step,
                   "intermediate steps not ordered")
      previous_step = item["step"]
      sequence = item["sequence"]
      assert_trace(sequence.is_a?(Integer) && !intermediate_by_sequence.key?(sequence),
                   "duplicate intermediate sequence")
      assert_trace(item["score"].is_a?(Numeric) && item["score"].finite?,
                   "non-finite intermediate score")
      assert_trace([true, false].include?(item["should_prune"]), "invalid prune flag")
      intermediate_by_sequence[sequence] = [id, item]
      prune_items << item if item["should_prune"]
    end
    if status == "PRUNED"
      assert_trace(prune_items.length == 1 && prune_items[0].equal?(trial["intermediates"][-1]),
                   "invalid prune decision")
      assert_trace(trial["pruned_at"] == prune_items[0]["step"], "pruned_at mismatch")
    else
      assert_trace(prune_items.empty?, "non-PRUNED trial has prune decision")
    end
    by_id[id] = trial
  end

  states = by_id.keys.to_h { |id| [id, "NOT_ASKED"] }
  max_in_flight = trace.dig("study", "execution", "max_in_flight")
  assert_trace(max_in_flight.is_a?(Integer) && max_in_flight.positive?, "invalid max_in_flight")
  in_flight = 0
  observed_max = 0
  seen_intermediates = {}
  events.each_with_index do |event, sequence|
    assert_trace(event["sequence"] == sequence, "event sequence gap")
    id = event["trial_id"]
    assert_trace(by_id.key?(id), "unknown event trial")
    case event["kind"]
    when "ASK"
      assert_trace(states[id] == "NOT_ASKED", "duplicate ASK")
      states[id] = "RUNNING"
      in_flight += 1
      observed_max = [observed_max, in_flight].max
      assert_trace(in_flight <= max_in_flight, "max_in_flight exceeded")
    when "INTERMEDIATE"
      assert_trace(states[id] == "RUNNING", "intermediate after terminal")
      expected_id, expected = intermediate_by_sequence.fetch(sequence)
      assert_trace(expected_id == id && expected["step"] == event["step"] &&
                   expected["score"] == event["score"] &&
                   expected["should_prune"] == event["should_prune"],
                   "intermediate event mismatch")
      seen_intermediates[sequence] = true
    when "TERMINAL"
      assert_trace(states[id] == "RUNNING", "duplicate terminal")
      assert_trace(by_id[id]["status"] == event["status"], "terminal status mismatch")
      if WITH_ERROR.include?(event["status"])
        assert_trace(by_id[id].dig("error", "code") == event["error_code"],
                     "terminal error mismatch")
      end
      states[id] = event["status"]
      in_flight -= 1
    else
      raise "unknown event kind"
    end
  end

  assert_trace(states.values.all? { |status| TERMINAL.include?(status) }, "open trial remains")
  assert_trace(seen_intermediates.length == intermediate_by_sequence.length,
               "missing intermediate event")
  counts = trials.map { |trial| trial["status"] }.tally
  summary = trace.fetch("summary")
  assert_trace(summary["n_trials"] == trials.length, "summary n_trials mismatch")
  assert_trace(summary["n_intermediates"] == intermediate_by_sequence.length,
               "summary intermediates mismatch")
  assert_trace(summary["max_observed_in_flight"] == observed_max,
               "summary concurrency mismatch")
  assert_trace(summary["status_counts"] == counts, "summary status mismatch")
end

abort "usage: validate_study_trace.rb TRACE.json [...]" if ARGV.empty?
ARGV.each do |filename|
  document = JSON.parse(File.read(filename, encoding: "UTF-8"))
  document.is_a?(Array) ? validate_native_trace(document) : validate_study_trace(document)
  puts "OK #{filename}"
end
