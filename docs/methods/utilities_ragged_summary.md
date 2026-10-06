# ragged_summary

`utilities.ragged_summary` runs through the existing native procedure ABI. Python exposes `n4m.roles.RaggedSummary`; generated R and JavaScript constructors expose the same native catalog entry.

An entirely absent cohort is refused by the shared procedure input contract (at least one packed row is required). A dataset with absent sequences and at least one observed packed row is supported with the explicit policy.

Input `X` packs all nonempty sequences into rows, with channels in columns. Required `offsets` start at zero, are nondecreasing, and end at the packed row count. Optional `presence` contains one 0/1 value per sequence (default: all present). Empty sequences require explicit absence; absent sequences must be empty. Optional flat `time_coordinates` must be finite and strictly increasing within each sequence.

The default `missing_policy="reject"` refuses absent sequences. Explicit `zero_with_indicator` emits a zero feature row for absence. Present feature rows contain mean, population standard deviation, minimum and maximum per channel, then sequence length, duration, and presence indicator. Duration uses physical time when supplied and otherwise the last-minus-first integer index. There is no resampling or learned imputation. This stateless projection does not train a target model or make a partial-target CV claim.

Qualification uses independent NumPy reductions and malformed offset, absence, time and nonfinite input cases in `bindings/python/tests/test_ragged_summary.py`.
