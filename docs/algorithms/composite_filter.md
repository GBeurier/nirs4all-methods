# CompositeFilter (boolean combination of sub-filters)

Aggregates the keep-masks of any number of sub-filters via either ANY or ALL semantics. The composite does NOT replay sub-filter algorithms — it holds references to the sub-filter handles, asks each one for its mask, and combines the masks.

## Modes

```c
typedef enum c4a_composite_mode_t {
    C4A_COMPOSITE_ANY = 0,   /* exclude if ANY sub-filter excludes */
    C4A_COMPOSITE_ALL = 1    /* exclude only if ALL sub-filters exclude */
} c4a_composite_mode_t;
```

- **ANY** (default): a sample is excluded if ANY sub-filter excludes it. Equivalent to `keep_mask = AND_k(submask_k)`. Most restrictive; only samples that pass every sub-filter are kept.
- **ALL**: a sample is excluded only if ALL sub-filters exclude it. Equivalent to `keep_mask = OR_k(submask_k)`. Most permissive; only samples that fail every sub-filter are dropped.

An empty composite (no sub-filters attached) keeps every sample.

## Ownership

The composite holds REFERENCES to its sub-filters. The caller must:

1. Create each sub-filter independently.
2. Fit any stateful sub-filter before attaching.
3. Attach the sub-filter via the appropriate `add_*` entry point.
4. Destroy the composite first, then destroy each sub-filter separately.

The composite never frees the sub-filter handles.

## ABI

```c
c4a_status_t c4a_filter_composite_create(c4a_filter_composite_handle_t** out,
                                          c4a_composite_mode_t mode);
void         c4a_filter_composite_destroy(c4a_filter_composite_handle_t* h);

c4a_status_t c4a_filter_composite_add_leverage(c4a_filter_composite_handle_t* h,
                                                c4a_filter_leverage_handle_t* sub);
c4a_status_t c4a_filter_composite_add_quality(c4a_filter_composite_handle_t* h,
                                               c4a_filter_quality_handle_t* sub);

c4a_status_t c4a_filter_composite_apply(const c4a_filter_composite_handle_t* h,
                                         c4a_matrix_view_t X,
                                         uint8_t* mask_out,
                                         c4a_filter_stats_t* stats_out);
```

Phase 12 extends this list with `add_y_outlier`, `add_x_outlier`, `add_metadata` as the matching handles land in the public ABI.

## Numerical contract

Composite is purely a boolean aggregator over already-computed sub-filter masks. The output is exact (mask equality with the frozen reference) since no float arithmetic is involved.

## Reference

- nirs4all 0.8.x — `nirs4all.operators.filters.CompositeFilter` in `base.py`
- `parity/python_generator/src/c4a_parity_filters_ref/composite.py`
