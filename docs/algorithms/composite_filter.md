# CompositeFilter (boolean combination of sub-filters)

Aggregates the keep-masks of any number of sub-filters via either ANY or ALL semantics. The composite does NOT replay sub-filter algorithms — it holds references to the sub-filter handles, asks each one for its mask, and combines the masks.

## Modes

```c
typedef enum n4m_composite_mode_t {
    N4M_COMPOSITE_ANY = 0,   /* exclude if ANY sub-filter excludes */
    N4M_COMPOSITE_ALL = 1    /* exclude only if ALL sub-filters exclude */
} n4m_composite_mode_t;
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
n4m_status_t n4m_filter_composite_create(n4m_filter_composite_handle_t** out,
                                          int32_t mode);
void         n4m_filter_composite_destroy(n4m_filter_composite_handle_t* h);

n4m_status_t n4m_filter_composite_add_leverage(n4m_filter_composite_handle_t* h,
                                                n4m_filter_leverage_handle_t* sub);
n4m_status_t n4m_filter_composite_add_quality(n4m_filter_composite_handle_t* h,
                                               n4m_filter_quality_handle_t* sub);

n4m_status_t n4m_filter_composite_apply(const n4m_filter_composite_handle_t* h,
                                         n4m_matrix_view_t X,
                                         uint8_t* mask_out,
                                         n4m_filter_stats_t* stats_out);
```

Phase 12 extends this list with `add_y_outlier`, `add_x_outlier`, `add_metadata` as the matching handles land in the public ABI.

## Numerical contract

Composite is purely a boolean aggregator over already-computed sub-filter masks. The output is exact (mask equality with the internal parity fixture) since no float arithmetic is involved.

## Reference

- nirs4all 0.8.x — `nirs4all.operators.filters.CompositeFilter` in `base.py`
- `parity/python_generator/src/n4m_parity_filters_ref/composite.py`
