# R2 native artifact preflight

This page connects the R2 nirs4all native training route to the Methods wire
APIs. It describes a locally qualified release candidate, not a published
package or installer. Availability and digests are authoritative only after the
signed release manifest is locked.

An R2 `nirs4all.run(..., engine="native")` result is a Core Archive V2. The
archive manifest identifies a raw N4MM Methods payload; filename suffixes do
not. Validate the complete payload before importing fitted state.

## Python inspection

```python
from pathlib import Path

from pls4all import (
    SERIALIZED_MODEL_CAPABILITY_PIPELINE,
    SERIALIZED_MODEL_CAPABILITY_PREDICT,
    inspect_n4mm,
)

# An arbitrary extraction name is used: N4MM has no canonical extension.
payload = Path("extracted-methods-payload").read_bytes()
info = inspect_n4mm(payload)

if not info.capabilities & SERIALIZED_MODEL_CAPABILITY_PREDICT:
    raise RuntimeError("artifact is not prediction-capable")

if info.format_version == 2:
    if info.pipeline is None:
        raise RuntimeError("format 2 is missing its attested pipeline")
    if not info.capabilities & SERIALIZED_MODEL_CAPABILITY_PIPELINE:
        raise RuntimeError("pipeline capability is missing")
    print(info.pipeline.fingerprint)
```

The full Python package exposes the same inspector as
`n4m.lowlevel.migration.inspect_n4mm`. Inspection validates the magic, wire
version, recipe, dimensions, array lengths, checksum, exact end of payload and,
for format 2, the complete pipeline attestation. It does not fit, predict, or
import model state.

## JavaScript and C inspection

The JavaScript/WASM binding exposes the same complete-byte gate:

```javascript
import {
  inspectN4mm,
  SERIALIZED_MODEL_CAPABILITY_PREDICT,
} from "@nirs4all/methods";

const info = inspectN4mm(payload);
if ((info.capabilities & SERIALIZED_MODEL_CAPABILITY_PREDICT) === 0n) {
  throw new Error("artifact is not prediction-capable");
}
```

C consumers call `n4m_serialization_inspect_model_v1` and, when plan details
matter, `n4m_serialization_inspect_pipeline_v1`. The latter receives both the
output pointer and its allocation size. See the
[serialization contract](architecture/serialization.md) for the ABI structs,
status codes, zero-on-error rule and fingerprint algorithm.

## Limitations and their preflight

| Public limitation | Preflight | Stable outcome/remediation |
| --- | --- | --- |
| N4MM format 2 represents only exact row-wise SNV (`ddof=0`) -> Savitzky-Golay interpolation (`derivative=0`, `delta=1`) -> PLS regression. It is not a general pipeline envelope. | Require `format_version == 2`, the `PIPELINE` capability and a non-null pipeline descriptor; inspect its semantic profile, ordered operators, parameters, widths and fingerprint. | Unsupported fit recipes fail before export; altered recipes return `N4M_ERR_UNSUPPORTED` or `N4M_ERR_CORRUPT_BUFFER`. Use a supported native pipeline or retain the full source contract. |
| Format 1 is valid raw fitted-model state and carries no pipeline. | `format_version == 1` and `pipeline is None`. | Do not infer preprocessing from a filename or sidecar. Supply preprocessing under a separately versioned contract. |
| Imported affine predictors are prediction-only. | Check `PREDICT | AFFINE`; confirm `TRANSFORM` is absent. | Do not call latent transform APIs for that artifact. |
| Future wire versions are not silently accepted. | Run the inspector on all bytes before import. | `N4M_ERR_VERSION_INCOMPATIBLE`; upgrade the reader or use a supported conversion path. |
| Truncated, checksum-invalid or trailing data is rejected. | Run the inspector before allocating downstream model state. | `N4M_ERR_CORRUPT_BUFFER`; reacquire the artifact. |
| The writer ABI triple is provenance, not a strict load gate. | Read `writer_abi` and apply the release manifest's compatibility policy. | An ABI difference currently records a context warning; never claim that it guarantees rejection. |
| N4MM is neither a full Core Archive V2 nor a `.n4a` bundle and has no canonical standalone extension. | Resolve it through the owning archive manifest, then inspect its bytes. | Keep archive metadata, dataset/pipeline contracts and scientific evidence outside N4MM. |
| Python imports fitted N4MM state; JS currently inspects it; R and MATLAB/Octave have no N4MM import/export wrappers. | Check the selected binding's documented surface before artifact transfer. | Do not infer cross-binding artifact support from method-fit parity. Use C/Python import where supported. |

## Scientific contract

The bounded format-2 plan is intentionally tied to documented scientific
methods. Review the method pages for [SNV](methods/pp_snv.md),
[Savitzky-Golay](methods/pp_savgol.md), and [PLS](methods/pls.md), including
their equations, assumptions, and primary references. The consolidated
[bibliography](bibliography.md) and downloadable
[BibTeX database](_static/bibliography.bib) preserve the scientific citations
used by those pages.

