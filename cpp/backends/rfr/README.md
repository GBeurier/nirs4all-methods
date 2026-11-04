# Optional random-forest backend

This directory contains only the header dependency closure required by the
native fANOVA adapter, copied from
[`automl/random_forest_run`](https://github.com/automl/random_forest_run) commit
`1ba17650e734edd3ea536efc91d32815c4874e72`. The reviewed source manifest is
`docs/ranking_external_probes/rfr-header-manifest.txt`; redundant `../` paths in
that manifest are normalized to one physical copy. No binaries or Python
`fanova` implementation are vendored.

The AutoML headers use BSD-3-Clause (`LICENSE.txt`). Their cereal 1.2.2 headers
use BSD-3-Clause (`LICENSE_cereal-1.2.2`). The included RapidJSON source headers
use MIT; upstream `LICENSE_rapidjson` reproduces its complete notice and notices
for other upstream components. No JSON_checker sources or JSON-licensed test
data are included. Individual header notices are retained.

Enable with the `analysis-fanova` or `analysis-all` CMake preset. The default
build excludes this backend. It is a compile-time dependency, with no download,
runtime plugin, or Python dependency. Its headers are private system includes;
no upstream C++ type appears in the public C ABI. Install packaging must retain
all three notices when distributing an enabled binary.

The adapter supplies the standard `<cassert>` header before including upstream
templates, which use `assert` without including its declaration. Integration
and component variance calculations are implemented in the n4m adapter from the
published mathematical decomposition. Coordinate and response normalization
protect upstream numerical arithmetic; the declared product measure is preserved.

One compatibility patch in `cereal/external/rapidjson/document.h` deletes
`GenericStringRef` copy assignment. Upstream's definition assigns to its const
pointer and length members, which Clang rejects when parsing the template.
Explicitly deleting that impossible operation preserves immutable-reference
semantics. The adapter never calls it. All other headers are unmodified.
`SOURCES.sha256` records the shipped header bytes, including this patch.
