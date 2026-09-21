# Native cross-language finetuning for nirs4all

**ARCHIVÉ — 17 septembre 2026.** Proposition antérieure à l’implémentation de l’optimiseur natif et des profils HPO DAG-ML.

[Lire le document historique complet](../archives/planification/2026-09-17/NATIVE_FINETUNING.md).

Pour reprendre le travail :

- [Optimisation actuelle](methods/optimization.md).
- [Backlog actif](../../BACKLOG_ECOSYSTEME.md).
- [Premier chantier](../../ROADMAP_CONSOLIDATION_MULTIMODALE.md).

Les anciennes priorités et commandes ne sont plus un plan à exécuter.

<details>
<summary>Anciennes sections — liens vers l’archive</summary>

- <a id="0-tldr--the-recommended-path"></a>[0. TL;DR — the recommended path](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#0-tldr--the-recommended-path)
- <a id="1-three-meanings-of-finetuning-disambiguation-first"></a>[1. Three meanings of "finetuning" (disambiguation first)](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#1-three-meanings-of-finetuning-disambiguation-first)
- <a id="2-état-des-lieux--current-finetuning-per-package"></a>[2. État des lieux — current finetuning, per package](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#2-état-des-lieux--current-finetuning-per-package)
- <a id="21-two-honest-corrections-from-adversarial-verification"></a>[2.1 Two honest corrections from adversarial verification](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#21-two-honest-corrections-from-adversarial-verification)
- <a id="3-the-recommended-architecture"></a>[3. The recommended architecture](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#3-the-recommended-architecture)
- <a id="31-one-rule-two-seams"></a>[3.1 One rule, two seams](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#31-one-rule-two-seams)
- <a id="32-three-flavors-three-loops"></a>[3.2 Three flavors, three loops](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#32-three-flavors-three-loops)
- <a id="33-why-host-drives-the-loop-not-a-chost-objective-callback"></a>[3.3 Why "host drives the loop", not a C→host objective callback](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#33-why-host-drives-the-loop-not-a-chost-objective-callback)
- <a id="4-what-to-build-in-nirs4all-methods"></a>[4. What to build in nirs4all-methods](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#4-what-to-build-in-nirs4all-methods)
- <a id="41-search-space-contract-typed-c-abi-11-with-the-existing-dsl"></a>[4.1 Search-space contract (typed, C-ABI, 1:1 with the existing DSL)](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#41-search-space-contract-typed-c-abi-11-with-the-existing-dsl)
- <a id="42-optimizer-handle--asktell"></a>[4.2 Optimizer handle + ask/tell](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#42-optimizer-handle--asktell)
- <a id="43-pure-native-finetune-zero-host-code-the-portability-win"></a>[4.3 Pure-native finetune (zero host code, the portability win)](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#43-pure-native-finetune-zero-host-code-the-portability-win)
- <a id="44-algorithm-set--priorities"></a>[4.4 Algorithm set + priorities](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#44-algorithm-set--priorities)
- <a id="45-abi-implications"></a>[4.5 ABI implications](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#45-abi-implications)
- <a id="5-what-to-build-in-dag-ml--dag-ml-data"></a>[5. What to build in dag-ml / dag-ml-data](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#5-what-to-build-in-dag-ml--dag-ml-data)
- <a id="6-per-language-idiomatic-finetuning"></a>[6. Per-language idiomatic finetuning](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#6-per-language-idiomatic-finetuning)
- <a id="7-hard-engineering-decisions-to-settle-before-building"></a>[7. Hard engineering decisions to settle *before* building](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#7-hard-engineering-decisions-to-settle-before-building)
- <a id="8-non-goals--what-stays-python-only"></a>[8. Non-goals — what stays Python-only](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#8-non-goals--what-stays-python-only)
- <a id="9-phased-roadmap"></a>[9. Phased roadmap](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#9-phased-roadmap)
- <a id="10-bottom-line"></a>[10. Bottom line](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#10-bottom-line)
- <a id="appendix--evidence-map-fileline"></a>[Appendix — evidence map (file:line)](../archives/planification/2026-09-17/NATIVE_FINETUNING.md#appendix--evidence-map-fileline)

</details>
