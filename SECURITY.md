# Security Policy

## Supported Versions

`pls4all` is in active pre-release development. The `main` branch and the most recent tagged release are the only versions receiving security updates.

| Version | Supported |
|---------|-----------|
| `main`  | yes |
| `0.x`   | yes (latest tag only) |
| `< 0.x` | no  |

## Reporting a Vulnerability

If you believe you have found a security vulnerability in `pls4all`, please **do not** open a public GitHub issue. Instead, use one of:

1. The GitHub private vulnerability reporting form for this repository (Security tab → "Report a vulnerability").
2. Email the maintainer directly at the address listed on the GitHub profile attached to the most recent release tag.

Please include:

- A clear description of the vulnerability and its impact.
- A minimal reproducer (input data, configuration, expected vs observed behaviour).
- The pls4all version, operating system, compiler, and binding language(s) involved.
- Any suggested mitigations or patches.

You will receive an acknowledgement within 5 business days. We aim to publish a coordinated fix within 30 days for confirmed vulnerabilities; the timeline may be longer for issues that require ABI changes.

## Threat model snapshot

`pls4all` is a numerical library. The most plausible vulnerability classes are:

- Out-of-bounds reads/writes when handling user-supplied `p4a_matrix_view_t` with malformed strides.
- Heap corruption on `p4a_model_import_from_buffer` with crafted serialized models (Phase 1+).
- Integer overflow in dimension arithmetic on very large matrices.
- Denial of service via unbounded iteration counts on adversarial input.

The library does not parse JSON / YAML in the core, does not open network connections, and does not load arbitrary shared libraries at runtime. Optional accelerator backends (BLAS, CUDA) are only invoked when explicitly selected via `p4a_context_set_backend`.
