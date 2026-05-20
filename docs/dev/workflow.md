# Development workflow

Every roadmap and every implementation PR follows the loop captured in the master plan and `ROADMAP.md`:

```
1. Author drafts roadmap/phase-N.md or the PR diff.
2. Codex reviews it via `codex exec` (or the project-scoped MCP server
   declared in .mcp.json once Claude has reloaded MCP).
3. Author applies corrections. If the author disagrees with Codex,
   they give Codex more context once; if Codex still disagrees,
   Codex wins.
4. The corrected roadmap / diff lands on `main`.
5. Codex transcripts are checked into docs/reviews/phase-N/.
```

## Building locally

```bash
cmake --preset dev-debug    # or dev-release / ci-* / emscripten / android-*
cmake --build --preset dev-debug --parallel
ctest --preset dev-debug --output-on-failure
./build/dev-debug/cpp/cli/chemometrics4all_cli --selfcheck
```

## Regenerating the parity fixtures

```bash
cd parity/python_generator
pip install -r requirements-lock.txt -e .
generate-fixtures --regenerate   # rewrites parity/fixtures/*.json
generate-fixtures --check        # bit-identity check
```

## Running the parity gate

The unified parity gate is intentionally a local or dedicated-server check, not
a GitHub Actions push/PR workflow. It needs a controlled external reference
environment, especially for `nirs4all`. The resolver uses `NIRS4ALL_ROOT` when
set, otherwise the sibling checkout `../nirs4all/nirs4all`, then an installed
`nirs4all` package.

```bash
cmake --preset ci-linux-gcc12-release
cmake --build --preset ci-linux-gcc12-release --parallel
NIRS4ALL_ROOT=/path/to/nirs4all/nirs4all \
  python parity/python_generator/scripts/run_parity_gate.py \
    --build-dir build/ci-linux-gcc12-release
```

## Codex review (manual invocation)

Until the project-scoped Codex MCP server is wired into Claude Code, reviewers run:

```bash
codex exec -C $REPO -s read-only --skip-git-repo-check \
    -c reasoning_effort=medium \
    -o /tmp/review.md "$(cat review_prompt.txt)"
```

…then archive `/tmp/review.md` and the prompt under `docs/reviews/phase-N/NNNN-<topic>.md`.
