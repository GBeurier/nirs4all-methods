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
./build/dev-debug/cpp/cli/pls4all_cli --selfcheck
```

## Regenerating the parity fixtures

```bash
cd parity/python_generator
pip install -r requirements-lock.txt -e .
generate-fixtures --regenerate   # rewrites parity/fixtures/*.json
generate-fixtures --check        # bit-identity check (also run in CI)
```

## Codex review (manual invocation)

Until the project-scoped Codex MCP server is wired into Claude Code, reviewers run:

```bash
codex exec -C $REPO -s read-only --skip-git-repo-check \
    -c reasoning_effort=medium \
    -o /tmp/review.md "$(cat review_prompt.txt)"
```

…then archive `/tmp/review.md` and the prompt under `docs/reviews/phase-N/NNNN-<topic>.md`.
