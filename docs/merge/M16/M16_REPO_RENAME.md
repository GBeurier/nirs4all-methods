# M16 — repo rename + donor archive: PROCEDURE

**Status**: documentation only — destructive GitHub operations must be
done by the maintainer. This file describes the exact procedure.

**When to run**: after M5 + M6 + M9 + M10 + M11 + M14 + M15 all land
green, and before tagging `v1.0.0` (M17).

## Step 1 — rename donor repo first

The donor (`GBeurier/nirs4all-methods`) currently owns the target slug.
It must be renamed BEFORE the base repo can take that name.

```bash
# Via gh CLI:
gh repo rename --repo GBeurier/nirs4all-methods nirs4all-methods-archive

# Or via web: Settings → General → Repository name
```

After renaming, the donor URL becomes:
`https://github.com/GBeurier/nirs4all-methods-archive`

GitHub auto-redirects the old URL to the new one for 6 months. Existing
clones with the old URL keep working (with a redirect warning); fresh
clones should use the new URL.

## Step 2 — archive the donor

```bash
gh repo edit GBeurier/nirs4all-methods-archive --enable-issues=false \
    --enable-projects=false --enable-wiki=false
gh api -X PATCH /repos/GBeurier/nirs4all-methods-archive \
    -f archived=true
```

Or via web: Settings → General → Danger Zone → Archive this repository.

## Step 3 — add the redirect NOTICE.md on the archive

```bash
# Locally:
git clone https://github.com/GBeurier/nirs4all-methods-archive.git
cd nirs4all-methods-archive
git checkout main

cat > NOTICE.md <<EOF
# This repository has been merged into nirs4all-methods.

The content of this repository, including its commit history, the 18
`worktree-agent-*` archive tags, and the donor's parity / benchmarks /
binding scaffolds, has been merged into the unified library at:

  **https://github.com/GBeurier/nirs4all-methods**

This repository is preserved as **read-only archive** for citation
stability and DOI continuity. New issues, PRs, and releases land on the
unified repository.

For the merge timeline see the unified repo's MERGE_ROADMAP.md.
EOF

git add NOTICE.md
git commit -m "Archive: redirect to unified nirs4all-methods repo"
git push origin main
```

You will need to **temporarily unarchive** the donor to push the
NOTICE, then re-archive. Workflow:
1. Unarchive (Settings → General → Danger Zone → Unarchive)
2. Push the NOTICE commit
3. Re-archive

## Step 4 — rename base repo to nirs4all-methods

Now the slug `nirs4all-methods` is free.

```bash
gh repo rename --repo GBeurier/pls4all nirs4all-methods
```

The base repo URL becomes:
`https://github.com/GBeurier/nirs4all-methods`

GitHub auto-redirects from `GBeurier/pls4all` for 6 months.

## Step 5 — update local clones

For each contributor:

```bash
cd /path/to/pls4all
git remote set-url origin https://github.com/GBeurier/nirs4all-methods.git
git remote -v   # verify
git fetch       # confirm working
```

## Step 6 — batch import donor issues / PRs / releases

The donor at capture time had 0 issues, 0 PRs, 0 releases, 0 tags
(verified in M0/donor-archive-manifest.json). So this step is **a
no-op** for this merge.

If the donor had accumulated artefacts between M0 and M16, the import
would use:

```bash
# Issues:
gh api 'repos/GBeurier/nirs4all-methods-archive/issues?state=all' \
    --paginate --jq '.[]' > /tmp/donor-issues.json

# For each issue, gh issue create on the unified repo with the original
# title prefix '[archive #N]' to maintain ID continuity.
```

## Step 7 — update CITATION.cff, badges, docs URLs

```bash
cd /path/to/nirs4all-methods  (renamed)
# Edit CITATION.cff: url → https://github.com/GBeurier/nirs4all-methods
# Edit README.md badges
# Edit docs/_extras/*.json that hardcodes the repo URL
```

Then commit + push.

## Step 8 — verify on-ReadTheDocs / GitHub Pages

ReadTheDocs and GitHub Pages may need re-linking:

- ReadTheDocs: Admin → Edit → repository URL → update.
- GitHub Pages: Settings → Pages — if the source is `main`, Pages
  auto-follows. If it's a separate branch like `gh-pages`, no change
  needed.

## Step 9 — final verification

```bash
gh repo view GBeurier/nirs4all-methods --json url,defaultBranchRef,description,isArchived
# expect: not archived, default main, url matches
gh repo view GBeurier/nirs4all-methods-archive --json url,isArchived
# expect: archived = true
```

## Rollback (if something goes wrong)

GitHub repo rename is reversible within the same UI:
- Settings → General → Repository name → change back

Old URL redirects remain active. No data is lost.

For the archive flag: unarchive → re-archive.

For the donor archive's NOTICE.md commit: standard `git revert`.
