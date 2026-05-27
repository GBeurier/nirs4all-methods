# n4m developer entry points.
#
# The canonical build system is CMake (use the presets in
# CMakePresets.json). This Makefile is a *thin* convenience layer: it
# wraps Docker / devcontainer plumbing and the heavy parity / dashboard
# operations so they have stable command names.
#
# All targets assume you are running them from the repo root.

SHELL := /bin/bash

DOCKER_COMPOSE := docker compose -f .devcontainer/docker-compose.yml

.DEFAULT_GOAL := help

.PHONY: help
help:
	@printf "n4m developer targets\n\n"
	@printf "  \033[1mDev shell\033[0m\n"
	@printf "    \033[36mmake bootstrap\033[0m        Interactive — prefers devcontainer, falls back to bare-metal\n"
	@printf "    \033[36mmake shell\033[0m            Open an interactive shell inside the devcontainer\n"
	@printf "    \033[36mmake doctor\033[0m           Verify required tools resolve (versions + paths)\n\n"
	@printf "  \033[1mBuild / test\033[0m\n"
	@printf "    \033[36mmake build PRESET=<p>\033[0m Configure + build via cmake preset (default: dev-debug)\n"
	@printf "    \033[36mmake test PRESET=<p>\033[0m  ctest under the given preset\n\n"
	@printf "  \033[1mParity\033[0m\n"
	@printf "    \033[36mmake parity METHOD=<id>\033[0m   Run parity for one method (or METHOD=all)\n"
	@printf "    \033[36mmake snapshot METHOD=<id> REF=<r>\033[0m\n"
	@printf "                                       Regenerate reference snapshot inside the right env image\n"
	@printf "    \033[36mmake parity-paper-only METHOD=<id>\033[0m\n"
	@printf "                                       Run self-consistency checks for paper_only methods\n\n"
	@printf "  \033[1mDocs\033[0m\n"
	@printf "    \033[36mmake docs-refresh\033[0m     Rebuild method/operator pages + benchmark matrix from results/\n\n"
	@printf "  \033[1mDashboard\033[0m\n"
	@printf "    \033[36mmake dashboard-data\033[0m   Aggregate parity + bench JSON into dashboard/public/dashboard.json\n"
	@printf "    \033[36mmake dashboard-serve\033[0m  Local preview (vite dev server)\n"
	@printf "    \033[36mmake dashboard-build\033[0m  Production build of the SPA\n\n"

# ---------------------------------------------------------------------------
# Dev shell
# ---------------------------------------------------------------------------
.PHONY: bootstrap shell doctor

bootstrap:
	@scripts/bootstrap.sh

shell:
	@if [[ -S /var/run/docker.sock ]] || command -v docker >/dev/null 2>&1; then \
		$(DOCKER_COMPOSE) run --rm dev bash; \
	else \
		echo "Docker not available. Run 'make doctor' to verify a bare-metal env."; \
		exit 1; \
	fi

doctor:
	@scripts/doctor.sh

# ---------------------------------------------------------------------------
# Build / test (thin wrappers over CMake presets)
# ---------------------------------------------------------------------------
.PHONY: build test

PRESET ?= dev-debug

build:
	cmake --preset $(PRESET)
	cmake --build --preset $(PRESET) --parallel

test:
	ctest --preset $(PRESET) --output-on-failure

# ---------------------------------------------------------------------------
# Parity (Phase C-min pilot — see parity/SCENARIOS_MIN.md). Full Phase C
# (Docker reference envs, catalog-coupled scenarios, 202-fixture migration,
# the workflow_dispatch CI workflows) is deferred.
# ---------------------------------------------------------------------------
.PHONY: parity snapshot parity-paper-only

METHOD ?= all
REF ?= n4m_self

# Binding/reference parity summary over the committed results (parity/results).
parity:
	python parity/comparator/run.py --method $(METHOD)

# Regenerate the golden n4m_self snapshots (run.py --check is the read-only gate).
snapshot:
	python parity/generators/run.py --method $(METHOD) --ref $(REF) --write

# Self-consistency gate for paper-only differentiators (AOM-PLS, POP-PLS).
parity-paper-only:
	python parity/comparator/self_consistency.py --method $(METHOD)

# ---------------------------------------------------------------------------
# Docs (Sphinx site — method/operator pages + interactive benchmark matrix)
# ---------------------------------------------------------------------------
.PHONY: docs-refresh

docs-refresh:  ## Rebuild all doc artifacts from benchmarks/cross_binding/results
	scripts/refresh_docs.sh

# ---------------------------------------------------------------------------
# Dashboard (wired in Phase D — placeholders document the interface)
# ---------------------------------------------------------------------------
.PHONY: dashboard-data dashboard-serve dashboard-build

dashboard-data:
	@if [[ ! -d dashboard ]]; then \
		echo "dashboard/ not yet bootstrapped (Phase D)."; \
		exit 0; \
	fi
	python dashboard/scripts/aggregate.py

dashboard-serve:
	@if [[ ! -f dashboard/package.json ]]; then echo "dashboard/ not yet bootstrapped (Phase D)."; exit 0; fi
	cd dashboard && npm run dev

dashboard-build:
	@if [[ ! -f dashboard/package.json ]]; then echo "dashboard/ not yet bootstrapped (Phase D)."; exit 0; fi
	cd dashboard && npm run build

# ---------------------------------------------------------------------------
# Skeleton generator (wired in Phase F-prep — stub here)
# ---------------------------------------------------------------------------
.PHONY: new-binding

LANG ?=

new-binding:
	@if [[ -z "$(LANG)" ]]; then echo "Usage: make new-binding LANG=<lang>"; exit 1; fi
	@if [[ ! -x bindings/scripts/new_binding.py ]]; then \
		echo "binding skeleton generator not yet wired (Phase F-prep). Requested LANG=$(LANG)"; \
		exit 0; \
	fi
	python bindings/scripts/new_binding.py --lang $(LANG)
