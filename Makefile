# llm-inference-performance — common workflows (uv-managed env).
.DEFAULT_GOAL := help
PY := uv run

.PHONY: help install bench bench-quick bench-history bench-compare bench-compare-full \
	bench-baseline-update bench-noise tp-demo test check-kernel lint fmt clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv (Python 3.14) and install dev deps
	uv venv --python 3.14
	uv pip install -e ".[dev]"

bench: ## Run the full benchmark suite (writes docs/results.md + JSON)
	$(PY) python benchmarks/run_all.py

bench-quick: ## Fast smoke benchmark (tiny model)
	$(PY) python benchmarks/run_all.py --quick

bench-history: ## Print recent benchmark history for the current configuration
	$(PY) python scripts/bench_history.py $(ARGS)

bench-compare: ## Gate the quick benchmark config against its stored baseline (CI does this)
	$(PY) python scripts/bench_compare.py $(ARGS)

bench-compare-full: ## Gate the full `make bench` config against its stored baseline
	$(PY) python scripts/bench_compare.py --full $(ARGS)

bench-baseline-update: ## Record the current quick-config numbers as the new bench-compare baseline
	$(PY) python scripts/bench_compare.py --update-baseline $(ARGS)

bench-noise: ## Sample bench-compare's config N times and report the spread (default n=10)
	$(PY) python scripts/bench_noise.py $(ARGS)

tp-demo: ## Tensor-parallel demo + all-reduce micro-benchmark (2 procs, gloo)
	$(PY) torchrun --nproc_per_node=2 scripts/tp_demo.py

test: ## Run the test suite (incl. 2-proc gloo tensor-parallel check)
	$(PY) -m pytest

check-kernel: ## Compile + link the CUDA kernel (needs nvcc + a CUDA torch; no GPU)
	$(PY) python scripts/check_kernel_builds.py --self-test

lint: ## Lint with ruff
	$(PY) -m ruff check src benchmarks scripts tests

fmt: ## Auto-format with ruff
	$(PY) -m ruff check --fix src benchmarks scripts tests

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache benchmarks/results/*.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
