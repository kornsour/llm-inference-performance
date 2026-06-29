# llm-inference-performance — common workflows (uv-managed env).
.DEFAULT_GOAL := help
PY := uv run

.PHONY: help install bench bench-quick tp-demo test lint fmt clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv (Python 3.12) and install dev deps
	uv venv --python 3.12
	uv pip install -e ".[dev]"

bench: ## Run the full benchmark suite (writes docs/results.md + JSON)
	$(PY) python benchmarks/run_all.py

bench-quick: ## Fast smoke benchmark (tiny model)
	$(PY) python benchmarks/run_all.py --quick

tp-demo: ## Tensor-parallel demo + all-reduce micro-benchmark (2 procs, gloo)
	$(PY) torchrun --nproc_per_node=2 scripts/tp_demo.py

test: ## Run the test suite (incl. 2-proc gloo tensor-parallel check)
	$(PY) -m pytest

lint: ## Lint with ruff
	$(PY) -m ruff check src benchmarks scripts tests

fmt: ## Auto-format with ruff
	$(PY) -m ruff check --fix src benchmarks scripts tests

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache benchmarks/results/*.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
