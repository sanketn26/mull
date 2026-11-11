# Developer entry points for Mull (a fork of AirLLM; import package: mull).
#
#   make setup   install Python, the package (editable) and dev tools with uv (creates .venv)
#   make run     generate tokens (MODEL=, PROMPT=, MAX_NEW_TOKENS=, COMPRESSION=4bit|8bit, MEMORY_PERCENT=)
#   make test    run the unit tests that work offline on this machine
#   make lint    static checks (ruff)
#   make clean   remove build and cache artifacts (add `make clean-all` to drop .venv too)
#   make lock    refresh uv.lock after changing dependencies in pyproject.toml
#   make clean-prof  delete profiling scripts and synthetic test models in .prof/ (tens of GB)

VENV           ?= .venv
UV             ?= uv

MODEL          ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
PROMPT         ?= What is the capital of the United States?
MAX_NEW_TOKENS ?= 20
COMPRESSION    ?=
MEMORY_PERCENT ?=

# Extras from pyproject.toml: the MLX runtime on macOS. Dev tools come from the default `dev` group.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
EXTRAS := --extra mac
else
EXTRAS :=
endif

PYTEST_ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help setup lock run test lint clean clean-all clean-prof

help:
	@sed -n '3,9p' Makefile | sed 's/^# \{0,1\}//'

setup:
	$(UV) sync $(EXTRAS)

lock:
	$(UV) lock

run:
	$(UV) run $(EXTRAS) python examples/run_inference.py --model "$(MODEL)" --prompt "$(PROMPT)" \
		--max-new-tokens $(MAX_NEW_TOKENS) \
		$(if $(COMPRESSION),--compression $(COMPRESSION)) \
		$(if $(MEMORY_PERCENT),--memory-percent $(MEMORY_PERCENT))

test:
	$(UV) run $(EXTRAS) pytest tests -q $(PYTEST_ARGS)

lint:
	$(UV) run $(EXTRAS) ruff check .

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path './$(VENV)/*' -prune -exec rm -rf {} +
	find . -name .ipynb_checkpoints -type d -not -path './$(VENV)/*' -prune -exec rm -rf {} +

clean-all: clean
	rm -rf $(VENV)

clean-prof:
	rm -rf .prof
