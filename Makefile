# Developer entry points for Mull (a fork of AirLLM; import package: mull).
#
#   make setup   create .venv and install the package (editable) plus dev tools
#   make run     generate tokens (MODEL=, PROMPT=, MAX_NEW_TOKENS=, COMPRESSION=4bit|8bit, MEMORY_PERCENT=)
#   make test    run the unit tests that work offline on this machine
#   make lint    static checks (ruff)
#   make clean   remove build and cache artifacts (add `make clean-all` to drop .venv too)
#   make clean-prof  delete profiling scripts and synthetic test models in .prof/ (tens of GB)

PYTHON         ?= python3.12
VENV           ?= .venv
BIN            := $(VENV)/bin

MODEL          ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
PROMPT         ?= What is the capital of the United States?
MAX_NEW_TOKENS ?= 20
COMPRESSION    ?=
MEMORY_PERCENT ?=

# Extras from pyproject.toml: the MLX runtime on macOS, plus test and lint tools everywhere.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
EXTRAS := mac,dev
else
EXTRAS := dev
endif

PYTEST_ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help setup run test lint clean clean-all clean-prof

help:
	@sed -n '3,8p' Makefile | sed 's/^# \{0,1\}//'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

setup: $(BIN)/python
	$(BIN)/pip install -e ".[$(EXTRAS)]"

run: $(BIN)/python
	$(BIN)/python examples/run_inference.py --model "$(MODEL)" --prompt "$(PROMPT)" \
		--max-new-tokens $(MAX_NEW_TOKENS) \
		$(if $(COMPRESSION),--compression $(COMPRESSION)) \
		$(if $(MEMORY_PERCENT),--memory-percent $(MEMORY_PERCENT))

test: $(BIN)/python
	$(BIN)/python -m pytest tests -q $(PYTEST_ARGS)

lint: $(BIN)/python
	$(BIN)/ruff check .

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path './$(VENV)/*' -prune -exec rm -rf {} +
	find . -name .ipynb_checkpoints -type d -not -path './$(VENV)/*' -prune -exec rm -rf {} +

clean-all: clean
	rm -rf $(VENV)

clean-prof:
	rm -rf .prof
