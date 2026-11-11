# Developer entry points for AirLLM.
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
PKG_DIR        := air_llm

MODEL          ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
PROMPT         ?= What is the capital of the United States?
MAX_NEW_TOKENS ?= 20
COMPRESSION    ?=
MEMORY_PERCENT ?=

# The macOS backend is MLX; it is not a dependency in setup.py, so install it here.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
PLATFORM_DEPS := mlx psutil
else
PLATFORM_DEPS :=
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
	$(BIN)/pip install -e $(PKG_DIR) $(PLATFORM_DEPS) pytest ruff

run: $(BIN)/python
	$(BIN)/python $(PKG_DIR)/run_inference.py --model "$(MODEL)" --prompt "$(PROMPT)" \
		--max-new-tokens $(MAX_NEW_TOKENS) \
		$(if $(COMPRESSION),--compression $(COMPRESSION)) \
		$(if $(MEMORY_PERCENT),--memory-percent $(MEMORY_PERCENT))

test: $(BIN)/python
	cd $(PKG_DIR) && ../$(BIN)/python -m pytest tests -q $(PYTEST_ARGS)

lint: $(BIN)/python
	$(BIN)/ruff check $(PKG_DIR)

clean:
	rm -rf $(PKG_DIR)/build $(PKG_DIR)/dist $(PKG_DIR)/*.egg-info $(PKG_DIR)/README.md
	rm -rf .pytest_cache .ruff_cache $(PKG_DIR)/.pytest_cache $(PKG_DIR)/.ruff_cache
	find . -name __pycache__ -type d -not -path './$(VENV)/*' -prune -exec rm -rf {} +
	find . -name .ipynb_checkpoints -type d -not -path './$(VENV)/*' -prune -exec rm -rf {} +

clean-all: clean
	rm -rf $(VENV)

clean-prof:
	rm -rf .prof
