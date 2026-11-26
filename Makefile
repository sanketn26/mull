# Developer entry points for Mull (a fork of AirLLM; import package: mull).
#
#   make setup   install Python, the package (editable) and dev tools with uv (creates .venv)
#   make run     generate tokens (MODEL=, PROMPT=, MAX_NEW_TOKENS=, COMPRESSION=4bit|8bit, MEMORY_PERCENT=)
#   make serve   start the job server (MODEL= a packed MLX checkpoint, PORT=8765, WEB=1 for the web interface)
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
PORT           ?= 8765
WEB            ?=
KEEP_FREE      ?=
MAX_BATCH      ?=

# Extras from pyproject.toml: the MLX runtime on macOS. Dev tools come from the default `dev` group.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
EXTRAS := --extra mac --extra mcp
else
EXTRAS := --extra mcp
endif

PYTEST_ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help setup lock run serve test lint clean clean-all clean-prof

help:
	@sed -n '3,10p' Makefile | sed 's/^# \{0,1\}//'

setup:
	$(UV) sync $(EXTRAS)

lock:
	$(UV) lock

run:
	$(UV) run $(EXTRAS) python examples/run_inference.py --model "$(MODEL)" --prompt "$(PROMPT)" \
		--max-new-tokens $(MAX_NEW_TOKENS) \
		$(if $(COMPRESSION),--compression $(COMPRESSION)) \
		$(if $(MEMORY_PERCENT),--memory-percent $(MEMORY_PERCENT))

serve:
	$(UV) run $(EXTRAS) python -m mull.server --model "$(MODEL)" --port $(PORT) \
		$(if $(MEMORY_PERCENT),--memory-percent $(MEMORY_PERCENT)) $(if $(WEB),--web) \
		$(if $(KEEP_FREE),--keep-free-percent $(KEEP_FREE)) \
		$(if $(MAX_BATCH),--max-batch $(MAX_BATCH)) $(if $(UNLOAD_AFTER),--unload-after-seconds $(UNLOAD_AFTER))

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
