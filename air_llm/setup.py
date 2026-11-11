import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# The single source of truth for the README is the repository's top-level README.md. The release
# workflow copies it next to this setup.py before building, so it ships in both the sdist and the
# wheel; for a plain local checkout we fall back to the copy one directory up.
long_description = ""
for _readme in (os.path.join(here, "README.md"), os.path.join(here, os.pardir, "README.md")):
    if os.path.exists(_readme):
        with open(_readme, "r", encoding="utf-8") as fh:
            long_description = fh.read()
        break

setuptools.setup(
    # Mull: a fork of AirLLM (Gavin Li, Apache 2.0). The import package is still `airllm`.
    name="mull",
    version="0.1.0",
    author="Sanket Naik; based on AirLLM by Gavin Li",
    description="Mull runs language models larger than memory by streaming layers from disk, "
                "for background work on Apple silicon. A fork of AirLLM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sanketn26/mull",
    packages=setuptools.find_packages(),
    # Keep the dependency surface small and pinned to a tested range so a plain
    # `make setup` gives users a known-good stack with no manual upgrades.
    install_requires=[
        'tqdm',
        'torch>=2.4',
        'transformers>=4.49,<6',
        'accelerate>=1.0',
        'safetensors',
        'huggingface-hub',
        'scipy',
        'sentencepiece',
        # Free-RAM checks for sliding adaptive processing (falls back to os.sysconf without it).
        'psutil',
        # 'bitsandbytes' is optional (used only for --compression); we fall back gracefully when absent.
        # 'compressed-tensors' is optional too: only checkpoints stored in that format (Kimi K3's
        # MXFP4 weights) need it, and transformers raises a clear error naming it when it is missing.
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
