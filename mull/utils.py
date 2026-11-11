"""Compatibility re-exports. New code should import from the focused modules:

* memory      - device choice, freeing memory between layers, free-RAM safety checks
* sliding     - sliding adaptive processing planner
* checkpoint  - downloading, splitting into per-layer shards, reading shards
* compression_bnb - bitsandbytes 4/8-bit shard compression (CUDA)
* ngram       - file-mapped n-gram (PLE) embedding tables
"""
from .checkpoint import (  # noqa: F401
    NotEnoughSpaceException, check_space, cpu_resident_module_names, find_or_create_local_splitted_path,
    layer_owner, layer_tensor_names, link_or_copy_file, load_layer, load_layer_subset,
    remove_real_and_linked_file, split_and_save_layers,
)
from .compression_bnb import (  # noqa: F401
    bitsandbytes_installed, compress_layer_state_dict, save_quant_state_to_dict, uncompress_layer_state_dict,
)
from .memory import (  # noqa: F401
    GC_FULL_EVERY, available_memory_bytes, clean_memory, collect_garbage, default_device,
    has_room_to_prefetch, min_free_ram_bytes, system_ram_bytes,
)
from .ngram import (  # noqa: F401
    MmapEmbedding, _force_meta_embeddings, load_merged_ngram_embedding, merge_ngram_embedding_shards,
    ngram_mmap_exists, open_ngram_mmap_table, persist_ngram_mmap,
)
from .sliding import (  # noqa: F401
    CACHE_READ_SPEEDUP, DEFAULT_MEMORY_PERCENT, DISK_READ_SPEEDUP_BY_WINDOW, MAX_PREFETCH_WINDOW, SlidingPlan,
    plan_resident_layers, plan_sliding, validate_memory_percent,
)
