## Goal
INT4 Mixtral-8x7B inference engine with expert offloading, targeting single-T4 (16GB) 
where the ~24GB model doesn't fit. Later: multi-GPU distributed serving on 2xT4.

## Hardware
Kaggle: 2x Tesla T4 (16GB each), 4 CPU cores, ~29GB RAM, PCIe (PHB, no NVLink)

## Constraints
- Correctness before performance (synchronous version first, then async)
- Core logic (cache eviction, transfer scheduling, prefetch policy) is hand-designed 
  by me via spec/pseudocode — implement exactly what's specified, don't redesign
- Metrics required: cache hit/miss, evictions, expert residency, transfer bandwidth, 
  TTFT, TPS, per-stage timing (compute/attention/router/sync)

## Reference
Baseline technique: Eliseev & Mazur LRU offloading (arXiv:2312.17238)