# MoEInfra
 
A memory-efficient inference engine for large Mixture-of-Experts (MoE) models, built to run **Mixtral-8x7B-Instruct-v0.1** on hardware too small to hold the full model.
 
The core idea is simple: keep quantized expert weights outside the GPU working set, move only the experts required by the current routed tokens onto the GPU, execute them in a bounded working set, and release temporary GPU residency when they are no longer needed.
 
> **Status:** working research prototype on a single NVIDIA T4. NF4-quantized experts, CPU/GPU caching, pinned staging, and chunked expert execution are implemented and benchmarked end-to-end. The major optimization work — persistent GPU caching and predictive prefetching — is still ahead.
 
## Why
 
MoE models cut *compute* per token by activating a handful of experts instead of the whole network but total parameter count stays large, so memory capacity becomes the real constraint. The core engineering problem isn't running the model; it's deciding **which experts live on the GPU, when to fetch the rest, and how to keep that working set bounded.** MoEInfra treats expert weights as an offloaded working set rather than assuming the model fits in VRAM.
 
## Architecture
 
```
tokens → embeddings → decoder layer (attention + MoE)
                            │
                      top-2 router
                            │
              ┌─────────────┴─────────────┐
         GPU cache hit               CPU cache (authoritative)
                            │         → pinned staging → H2D → NF4 reconstruct
              └─────────────┬─────────────┘
                     expert compute → weighted combine
                            │
                  demote temporary GPU experts → next layer
```
 
Routing runs once per token batch; the union of selected experts is processed in bounded chunks (chunk size 2) rather than materializing every expert the sequence touches at once. GPU residency is temporary and disposable and the CPU-side NF4 store is the source of truth.
 
```
MoEInfra/
├── engine/      # inference engine, routing, decoder orchestration
├── cache/       # CPU/GPU expert cache
├── transfer/    # staging, H2D transfer, NF4 reconstruction
├── model/       # expert representation, model loading
├── benchmarks/  # bench_chunked_inference.py — primary end-to-end benchmark
├── metrics/
└── tests/
```
 
## Benchmark
 
| Metric | Cold | Warm (median, 3 runs) |
|---|---:|---:|
| Wall time | 62,139.54 ms | 55,223.26 ms |
| TTFT / prefill | 13,209.33 ms | 6,285.34 ms |
| Decode time | 48,930.14 ms | 49,044.98 ms |
| **Decode throughput** | **0.634 tok/s** | **0.632 tok/s** |
 
Mixtral-8x7B-Instruct-v0.1 · NF4 experts · 32/32 layers · Kaggle T4 · 40-token prompt → 32 generated tokens · peak GPU 3.31 GiB.
 
**Reference point:** [Eliseev & Mazur](https://arxiv.org/abs/2312.17238) report, on T4, 0.661 tok/s for naive 2-bit offloading vs. 2.092 tok/s with their full LRU-cache + speculative-loading system. MoEInfra's current throughput (0.632 tok/s, at the heavier NF4/~4-bit precision) lands in the same range as their *naive* baseline — expected, since persistent GPU caching and prefetching aren't implemented yet. This is not an apples-to-apples benchmark (different hardware instance, quantization bit-width, and prompt), but it's a useful sanity check that the fundamentals are sound and shows the real headroom their full algorithm demonstrates is achievable.
 
## What's implemented
 
- Correct top-2 Mixtral routing (verified against the reference implementation)
- Chunked expert execution with a bounded GPU working set
- CPU-authoritative NF4 expert storage with disposable GPU residency
- Reusable pinned staging for H2D transfer
- Prequantized NF4 expert format with fast CUDA reconstruction
- End-to-end 32-layer inference, measured on real hardware
## Known issues
 
**Kaggle I/O variance.** Copying the ~24 GiB expert dataset into local storage has taken anywhere from ~300s to ~1400s across sessions, consistent with shared cloud storage load rather than anything in the engine itself. Filesystem/cold-start time is reported separately from engine throughput for this reason.
 
## Roadmap
 
1. Per-layer LRU GPU caching (currently a global bounded cache)
2. Speculative next-layer expert prefetching
3. Asynchronous H2D transfer overlapped with compute
4. Contiguous pinned buffers to cut transfer overhead
5. Multi-GPU execution once single-T4 is stable
## Setup
 
```bash
git clone <your-repository-url> && cd MoEInfra
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```
 
Requires a CUDA-enabled PyTorch build and a local copy of Mixtral-8x7B-Instruct-v0.1 plus a prequantized NF4 expert store (256 experts: 32 layers × 8). Update model/expert paths in `config.yaml` to match your filesystem.
 
```bash
pytest -q   # CUDA tests need a GPU; real-model tests need the checkpoint + expert store present
python -m benchmarks.bench_chunked_inference --config config.yaml --max-new-tokens 32 --warm-runs 3 --chunk-size 2
```
 
## Reference
 
Eliseev, A. & Mazur, D. *Fast Inference of Mixture-of-Experts Language Models with Offloading.* arXiv:2312.17238, 2023.
 
## License
 
MIT