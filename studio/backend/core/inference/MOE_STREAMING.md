# MoE Streaming: running a ~1.5 TB model in 128 GB of RAM

Design reference for Hyposloth Studio's MoE streaming offload planner. This
document is candid about provenance: some methods are established practice
(mmap offload, llama.cpp `--n-cpu-moe`), some are newly synthesized here and
marked as such, and one (router-lookahead prefetch) is an upstream design
proposal that no inference engine implements today. Throughput numbers are
**modelled estimates, not measurements** — no 1.5 TB model was loaded in the
development of this document.

## The problem

A 1.5 TB MoE GGUF (1T+ parameter class: DeepSeek-V3/R1, Kimi-K2, Qwen3-MoE
scale) on a machine with:

- 128 GB system RAM (~110 GB usable after OS + Studio),
- one or more GPUs (24–96 GB VRAM total, possibly zero),
- NVMe SSD (1–7 GB/s sequential read).

Naive loading needs ~1.5 TB of addressable memory. The saving grace is
**sparsity**: a Mixture-of-Experts layer routes each token to a handful of
experts (e.g. 8 of 256), so the bytes actually *touched* per token are a small
fraction of the file. The whole design below exists to turn that observation
into a residency plan: keep what every token needs in RAM/VRAM, stream what
only some tokens need from NVMe, and refuse early when the arithmetic says
the result would be unusably slow.

### Feasibility envelope (first-order math)

```
non_expert_bytes   ≈ 30–80 GB      (attention, embeddings, norms, shared experts)
expert_bytes       ≈ 0.9 × file    (routed experts dominate MoE parameter count)
active bytes/token ≈ expert_bytes × (n_active / n_experts) × miss_fraction
tok/s              ≈ NVMe_BW / active bytes per token   (disk-bound ceiling)
```

Worked example (DeepSeek-V3-ish, Q4, file ≈ 400 GB scaled here to 1.5 TB for
the larger class): 8/256 active experts → ~3% of expert bytes per token before
caching. At 40 GB of naive active bytes/token and 7 GB/s NVMe that is
~0.2 tok/s; with a warm-expert cache at 80% hit rate it becomes ~1 tok/s;
speculative decoding accepting ~2 tokens/step roughly doubles it. This is why
methods 1–8 below all exist: each one attacks a factor in that equation.

**Dense models are out of scope.** A dense 1.5 TB model touches every layer
per token, so active bytes/token ≈ file size. The planner refuses dense
streaming loads with an explicit reason instead of silently delivering
0.001 tok/s.

## Method 1 — Three-tier expert residency

Split the model across three physical tiers instead of llama.cpp's default
two (GPU vs host):

| Tier | Contents | Mechanism |
|------|----------|-----------|
| Hot (VRAM) | non-expert tensors of as many layers as fit; hottest expert layers when headroom remains | `-ngl`, `-ot` device pinning |
| Warm (RAM) | remaining non-expert tensors, KV cache, compute buffers, OS page cache over hot experts | resident set, `--mlock` policy |
| Cold (NVMe) | all remaining routed experts | mmap (default), never mlocked |

In mainline llama.cpp the cold tier is "free": mmap'd tensors are read
through the page cache on demand, and `--n-cpu-moe N` keeps MoE expert
tensors off the GPU entirely (`llama_cpp.py` already owns this flag via
`_resolve_cpu_moe_flag`). Studio's contribution is *sizing* the tiers from a
measured hardware profile (Method 8) rather than a hand-tuned slider.

## Method 2 — Router-logit lookahead prefetch (upstream proposal, novel)

MoE routing is decided per layer by a tiny router network whose input (the
token's hidden state) is available *before* the layer's expert matmuls — and
adjacent layers' routing decisions are strongly correlated with the current
layer's hidden state. That suggests a prefetch pipeline:

1. At layer N, after the router computes top-k experts, also issue an
   *asynchronous* `readahead`/`posix_madvise(WILLNEED)` for the expert slices
   that layer N+1's router is predicted to select (predictor: layer N's
   router probabilities fed through a linear probe trained once per model,
   or simply last-token's layer-N+1 choice, which empirically correlates).
2. The NVMe fetch proceeds concurrently with layer N's attention + shared
   expert compute (~10–50 ms of cover at these model sizes).
3. On a correct prediction the layer N+1 expert read hits the page cache;
   on a miss it degrades to today's synchronous read.

No mainline inference engine implements this; ik_llama.cpp's expert cache is
reactive, not predictive. It is documented here as an upstream proposal with
the interface sketch above; Hyposloth's planner is structured so a future
`prefetch=lookahead` flag can drop into the tiering math without reshaping
the plan object.

## Method 3 — Profile-guided expert pinning

Expert popularity is heavily skewed and stable per workload: a calibration
run over a few hundred representative prompts yields an activation histogram
in which a small subset of (layer, expert) pairs carries most of the mass.
Pinning those to the fastest available tier converts cold NVMe reads into
warm hits.

- Calibration: run the model once with router-statistics logging
  (ik_llama.cpp reports per-expert activation counts; mainline can be
  patched to), dump `{"<layer>.<expert>": weight}` JSON.
- Planning: `moe_streaming_planner.plan_moe_streaming` accepts the profile
  as `StreamingTargets.expert_profile`, sorts MoE layers by summed
  popularity, and pins the hottest layers to VRAM (when headroom exists)
  via `--override-tensor` regexes (`-ot "blk\.(ids)\.ffn_.*_exps\.weight=CUDA0"`).
  Mainline `-ot` matches whole `_exps` tensors, so pinning granularity is a
  MoE layer, not an individual expert; per-expert residency requires
  ik_llama.cpp (Method 9).

## Method 4 — Asymmetric quantization policy

Quality in MoE models is disproportionately carried by the shared path
(attention, shared expert, embeddings); routed experts tolerate aggressive
quantization. Policy when producing or selecting a GGUF for streaming:

- routed expert tensors: ~2-bit class (Q2_K / IQ2_XXS),
- attention + shared expert + embeddings: Q4_K–Q6_K,
- KV cache: q8_0 default, q4_0 under memory pressure (planner downgrades
  automatically when the KV reserve would exceed 25% of the RAM budget).

This is a *build/download-time* decision — the planner does not re-quantize —
but the tier math depends on the resulting sizes, so the policy lives here
next to the planner that consumes it.

## Method 5 — KV-cache diet

At long context the KV cache becomes the second-largest resident consumer
after non-expert weights. The planner reserves KV space *before* sizing the
warm-expert budget (Starving the cache to buy expert warmth trades a hard
OOM for a softer slowdown — the wrong direction). Mechanisms: quantized KV
(Method 4), context caps (`_fit_context_to_vram` already binary-searches max
ctx under a VRAM budget; the planner adds the symmetric RAM budget), slot
reduction (`_slots_that_fit_on_gpu` analogue: parallel slots drop to 1
before context shrinks below 2048, which is the planner's hard floor).

## Method 6 — Speculative decoding as a disk-bandwidth multiplier

Streaming cost is per *forward pass*, not per token. A resident MTP/drafter
head (already budgeted in `llama_cpp.py:_estimate_mtp_overhead_bytes`) lets
each streamed expert serve ~1.5–2.5 accepted tokens, multiplying effective
throughput by the acceptance length. The planner models this as a 1.8×
factor on disk-bound tok/s when the model ships an MTP drafter, and keeps
the drafter resident even at the cost of warm-expert budget — a drafter in
RAM is worth more than a few extra cached experts, because it multiplies
the value of *every* expert read.

## Method 7 — NVMe read-ahead and mlock policy

- `--mlock` (or platform equivalent) applied to the resident set only:
  pinning cold experts into RAM would defeat the design. Mainline llama.cpp
  mlocks the whole model when asked, so streaming plans deliberately leave
  `--mlock` off and rely on the page cache; a per-tensor mlock is future
  work (Method 2's infrastructure would carry it).
- Sequential read-ahead: expert reads are large contiguous slices; the
  planner's disk probe (`disk_speed.probe_sequential_read_mbps`) measures
  the *sequential* rate deliberately, because that is the access pattern
  mmap'd expert reads produce.
- Multi-drive: striping the GGUF shards across NVMe devices (OS-level
  RAID-0 or Storage Spaces) multiplies the disk-bound ceiling linearly;
  the probe measures whichever device the file actually sits on, so plans
  stay honest.

## Method 8 — Measured planning and the throughput estimator

All of the above is driven by `plan_moe_streaming(meta, hardware, targets)`
(`moe_streaming_planner.py`), a pure function that:

1. detects the streaming regime (`file_size > RAM budget + VRAM available`,
   MoE only — dense is refused with a reason),
2. estimates the non-expert resident set and places it VRAM-first
   (`--gpu-layers`, `--n-cpu-moe`),
3. reserves KV (Method 5) and compute buffers, sizes the warm-expert budget
   from what remains,
4. emits `-ot` pinning rules when a popularity profile is supplied
   (Method 3),
5. estimates `stream_bytes_per_token` and `tok/s =
   NVMe_BW / stream_bytes × hit-rate × speculative factor`,
6. **refuses the load** (`feasible=False`, human-readable reason) when the
   shared path alone exceeds RAM+VRAM, or when modelled throughput is below
   0.05 tok/s — a load that would appear hung is worse than a clear error.

Hardware inputs are measured, not assumed: free RAM via psutil, VRAM via the
existing GPU probes, NVMe sequential bandwidth via
`disk_speed.probe_sequential_read_mbps` on the model file itself (a cached
256 MiB timed read; portable, conservative — true cache-defeating reads need
O_DIRECT, which we intentionally avoid for portability).

## Method 9 — ik_llama.cpp runtime expert cache (alternate backend)

Mainline llama.cpp's host/GPU split is *static*: which expert bytes are
warm is decided by the OS page cache, with no per-expert control.
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (a llama.cpp fork
focused on CPU/hybrid MoE inference) adds the missing piece: a **runtime
LRU cache over experts** (`-rtr` / `--repurpose-tensor-cache`-class flags —
exact flag names are probed at runtime via `--help`, see below), which keeps
the most-recently-used experts resident and evicts cold ones, exploiting the
temporal locality of routing decisions across consecutive generated tokens.
On 1T-class MoE + 128 GB RAM this is the difference between ~0.3 and
~2 tok/s.

Studio integrates it as an **alternate inference binary**, not a fork of the
whole stack:

- Installer: `install_llama_prebuilt.py` gains an ik_llama.cpp source
  (separate published repo / release asset naming), exposed as an opt-in
  component so default installs are untouched.
- Backend selection: `LlamaCppBackend` treats "mainline vs ik" as a binary
  flavor; `probe_server_capabilities` already parses `--help` output, and
  ik-specific flags are emitted only when the probed binary advertises
  them — the same gating pattern as `supports_fit_ctx`.
- Planner interplay: when the ik binary is active, the planner's warm tier
  is handed to the runtime LRU instead of static `-ot` rules (which ik
  also honors, but the LRU adapts per-session without a calibration pass).
  A supplied expert profile still seeds the initial pins.

## Integration map

| Piece | Location |
|-------|----------|
| Planner (pure) | `studio/backend/core/inference/moe_streaming_planner.py` |
| Disk probe | `studio/backend/utils/hardware/disk_speed.py` |
| GGUF MoE metadata | `llama_cpp.py:_read_gguf_metadata`, `utils/models/gguf_metadata.py` |
| Load-path wiring | `llama_cpp.py:load_model` auto-mode fit block |
| API surface | `InferenceStatusResponse.streaming_plan` (`models/inference.py`) |
| Frontend badge | `model-config-page.tsx` GPU Memory panel |
| ik_llama.cpp flavor | `install_llama_prebuilt.py`, binary probing in `llama_cpp.py` |

## Honest limitations

- Throughput figures are analytical estimates; the first real 1.5 TB load
  will recalibrate the constants (`cache_miss_fraction`, the 1.8×
  speculative factor, the 0.05 tok/s floor).
- Page-cache warmth is an OS heuristic, not a guarantee; long sessions can
  silently lose warm experts to memory pressure. The ik_llama.cpp LRU
  (Method 9) is the robust answer.
- Windows mmap behavior differs from Linux (no `madvise`); Method 2 is
  therefore POSIX-first in its proposal sketch.
