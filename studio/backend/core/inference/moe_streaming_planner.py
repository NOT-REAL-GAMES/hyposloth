# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MoE streaming offload planner for GGUF MoE models far larger than RAM.

Planning companion to ``llama_cpp.py``: given parsed GGUF MoE metadata, a
probed hardware profile and user targets, decide whether a model that cannot
fit in RAM + VRAM can still be served by streaming cold expert weights from
NVMe, and produce the llama.cpp placement knobs (``--gpu-layers``,
``--n-cpu-moe``, ``-ot`` override-tensor rules, ``--cache-type-k/v``, ctx and
slot caps) plus a first-order throughput estimate.

Design doc: ``MOE_STREAMING.md`` in this directory. The three-tier residency
scheme used here is:

1. VRAM tier -- non-expert tensors (embeddings, attention, norms, shared
   experts) plus, when headroom remains, the hottest expert layers pinned via
   ``-ot`` rules.
2. RAM-warm tier -- the rest of the non-expert path plus as many expert
   layers as the leftover budget holds. llama.cpp mmaps the GGUF, so
   "RAM-warm" is an OS page-cache / mlock distinction, not a copy.
3. NVMe-cold tier -- everything else, demand-paged per token.

The module is deliberately pure stdlib and side-effect free so it can be
unit-tested without torch, psutil or a llama.cpp binary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

# Assumed sequential-read rate when no probe result is available. A mid-range
# PCIe 3.0 NVMe; deliberately mid-pack so an unprobed machine neither looks
# hopeless nor over-promises.
DEFAULT_NVME_MBPS = 3500.0

GIB = 1024 ** 3

# Flat per-process working set llama.cpp needs beyond weights + KV: compute
# buffers, graph workspace, mmap bookkeeping. 2 GiB is the observed floor for
# large-context runs; skimping here causes OOMs only under load.
_COMPUTE_RESERVE_BYTES = 2 * GIB

# Expert share of file bytes when it cannot be derived from metadata, and the
# clamp window when it can. MoE GGUFs that enter the streaming regime are
# expert-dominated almost by definition (a model that is mostly attention
# would not be MoE), so the window is narrow and high.
_EXPERT_SHARE_DEFAULT = 0.9
_EXPERT_SHARE_MIN = 0.5
_EXPERT_SHARE_MAX = 0.97

# KV cache quant policy. q8_0 is the quality default; q4_0 keeps ~0.5625 of
# the q8_0 bytes (4.5 bpw vs 8 bpw + block scales). We downgrade only when the
# q8_0 reservation would eat more than a quarter of the RAM budget, and we
# never let KV -- at any quant -- take more than half of it.
_KV_Q4_SCALE = 0.5625
_KV_Q4_TRIGGER_FRACTION = 0.25
_KV_RAM_CAP_FRACTION = 0.5
_MIN_CTX_TOKENS = 2048

# Throughput band. Below 0.05 tok/s (20 s per token) the session is unusable
# and we refuse instead of launching a server that looks hung. 30 tok/s is
# the optimistic ceiling for a single-user stream; capping there keeps the
# estimate honest when the model is mostly resident.
_MIN_TOKENS_PER_SEC = 0.05
_MAX_TOKENS_PER_SEC = 30.0

# Multi-token-prediction drafters (DeepSeek MTP, etc.) amortise speculative
# acceptance over streamed bytes: each verified draft step advances several
# tokens per expert fetch. 1.8x is the conservative end of measured
# acceptance-length gains for 1-token MTP heads (method 6 of the design doc).
_MTP_SPEEDUP = 1.8

# Page-cache hits are never perfect even with a warm tier; keep a 5% miss
# floor so the estimator cannot claim infinite speed on a fully warm model.
_MISS_FRACTION_FLOOR = 0.05


@dataclass(frozen = True)
class GgufMoeMeta:
    """Parsed GGUF metadata relevant to MoE streaming.

    Field names follow the GGUF kv keys (``{arch}.expert_count`` etc.) so the
    metadata reader can fill them mechanically. Zeros mean "unknown" for the
    optional fields; the planner falls back to documented heuristics then.
    """

    arch: str  # e.g. "deepseek2", "llama"
    file_size_bytes: int  # total GGUF size incl. shards
    n_layers: int  # block_count
    n_moe_layers: int  # layers containing routed experts (0 for dense)
    n_experts: int  # {arch}.expert_count, 0 if dense/unknown
    n_experts_active: int  # {arch}.expert_used_count, 0 if unknown
    expert_ffn_length: int  # {arch}.feed_forward_expert_length, 0 if unknown
    embedding_length: int
    leading_dense_layers: int  # leading_dense_block_count
    has_mtp_drafter: bool = False  # multi-token-prediction drafter present


@dataclass(frozen = True)
class HardwareProfile:
    """Probed host capabilities. ``vram_available_bytes`` is already summed
    over the GPUs the user selected; ``nvme_read_mbps`` of 0 means "not
    probed" and falls back to ``DEFAULT_NVME_MBPS``."""

    ram_total_bytes: int
    ram_available_bytes: int
    vram_total_bytes: int  # 0 when CPU-only
    vram_available_bytes: int  # summed over selected GPUs
    nvme_read_mbps: float  # probed sequential read; 0 => unknown
    gpu_count: int = 0


@dataclass(frozen = True)
class StreamingTargets:
    """User-requested operating point. ``max_ram_bytes`` is a hard cap on the
    planned resident set; 0 defers to the planner default of 75% of available
    RAM so the OS, page cache and other apps keep headroom."""

    max_ram_bytes: int = 0  # hard cap for resident set; 0 => 75% of available
    context_tokens: int = 8192
    parallel_slots: int = 1
    # Optional {"<layer>.<expert>": weight} activation popularity profile,
    # e.g. measured from a calibration run. Drives which expert layers get
    # pinned / kept warm. None => uniform popularity assumed.
    expert_profile: Optional[Mapping[str, float]] = None


@dataclass(frozen = True)
class StreamingPlan:
    """Placement decision + modelled throughput. ``gpu_layers`` follows the
    llama.cpp convention (-1 = all/fit, >=0 = pinned count). When
    ``feasible`` is False, ``refusal_reason`` explains why and every other
    field is a neutral placeholder -- callers must check ``feasible`` first.
    """

    feasible: bool
    refusal_reason: Optional[str]
    streaming: bool  # True when in the NVMe streaming regime
    gpu_layers: int  # -1 = all/fit, >=0 pinned count
    n_cpu_moe: int  # value for --n-cpu-moe (MoE layers kept off GPU)
    override_tensor_rules: Tuple[str, ...]  # llama.cpp -ot rules (hot experts)
    fit: bool  # emit --fit on
    kv_cache_type_k: str  # e.g. "q8_0"
    kv_cache_type_v: str
    ctx_cap: int
    parallel_slots: int
    resident_ram_bytes: int  # non-expert + warm experts + KV + compute
    resident_vram_bytes: int
    stream_bytes_per_token: int  # estimated cold NVMe bytes per token
    est_tokens_per_sec: float  # modelled throughput
    notes: Tuple[str, ...]  # human-readable explanations


def _kv_reserve_bytes(meta: GgufMoeMeta, ctx: int, slots: int) -> int:
    """Conservative KV cache bytes for ``ctx`` tokens x ``slots`` at q8_0.

    Per token per layer we reserve ``2 * embedding_length * 4`` bytes: K and
    V, each sized as one full ``embedding_length`` vector, at 4 bytes per
    element. Real models use GQA/MLA with far fewer KV heads and q8_0 is 1
    byte per element, so this over-reserves several-fold -- deliberately: a
    planner that under-reserves KV OOMs llama-server hours into a session.
    """
    return int(ctx) * int(slots) * meta.n_layers * (2 * meta.embedding_length * 4)


def _expert_param_share(meta: GgufMoeMeta) -> float:
    """Estimated fraction of file bytes held by routed-expert tensors.

    Routed experts contribute ``n_moe_layers * n_experts * 3 *
    expert_ffn_length * embedding_length`` parameters (gate/up/down per
    expert per MoE layer). The non-expert remainder is approximated as
    ``n_layers * 4 * embedding_length**2`` (four square attention matrices
    per layer); the embedding table is ignored because GGUF metadata parsing
    here does not carry vocab size. Both sides of the ratio are quantised at
    roughly the same average bits-per-weight within one file, so the
    parameter share is a fair byte share -- slightly biased upward since the
    ignored embedding table sits on the non-expert side, which the 0.97 clamp
    absorbs. When the expert geometry is unknown we cannot do better than the
    flat 0.9 default, and the result is always clamped to [0.5, 0.97]: below
    0.5 the "MoE" label is not buying streamability, above 0.97 the shared
    path would be underestimated.
    """
    if (
        meta.n_experts <= 0
        or meta.n_moe_layers <= 0
        or meta.expert_ffn_length <= 0
        or meta.embedding_length <= 0
    ):
        return _EXPERT_SHARE_DEFAULT
    expert_params = (
        meta.n_moe_layers * meta.n_experts * 3 * meta.expert_ffn_length * meta.embedding_length
    )
    non_expert_params = meta.n_layers * 4 * meta.embedding_length ** 2
    share = expert_params / (expert_params + non_expert_params)
    return min(_EXPERT_SHARE_MAX, max(_EXPERT_SHARE_MIN, share))


def _refusal(reason: str, *, streaming: bool, targets: StreamingTargets) -> StreamingPlan:
    """Uniform infeasible plan: every knob at a neutral value so callers that
    forget to check ``feasible`` pass llama.cpp nothing dangerous."""
    return StreamingPlan(
        feasible = False,
        refusal_reason = reason,
        streaming = streaming,
        gpu_layers = 0,
        n_cpu_moe = 0,
        override_tensor_rules = (),
        fit = False,
        kv_cache_type_k = "q8_0",
        kv_cache_type_v = "q8_0",
        ctx_cap = targets.context_tokens,
        parallel_slots = max(1, targets.parallel_slots),
        resident_ram_bytes = 0,
        resident_vram_bytes = 0,
        stream_bytes_per_token = 0,
        est_tokens_per_sec = 0.0,
        notes = (reason,),
    )


def _aggregate_layer_profile(
    meta: GgufMoeMeta, profile: Optional[Mapping[str, float]]
) -> Dict[int, float]:
    """Collapse a ``{"<layer>.<expert>": weight}`` profile to per-layer mass.

    Malformed keys, out-of-range layers and non-positive weights are dropped
    rather than erroring: a profile is advisory telemetry, not a contract.
    Layers before ``leading_dense_layers`` are dropped too -- they have no
    routed experts, so an ``_exps`` rule for them would match nothing.
    """
    mass: Dict[int, float] = {}
    if not profile:
        return mass
    for key, weight in profile.items():
        try:
            layer_text, _expert_text = str(key).split(".", 1)
            layer = int(layer_text)
        except (ValueError, AttributeError):
            continue
        if layer < meta.leading_dense_layers or layer >= meta.n_layers:
            continue
        if not isinstance(weight, (int, float)) or weight <= 0:
            continue
        mass[layer] = mass.get(layer, 0.0) + float(weight)
    return mass


def plan_moe_streaming(
    meta: GgufMoeMeta, hw: HardwareProfile, targets: StreamingTargets
) -> StreamingPlan:
    """Decide how (or whether) to serve ``meta`` on ``hw`` given ``targets``.

    Pure function: no probing, no filesystem, no llama.cpp invocation -- the
    caller turns the returned plan into argv. All arithmetic is spelled out
    inline so the model's assumptions can be audited against MOE_STREAMING.md.
    """
    notes: List[str] = []
    ram_budget = targets.max_ram_bytes or int(0.75 * hw.ram_available_bytes)
    vram_budget = max(0, hw.vram_available_bytes)
    capacity = ram_budget + vram_budget
    is_moe = meta.n_experts > 0 and meta.n_moe_layers > 0

    # -- 1. Dense refusal -------------------------------------------------
    # A dense model touches every layer for every token, so streaming it from
    # NVMe means reading on the order of file_size bytes per token (the whole
    # model per forward pass; a context window of tokens still re-reads every
    # layer each step). MoE models read only the active experts. Refuse dense
    # over-capacity models outright instead of promising 0.001 tok/s.
    if not is_moe and meta.file_size_bytes > capacity:
        gib = meta.file_size_bytes / GIB
        return _refusal(
            f"dense model ({gib:.1f} GiB) exceeds resident capacity "
            f"({capacity / GIB:.1f} GiB) and cannot be streamed efficiently: "
            f"every layer is touched per token, so NVMe would have to serve "
            f"~the whole file per token",
            streaming = False,
            targets = targets,
        )

    # -- 2. Non-streaming regime ------------------------------------------
    # The whole file fits in RAM + VRAM: nothing to plan. Hand llama.cpp a
    # plain fit path (-1 layers, --fit on, no -ot rules) and report the
    # residency split for display purposes only.
    if meta.file_size_bytes <= capacity:
        resident_vram = min(meta.file_size_bytes, vram_budget)
        resident_ram = meta.file_size_bytes - resident_vram
        notes.append(
            f"model ({meta.file_size_bytes / GIB:.1f} GiB) fits resident capacity "
            f"({capacity / GIB:.1f} GiB): normal fit path, no streaming knobs"
        )
        notes.append(
            "throughput is memory-bandwidth bound, not NVMe bound; reported "
            f"at the {_MAX_TOKENS_PER_SEC:.0f} tok/s estimator ceiling"
        )
        return StreamingPlan(
            feasible = True,
            refusal_reason = None,
            streaming = False,
            gpu_layers = -1,
            n_cpu_moe = 0,
            override_tensor_rules = (),
            fit = True,
            kv_cache_type_k = "q8_0",
            kv_cache_type_v = "q8_0",
            ctx_cap = targets.context_tokens,
            parallel_slots = max(1, targets.parallel_slots),
            resident_ram_bytes = resident_ram,
            resident_vram_bytes = resident_vram,
            stream_bytes_per_token = 0,
            est_tokens_per_sec = _MAX_TOKENS_PER_SEC,
            notes = tuple(notes),
        )

    # -- 3. Streaming regime: expert/non-expert split ----------------------
    notes.append(
        f"streaming regime: file ({meta.file_size_bytes / GIB:.1f} GiB) exceeds "
        f"resident capacity ({capacity / GIB:.1f} GiB)"
    )
    share = _expert_param_share(meta)
    expert_bytes = int(meta.file_size_bytes * share)
    non_expert_bytes = meta.file_size_bytes - expert_bytes
    notes.append(
        f"expert share estimated at {share:.2f} of file bytes "
        f"(param-count heuristic, clamped to "
        f"[{_EXPERT_SHARE_MIN}, {_EXPERT_SHARE_MAX}]): "
        f"experts ~{expert_bytes / GIB:.1f} GiB, shared path "
        f"~{non_expert_bytes / GIB:.1f} GiB"
    )

    # The shared path (embeddings, attention, norms, shared experts) is read
    # on EVERY token; if it cannot be held resident, streaming cannot help.
    if non_expert_bytes > capacity:
        return _refusal(
            f"shared (non-expert) path alone needs ~{non_expert_bytes / GIB:.1f} GiB, "
            f"above the resident capacity of {capacity / GIB:.1f} GiB -- the "
            f"per-token hot path cannot be held resident",
            streaming = True,
            targets = targets,
        )

    # -- 4. Tier sizing: VRAM first ----------------------------------------
    # VRAM holds non-expert tensors up to 90% of available VRAM (the last 10%
    # covers llama.cpp's compute buffers, which it allocates on-device and
    # which are not visible to this planner). When VRAM comfortably holds the
    # whole shared path (10% headroom beyond the 90% fill), every layer's
    # non-expert part goes to GPU and ALL MoE expert layers stay CPU-side
    # (--n-cpu-moe = n_moe_layers): experts are what we stream, moving them
    # to VRAM wastes the fastest tier on the coldest bytes. Otherwise we pin
    # the proportional layer count llama.cpp would pick, and the MoE layers
    # that landed on GPU are subtracted from the --n-cpu-moe count.
    vram_tier_cap = int(0.9 * vram_budget)
    moe_layers_on_gpu = 0
    if vram_budget >= non_expert_bytes * 1.1:
        gpu_layers = meta.n_layers
        n_cpu_moe = meta.n_moe_layers
        vram_non_expert = min(non_expert_bytes, vram_tier_cap)
        notes.append(
            f"VRAM ({vram_budget / GIB:.1f} GiB) holds the whole shared path with "
            f"headroom: all {meta.n_layers} layers' attention on GPU, all "
            f"{meta.n_moe_layers} MoE layers CPU-side"
        )
    else:
        gpu_layers = max(0, int(meta.n_layers * vram_budget / max(meta.file_size_bytes, 1)))
        moe_layers_on_gpu = max(
            0, min(meta.n_moe_layers, gpu_layers - meta.leading_dense_layers)
        )
        n_cpu_moe = meta.n_moe_layers - moe_layers_on_gpu
        vram_non_expert = min(non_expert_bytes, vram_tier_cap)
        notes.append(
            f"VRAM ({vram_budget / GIB:.1f} GiB) covers "
            f"{vram_budget / meta.file_size_bytes:.1%} of the file: pinned "
            f"gpu_layers={gpu_layers}, {moe_layers_on_gpu} MoE layer(s) on GPU, "
            f"--n-cpu-moe={n_cpu_moe}"
        )
    non_expert_ram = non_expert_bytes - vram_non_expert

    # -- KV cache reservation + quant choice -------------------------------
    # Asymmetric quantisation (method 4): KV at q8_0 by default, q4_0 when
    # the q8_0 reservation would exceed a quarter of the RAM budget. Expert
    # weights at ~2-bit with shared tensors at Q4+ is a BUILD-TIME choice --
    # it is decided when the GGUF is produced/downloaded, so the planner can
    # only note it, not enforce it; KV cache quant is the runtime lever.
    kv_cap = int(_KV_RAM_CAP_FRACTION * ram_budget)
    ctx = max(1, targets.context_tokens)
    slots = max(1, targets.parallel_slots)

    def _kv_at(c: int, s: int) -> Tuple[int, bool]:
        raw = _kv_reserve_bytes(meta, c, s)
        q4 = raw > int(_KV_Q4_TRIGGER_FRACTION * ram_budget)
        return (int(raw * _KV_Q4_SCALE) if q4 else raw), q4

    kv_reserve, use_q4 = _kv_at(ctx, slots)
    if kv_reserve > kv_cap and slots > 1:
        # Parallel slots multiply KV linearly; dropping to one slot is the
        # cheapest way back under the cap before sacrificing context.
        slots = 1
        kv_reserve, use_q4 = _kv_at(ctx, slots)
        notes.append("KV over half the RAM budget: reduced parallel slots to 1")
    if kv_reserve > kv_cap:
        ctx = max(_MIN_CTX_TOKENS, int(ctx * (kv_cap / max(kv_reserve, 1))))
        kv_reserve, use_q4 = _kv_at(ctx, slots)
        notes.append(f"KV over half the RAM budget: capped context at {ctx} tokens")
    kv_type = "q4_0" if use_q4 else "q8_0"
    if use_q4:
        notes.append(
            f"KV cache downgraded to q4_0: q8_0 reservation would exceed "
            f"{_KV_Q4_TRIGGER_FRACTION:.0%} of the RAM budget"
        )
    notes.append(
        "expert-weight quant (~2-bit) vs shared-tensor quant (Q4+) is a "
        "build-time choice made when the GGUF is produced/downloaded; only "
        "KV quant is selectable at runtime"
    )

    if non_expert_ram + kv_reserve + _COMPUTE_RESERVE_BYTES > ram_budget:
        return _refusal(
            f"shared path in RAM ({non_expert_ram / GIB:.1f} GiB) + KV "
            f"({kv_reserve / GIB:.1f} GiB) + compute reserve "
            f"({_COMPUTE_RESERVE_BYTES / GIB:.1f} GiB) exceed the RAM budget "
            f"({ram_budget / GIB:.1f} GiB) even after the VRAM tier",
            streaming = True,
            targets = targets,
        )

    # -- RAM-warm tier ------------------------------------------------------
    # Whatever the RAM budget has left after the shared path, KV and compute
    # reserve can hold warm experts (page-cache resident). Cold experts are
    # also "CPU" to llama.cpp (the file is mmap'd), so RAM-warm vs NVMe-cold
    # is an OS page-cache / mlock distinction -- true runtime LRU expert
    # caching needs ik_llama.cpp and is wired separately.
    warm_budget = max(
        0, ram_budget - non_expert_ram - kv_reserve - _COMPUTE_RESERVE_BYTES
    )
    per_layer_expert_bytes = expert_bytes // max(meta.n_moe_layers, 1)
    warm_expert_bytes = min(warm_budget, expert_bytes)

    # -- Profile-guided pinning (method 3) ----------------------------------
    # Mainline llama.cpp cannot pin individual experts: -ot regexes match
    # tensor names and the `_exps` tensor holds ALL experts of a layer. The
    # useful lever is pinning WHOLE hot expert layers to the GPU when VRAM
    # headroom remains after the shared path. We rank MoE layers by summed
    # profile mass and pin the top-K that fit the headroom. When there is no
    # headroom (or no GPU, or no profile) we emit no rules at all -- a
    # "=CPU" rule would only restate llama.cpp's default for those tensors.
    layer_mass = _aggregate_layer_profile(meta, targets.expert_profile)
    headroom = vram_tier_cap - vram_non_expert - moe_layers_on_gpu * per_layer_expert_bytes
    pinned_layers: List[int] = []
    if layer_mass and hw.gpu_count > 0 and 0 < per_layer_expert_bytes <= headroom:
        remaining = headroom
        for layer, _mass in sorted(layer_mass.items(), key = lambda kv: (-kv[1], kv[0])):
            if per_layer_expert_bytes <= remaining:
                pinned_layers.append(layer)
                remaining -= per_layer_expert_bytes
    rules: Tuple[str, ...] = ()
    if pinned_layers:
        group = "|".join(str(layer) for layer in pinned_layers)
        rules = (f"blk\\.({group})\\.ffn_(gate|up|down)_exps\\.weight=CUDA0",)
        notes.append(
            f"pinned hottest expert layer(s) {pinned_layers} to CUDA0 via -ot "
            f"(per-expert pinning is impossible in mainline llama.cpp: the "
            f"_exps tensor holds all experts of a layer)"
        )
    elif layer_mass:
        notes.append(
            "expert profile supplied but no VRAM headroom remains after the "
            "shared path: no -ot pinning rules emitted"
        )

    # -- 7/8. Throughput model ----------------------------------------------
    # First-order model: each token activates n_experts_active experts, so the
    # cold bytes per token are the expert share scaled by the active fraction
    # and by the cache-miss fraction. Without a profile we model warmth as
    # the warm-budget fraction of expert bytes (uniform popularity); with a
    # profile we use the covered profile mass (GPU-pinned layers first, then
    # RAM-warm layers in popularity order), which is sharper when popularity
    # is skewed -- as it is in practice.
    if layer_mass:
        total_mass = sum(layer_mass.values())
        covered = sum(layer_mass[layer] for layer in pinned_layers)
        warm_left = warm_budget
        for layer, mass in sorted(layer_mass.items(), key = lambda kv: (-kv[1], kv[0])):
            if layer in pinned_layers:
                continue
            if 0 < per_layer_expert_bytes <= warm_left:
                covered += mass
                warm_left -= per_layer_expert_bytes
        miss_fraction = 1.0 - min(1.0 - _MISS_FRACTION_FLOOR, covered / max(total_mass, 1e-9))
    else:
        miss_fraction = 1.0 - min(
            1.0 - _MISS_FRACTION_FLOOR, warm_budget / max(expert_bytes, 1)
        )
    notes.append(f"modelled cold-miss fraction: {miss_fraction:.3f}")

    # Unknown active-expert count => assume all experts active (worst case;
    # keeps the estimate conservative for archs without expert_used_count).
    active = meta.n_experts_active if meta.n_experts_active > 0 else meta.n_experts
    active = min(active, meta.n_experts)
    stream_bytes = int(expert_bytes * (active / meta.n_experts) * miss_fraction)
    nvme_mbps = hw.nvme_read_mbps if hw.nvme_read_mbps > 0 else DEFAULT_NVME_MBPS
    if hw.nvme_read_mbps <= 0:
        notes.append(f"NVMe not probed; assuming {DEFAULT_NVME_MBPS:.0f} MB/s")
    est = min(nvme_mbps * 1e6 / max(stream_bytes, 1), _MAX_TOKENS_PER_SEC)
    if meta.has_mtp_drafter:
        # MTP drafter amortises speculative acceptance over streamed bytes
        # (method 6): one expert fetch advances several accepted tokens.
        est *= _MTP_SPEEDUP
        notes.append(f"MTP drafter present: throughput x{_MTP_SPEEDUP}")
    if est < _MIN_TOKENS_PER_SEC:
        return _refusal(
            f"modelled throughput {est:.3f} tok/s is below the "
            f"{_MIN_TOKENS_PER_SEC} tok/s floor: at {nvme_mbps:.0f} MB/s NVMe "
            f"and ~{stream_bytes / 1e6:.0f} MB of cold expert reads per token, "
            f"serving this model would look hung",
            streaming = True,
            targets = targets,
        )
    notes.append(
        f"~{stream_bytes / 1e6:.0f} MB cold per token at {nvme_mbps:.0f} MB/s "
        f"=> ~{est:.2f} tok/s"
    )

    # -- 9. fit flag ---------------------------------------------------------
    # Consistent rule: the planner pins placement explicitly whenever it ran
    # tier math (streaming regime), because llama.cpp's --fit cannot reason
    # about NVMe streaming, --n-cpu-moe or -ot expert pinning -- so streaming
    # plans always emit fit=False and --fit stays delegated to llama.cpp only
    # on the non-streaming pass-through above.
    resident_ram = non_expert_ram + warm_expert_bytes + kv_reserve + _COMPUTE_RESERVE_BYTES
    resident_vram = (
        vram_non_expert + (moe_layers_on_gpu + len(pinned_layers)) * per_layer_expert_bytes
    )
    return StreamingPlan(
        feasible = True,
        refusal_reason = None,
        streaming = True,
        gpu_layers = gpu_layers,
        n_cpu_moe = n_cpu_moe,
        override_tensor_rules = rules,
        fit = False,
        kv_cache_type_k = kv_type,
        kv_cache_type_v = kv_type,
        ctx_cap = ctx,
        parallel_slots = slots,
        resident_ram_bytes = resident_ram,
        resident_vram_bytes = resident_vram,
        stream_bytes_per_token = stream_bytes,
        est_tokens_per_sec = est,
        notes = tuple(notes),
    )
