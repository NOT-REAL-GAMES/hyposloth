# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the MoE streaming offload planner.

The planner decides whether a GGUF MoE model far larger than RAM + VRAM can
be served by streaming cold experts from NVMe, and emits the llama.cpp
placement knobs. These tests pin the tier math, the refusal paths and the
profile-guided -ot pinning so the wiring in llama_cpp.py can rely on them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load moe_streaming_planner.py directly to avoid dragging in the full backend
# chain via core/inference/__init__.py. The planner is dependency-free. The
# module must be registered in sys.modules before exec_module: dataclasses
# resolves string annotations against the module's namespace there.
_MSP_PATH = (
    Path(__file__).resolve().parent.parent / "core" / "inference" / "moe_streaming_planner.py"
)
_spec = importlib.util.spec_from_file_location("_msp_test_only", _MSP_PATH)
_msp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _msp
_spec.loader.exec_module(_msp)
GgufMoeMeta = _msp.GgufMoeMeta
HardwareProfile = _msp.HardwareProfile
StreamingTargets = _msp.StreamingTargets
StreamingPlan = _msp.StreamingPlan
plan_moe_streaming = _msp.plan_moe_streaming
GIB = _msp.GIB
DEFAULT_NVME_MBPS = _msp.DEFAULT_NVME_MBPS


# DeepSeek-V3-ish geometry: the canonical ~1.5 TB-class MoE GGUF. 61 blocks,
# 3 leading dense layers, 58 MoE layers x 256 routed experts with 8 active.
def _deepseek_meta(**over):
    kw = dict(
        arch = "deepseek2",
        file_size_bytes = int(1.5e12),
        n_layers = 61,
        n_moe_layers = 58,
        n_experts = 256,
        n_experts_active = 8,
        expert_ffn_length = 2048,
        embedding_length = 7168,
        leading_dense_layers = 3,
        has_mtp_drafter = False,
    )
    kw.update(over)
    return GgufMoeMeta(**kw)


def _hw(**over):
    kw = dict(
        ram_total_bytes = 128 * GIB,
        ram_available_bytes = 110 * GIB,
        vram_total_bytes = 24 * GIB,
        vram_available_bytes = 24 * GIB,
        nvme_read_mbps = 7000.0,
        gpu_count = 1,
    )
    kw.update(over)
    return HardwareProfile(**kw)


# ── Dense refusal ─────────────────────────────────────────────────────────


def test_dense_model_over_capacity_refused():
    meta = GgufMoeMeta(
        arch = "llama",
        file_size_bytes = int(200e9),
        n_layers = 80,
        n_moe_layers = 0,
        n_experts = 0,
        n_experts_active = 0,
        expert_ffn_length = 0,
        embedding_length = 8192,
        leading_dense_layers = 0,
    )
    hw = _hw(ram_available_bytes = 8 * GIB, vram_total_bytes = 0, vram_available_bytes = 0)
    plan = plan_moe_streaming(meta, hw, StreamingTargets())
    assert plan.feasible is False
    assert plan.streaming is False
    assert "dense" in plan.refusal_reason.lower()


def test_dense_model_that_fits_uses_normal_path():
    meta = GgufMoeMeta(
        arch = "llama",
        file_size_bytes = int(4e9),
        n_layers = 32,
        n_moe_layers = 0,
        n_experts = 0,
        n_experts_active = 0,
        expert_ffn_length = 0,
        embedding_length = 4096,
        leading_dense_layers = 0,
    )
    plan = plan_moe_streaming(meta, _hw(), StreamingTargets())
    assert plan.feasible is True
    assert plan.streaming is False


# ── Non-streaming (fits resident capacity) ────────────────────────────────


def test_small_moe_model_gets_passthrough_plan():
    meta = _deepseek_meta(
        file_size_bytes = int(4e9), n_layers = 32, n_moe_layers = 4, n_experts = 8,
        n_experts_active = 2, embedding_length = 4096, leading_dense_layers = 0,
    )
    hw = _hw(ram_available_bytes = 64 * GIB)
    plan = plan_moe_streaming(meta, hw, StreamingTargets())
    assert plan.feasible is True
    assert plan.streaming is False
    assert plan.gpu_layers == -1
    assert plan.n_cpu_moe == 0
    assert plan.override_tensor_rules == ()
    assert plan.fit is True
    assert plan.stream_bytes_per_token == 0
    assert plan.est_tokens_per_sec == 30.0


# ── Streaming tier math on the synthetic 1.5 TB model ─────────────────────


def test_streaming_tier_math_deepseekish():
    plan = plan_moe_streaming(_deepseek_meta(), _hw(), StreamingTargets())
    assert plan.feasible is True
    assert plan.streaming is True
    ram_budget = int(0.75 * 110 * GIB)
    assert plan.resident_ram_bytes <= ram_budget
    assert plan.n_cpu_moe > 0
    assert 0.1 <= plan.est_tokens_per_sec <= 30.0
    # Streaming plans pin placement explicitly; --fit stays off.
    assert plan.fit is False
    assert plan.gpu_layers >= 0
    # The conservative KV estimate at the default 8k ctx exceeds 25% of the
    # RAM budget on this geometry, so the planner downgrades KV to q4_0.
    assert plan.kv_cache_type_k == "q4_0"
    assert plan.kv_cache_type_v == "q4_0"
    assert plan.notes  # human-readable explanations present


def test_streaming_mtp_drafter_boosts_estimate():
    base = plan_moe_streaming(_deepseek_meta(), _hw(), StreamingTargets())
    mtp = plan_moe_streaming(_deepseek_meta(has_mtp_drafter = True), _hw(), StreamingTargets())
    assert base.feasible and mtp.feasible
    assert mtp.est_tokens_per_sec == pytest.approx(base.est_tokens_per_sec * 1.8)


def test_shared_path_over_capacity_refused():
    # Same model but a tiny machine: even the non-expert hot path cannot be
    # held resident, so streaming cannot help.
    hw = _hw(ram_available_bytes = 8 * GIB, vram_total_bytes = 0, vram_available_bytes = 0)
    plan = plan_moe_streaming(_deepseek_meta(), hw, StreamingTargets())
    assert plan.feasible is False
    assert plan.streaming is True
    assert "shared" in plan.refusal_reason.lower()


# ── Profile-guided -ot pinning ────────────────────────────────────────────

_PROFILE = {"5.0": 90.0, "5.3": 80.0, "17.1": 70.0, "42.0": 10.0}


def test_profile_pinning_emits_rules_when_vram_headroom():
    # 96 GiB VRAM holds the whole ~45 GB shared path with headroom to spare;
    # one ~25 GB expert layer then fits the headroom, and layer 5 carries the
    # most profile mass (90 + 80).
    hw = _hw(vram_total_bytes = 96 * GIB, vram_available_bytes = 96 * GIB)
    plan = plan_moe_streaming(
        _deepseek_meta(), hw, StreamingTargets(expert_profile = _PROFILE)
    )
    assert plan.feasible is True
    assert plan.gpu_layers == 61
    assert plan.n_cpu_moe == 58
    assert plan.override_tensor_rules == (
        "blk\\.(5)\\.ffn_(gate|up|down)_exps\\.weight=CUDA0",
    )
    joined = " ".join(plan.override_tensor_rules)
    assert "17" not in joined
    assert "42" not in joined


def test_profile_without_vram_headroom_emits_no_rules():
    # 24 GiB VRAM is fully consumed by the shared path: nothing left to pin
    # expert layers into, so no rules (a =CPU rule would restate the default).
    plan = plan_moe_streaming(
        _deepseek_meta(), _hw(), StreamingTargets(expert_profile = _PROFILE)
    )
    assert plan.feasible is True
    assert plan.override_tensor_rules == ()


# ── Refusals and KV quant policy ──────────────────────────────────────────


def test_hopeless_nvme_refused():
    hw = _hw(nvme_read_mbps = 50.0)
    plan = plan_moe_streaming(_deepseek_meta(), hw, StreamingTargets())
    assert plan.feasible is False
    assert plan.streaming is True
    assert "tok/s" in plan.refusal_reason


def test_huge_context_downgrades_kv_and_caps_ctx():
    plan = plan_moe_streaming(
        _deepseek_meta(), _hw(), StreamingTargets(context_tokens = 1_000_000)
    )
    assert plan.feasible is True
    assert plan.kv_cache_type_k == "q4_0"
    assert plan.kv_cache_type_v == "q4_0"
    assert 2048 <= plan.ctx_cap < 1_000_000
    assert plan.parallel_slots == 1


def test_small_context_keeps_q8_kv():
    plan = plan_moe_streaming(
        _deepseek_meta(), _hw(), StreamingTargets(context_tokens = 2048)
    )
    assert plan.feasible is True
    assert plan.kv_cache_type_k == "q8_0"
    assert plan.kv_cache_type_v == "q8_0"
    assert plan.ctx_cap == 2048


def test_unprobed_nvme_uses_default_and_reports_it():
    plan = plan_moe_streaming(_deepseek_meta(), _hw(nvme_read_mbps = 0.0), StreamingTargets())
    assert plan.feasible is True
    assert any(f"{DEFAULT_NVME_MBPS:.0f} MB/s" in note for note in plan.notes)
