# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the MoE-streaming wiring between the planner and llama_cpp.

Covers the GGUF metadata -> planner-meta mapping (expert_used_count /
expert_feed_forward_length), the _apply_streaming_plan argv shape, the
/status streaming_plan contract dict, the infeasible refusal message, the
gates in _plan_moe_streaming_for_load (dense / fits / unknown RAM / probe
failure), and the ik_llama.cpp flavor gating. Everything runs on bare
LlamaCppBackend instances with the RAM/disk probes monkeypatched -- no
subprocess, GPU or network.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import utils.hardware.disk_speed as disk_speed
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.moe_streaming_planner import StreamingPlan

GIB = 1024 ** 3


def _gguf_blob(kvs) -> bytes:
    """Minimal GGUF header with the scalar and bool-array KVs used below."""
    out = [struct.pack("<II", 0x46554747, 3), struct.pack("<QQ", 0, len(kvs))]
    for key, value in kvs:
        encoded = key.encode("utf-8")
        out.append(struct.pack("<Q", len(encoded)))
        out.append(encoded)
        if isinstance(value, str):
            text = value.encode("utf-8")
            out.append(struct.pack("<I", 8))  # STRING
            out.append(struct.pack("<Q", len(text)))
            out.append(text)
        elif isinstance(value, list):
            out.append(struct.pack("<IIQ", 9, 7, len(value)))  # ARRAY of BOOL
            out.append(bytes(bool(item) for item in value))
        else:
            out.append(struct.pack("<I", 4))  # UINT32
            out.append(struct.pack("<I", int(value)))
    return b"".join(out)


def _backend(**attrs) -> LlamaCppBackend:
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._model_identifier = "test-model"
    for name, value in attrs.items():
        setattr(b, name, value)
    return b


def _moe_backend() -> LlamaCppBackend:
    """DeepSeek-V3-ish geometry for _plan_moe_streaming_for_load."""
    return _backend(
        _architecture = "deepseek2",
        _n_layers = 61,
        _n_experts = 256,
        _leading_dense_block_count = 3,
        _n_experts_active = 8,
        _expert_ffn_length = 2048,
        _embedding_length = 7168,
        _nextn_predict_layers = None,
        _context_length = 4096,
    )


def _plan(**over) -> StreamingPlan:
    kw = dict(
        feasible = True,
        refusal_reason = None,
        streaming = True,
        gpu_layers = 20,
        n_cpu_moe = 41,
        override_tensor_rules = (r"blk\.(3|7)\.ffn_(gate|up|down)_exps\.weight=CUDA0",),
        fit = False,
        kv_cache_type_k = "q8_0",
        kv_cache_type_v = "q8_0",
        ctx_cap = 4096,
        parallel_slots = 1,
        resident_ram_bytes = 100 * GIB,
        resident_vram_bytes = 20 * GIB,
        stream_bytes_per_token = 40_000_000,
        est_tokens_per_sec = 1.5,
        notes = ("note one", "note two"),
    )
    kw.update(over)
    return StreamingPlan(**kw)


# ── GGUF metadata -> planner meta mapping ──────────────────────────────


def test_gguf_metadata_reads_moe_streaming_keys(tmp_path):
    path = tmp_path / "moe.gguf"
    path.write_bytes(
        _gguf_blob(
            [
                ("general.architecture", "deepseek2"),
                ("deepseek2.block_count", 61),
                ("deepseek2.leading_dense_block_count", 3),
                ("deepseek2.expert_count", 256),
                ("deepseek2.expert_used_count", 8),
                ("deepseek2.expert_feed_forward_length", 2048),
                ("deepseek2.embedding_length", 7168),
                ("deepseek2.nextn_predict_layers", 1),
            ]
        )
    )
    b = _backend()
    b._read_gguf_metadata(str(path))
    assert b._architecture == "deepseek2"
    assert b._n_layers == 61
    assert b._n_experts == 256
    assert b._n_experts_active == 8
    assert b._expert_ffn_length == 2048
    assert b._embedding_length == 7168
    assert b.n_moe_layers == 58
    # Drafter presence reuses the existing MTP (nextn) detection.
    assert bool(b._nextn_predict_layers) is True


def test_gguf_metadata_missing_moe_keys_default_zero(tmp_path):
    path = tmp_path / "dense.gguf"
    path.write_bytes(
        _gguf_blob(
            [
                ("general.architecture", "llama"),
                ("llama.block_count", 32),
            ]
        )
    )
    b = _backend()
    b._read_gguf_metadata(str(path))
    assert b._n_experts_active == 0
    assert b._expert_ffn_length == 0
    assert b.n_moe_layers == 0


def test_k3_virtual_kv_cpu_reservation_uses_only_mla_layers():
    b = _backend(
        _architecture = "kimi-k3",
        _n_layers = 93,
        _n_kv_heads = 1,
        _n_kv_heads_by_layer = [0] * 69 + [1] * 24,
        _kv_lora_rank = 512,
        _key_length_mla = 64,
        _kv_key_length = 576,
        _indexer_kpool = None,
    )
    per_token, fixed, cache_type = b._virtual_kv_cpu_reservation(1024 * 1024, "q8_0")
    # q8_0 stores 32 values in 34 bytes: 576 values = 612 bytes/layer.
    assert per_token == 24 * 612 + 16
    assert fixed == 4096 * 16 + 256 * 612
    assert cache_type == "q8_0"


def test_generic_virtual_kv_cpu_reservation_uses_actual_gqa_kv_rows():
    b = _backend(
        _architecture = "llama",
        _n_layers = 32,
        _n_kv_heads = 8,
        _n_kv_heads_by_layer = None,
        _n_heads = 32,
        _embedding_length = 4096,
        _kv_lora_rank = None,
        _kv_key_length = 128,
        _kv_value_length = 128,
        _shared_kv_layers = None,
    )
    per_token, fixed, cache_type = b._virtual_kv_cpu_reservation(1024 * 1024, "q4_0")
    # q4_0 row(8 * 128) = 576 bytes; ordinary attention stores K and V.
    assert per_token == 32 * (576 + 576) + 16
    assert fixed == 4096 * 16 + 256 * (576 + 576)
    assert cache_type == "q4_0"


def test_gguf_metadata_reads_glm_virtual_kv_indexer_geometry(tmp_path):
    path = tmp_path / "glm.gguf"
    path.write_bytes(
        _gguf_blob(
            [
                ("general.architecture", "glm5next"),
                ("glm5next.block_count", 78),
                ("glm5next.attention.indexer.key_length", 128),
                ("glm5next.attention.indexer.kpool", 4),
                ("glm5next.attention.indexer.types", [True, False, True]),
            ]
        )
    )
    b = _backend()
    b._read_gguf_metadata(str(path))
    assert b._indexer_key_length == 128
    assert b._indexer_kpool == 4
    assert b._indexer_types == [True, False, True]


def test_glm_virtual_kv_descriptor_context_caps_match_cuda_planner():
    glm53 = _backend(
        _architecture = "glm-dsa",
        _n_layers = 78,
        _context_length = 1024 * 1024,
        _indexer_key_length = 128,
        _indexer_kpool = None,
        _indexer_types = None,
    )
    # 21 native full indexers, Q8_0 row(128) = 136 bytes, 512 MiB cap.
    assert glm53._virtual_kv_descriptor_context_cap() == 187904

    glm_next = _backend(
        _architecture = "glm5next",
        _n_layers = 78,
        _context_length = 1024 * 1024,
        _indexer_key_length = 128,
        _indexer_kpool = 4,
        _indexer_types = [True] * 78,
    )
    assert glm_next._virtual_kv_descriptor_context_cap() == 202240


def test_glm_dsa_virtual_kv_reservation_uses_native_token_indexer():
    b = _backend(
        _architecture = "glm-dsa",
        _n_layers = 78,
        _n_kv_heads = 1,
        _n_kv_heads_by_layer = None,
        _kv_lora_rank = 512,
        _key_length_mla = 64,
        _kv_key_length = 576,
        _indexer_kpool = None,
    )
    per_token, fixed, cache_type = b._virtual_kv_cpu_reservation(32768, "q4_0")
    # q4_0 stores 32 values in 18 bytes: 576 values = 324 bytes/layer.
    assert per_token == 78 * 324 + 16
    assert fixed == 32768 * 16 + 324
    assert cache_type == "q4_0"


def test_virtual_kv_preflight_requires_experimental_and_accepts_compatible_architectures():
    with pytest.raises(ValueError, match = "Experimental"):
        LlamaCppBackend._validate_virtual_kv_request("glm-dsa", False)
    for architecture in ("glm-dsa", "glm5next", "kimi-k3", "llama", "qwen2", "qwen3"):
        LlamaCppBackend._validate_virtual_kv_request(architecture, True)
    with pytest.raises(ValueError, match = "not structurally validated"):
        LlamaCppBackend._validate_virtual_kv_request("gpt2", True)


# ── _apply_streaming_plan argv ─────────────────────────────────────────


def test_apply_streaming_plan_emits_placement_flags():
    cmd = ["llama-server", "-m", "model.gguf"]
    LlamaCppBackend._apply_streaming_plan(cmd, _plan(), n_moe_layers = 58, leading_dense = 3)
    text = " ".join(cmd)
    assert "--gpu-layers 20" in text
    assert "--fit off" in text
    # --n-cpu-moe counts from layer 0: 3 leading dense + 41 MoE layers.
    assert "--n-cpu-moe 44" in text
    assert r"-ot blk\.(3|7)\.ffn_(gate|up|down)_exps\.weight=CUDA0" in text
    assert "--cache-type-k q8_0" in text
    assert "--cache-type-v q8_0" in text


def test_apply_streaming_plan_ik_rtr_replaces_ot_rules():
    cmd = []
    LlamaCppBackend._apply_streaming_plan(
        cmd, _plan(), n_moe_layers = 58, leading_dense = 3, ik_rtr = True
    )
    assert "-rtr" in cmd
    assert "-ot" not in cmd
    # Placement and KV types are still emitted under the ik runtime cache.
    assert "--n-cpu-moe" in cmd
    assert "--cache-type-k" in cmd


def test_apply_streaming_plan_without_rules_emits_no_ot():
    cmd = []
    LlamaCppBackend._apply_streaming_plan(
        cmd, _plan(override_tensor_rules = ()), n_moe_layers = 58, leading_dense = 3
    )
    assert "-ot" not in cmd
    assert "-rtr" not in cmd


def test_apply_streaming_plan_cpu_only_skips_moe_flag():
    cmd = []
    LlamaCppBackend._apply_streaming_plan(
        cmd, _plan(gpu_layers = 0, n_cpu_moe = 0), n_moe_layers = 58, leading_dense = 3
    )
    assert "--gpu-layers 0" in " ".join(cmd)
    assert "--n-cpu-moe" not in cmd


# ── /status contract dict + refusal message ────────────────────────────


def test_streaming_plan_status_dict_shape():
    d = LlamaCppBackend._streaming_plan_status_dict(_plan())
    assert d == {
        "streaming": True,
        "resident_ram_gb": 100.0,
        "resident_vram_gb": 20.0,
        "est_tokens_per_sec": 1.5,
        "n_cpu_moe": 41,
        "notes": ["note one", "note two"],
    }


def test_streaming_plan_status_dict_rounds_gb_to_one_decimal():
    # 100.25 GiB is exact in binary; round-half-even gives 100.2.
    d = LlamaCppBackend._streaming_plan_status_dict(
        _plan(resident_ram_bytes = 100 * GIB + GIB // 4)
    )
    assert d["resident_ram_gb"] == 100.2


def test_streaming_refusal_message_includes_reason():
    plan = _plan(
        feasible = False,
        refusal_reason = "modelled throughput 0.010 tok/s is below the 0.05 tok/s floor",
    )
    msg = LlamaCppBackend._streaming_refusal_message(plan)
    assert "modelled throughput 0.010 tok/s is below the 0.05 tok/s floor" in msg
    assert "GGUF" in msg


# ── _plan_moe_streaming_for_load gates ─────────────────────────────────


def _call_plan(b, model_path, gguf_size, **kw):
    args = dict(
        gpus = [],
        total_by_idx = {},
        effective_ctx = 4096,
        n_parallel = 1,
        mtp_draft_path = None,
    )
    args.update(kw)
    return b._plan_moe_streaming_for_load(
        model_path = str(model_path), gguf_size = gguf_size, **args
    )


def test_plan_for_load_skips_dense_models(tmp_path):
    b = _moe_backend()
    b._n_experts = None
    # The dense gate fires before any file/disk access: a missing path is fine.
    assert _call_plan(b, tmp_path / "nope.gguf", 10 ** 15) is None


def test_plan_for_load_skips_when_ram_unknown(tmp_path, monkeypatch):
    b = _moe_backend()
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: None)
    )
    assert _call_plan(b, tmp_path / "nope.gguf", 10 ** 15) is None


def test_plan_for_load_skips_models_that_fit(tmp_path, monkeypatch):
    b = _moe_backend()
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 512 * 1024)
    )
    model = tmp_path / "small.gguf"
    model.write_bytes(b"x" * 1024)
    # 1 KiB fits the 384 GiB RAM budget easily: no streaming, no disk probe.
    assert _call_plan(b, model, 1024) is None


def test_plan_for_load_streams_oversize_moe(tmp_path, monkeypatch):
    b = _moe_backend()
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 110 * 1024)
    )
    monkeypatch.setattr(disk_speed, "probe_sequential_read_mbps", lambda path: 7000.0)
    plan = _call_plan(
        b,
        tmp_path / "huge.gguf",
        int(1.5e12),
        gpus = [(0, 24 * 1024)],
        total_by_idx = {0: 24 * 1024},
    )
    assert plan is not None
    assert plan.streaming is True
    assert plan.feasible is True
    # 24 GiB VRAM cannot hold the shared path: all 58 MoE layers stay CPU-side.
    assert plan.n_cpu_moe == 58
    assert plan.gpu_layers >= 0


def test_plan_for_load_survives_disk_probe_failure(tmp_path, monkeypatch):
    b = _moe_backend()
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 110 * 1024)
    )

    def _boom(path):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(disk_speed, "probe_sequential_read_mbps", _boom)
    plan = _call_plan(b, tmp_path / "huge.gguf", int(1.5e12))
    # A probe failure degrades to the planner's default NVMe rate, never a crash.
    assert plan is not None
    assert plan.streaming is True


# ── ik_llama.cpp flavor gating ─────────────────────────────────────────


def _ik_name() -> str:
    return "llama-server-ik.exe" if sys.platform == "win32" else "llama-server-ik"


def test_ik_flavor_requires_env_and_binary(tmp_path, monkeypatch):
    mainline = tmp_path / "llama-server"
    mainline.write_text("stub")
    monkeypatch.delenv("UNSLOTH_STUDIO_LLAMA_FLAVOR", raising = False)
    binary, active = LlamaCppBackend._resolve_llama_flavor(str(mainline))
    assert binary == str(mainline) and active is False

    monkeypatch.setenv("UNSLOTH_STUDIO_LLAMA_FLAVOR", "ik")
    # Env var alone: no sibling binary -> mainline, inactive.
    binary, active = LlamaCppBackend._resolve_llama_flavor(str(mainline))
    assert binary == str(mainline) and active is False

    sibling = tmp_path / _ik_name()
    sibling.write_text("stub")
    binary, active = LlamaCppBackend._resolve_llama_flavor(str(mainline))
    assert binary == str(sibling) and active is True


def test_ik_flavor_missing_binary_stays_inert(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_LLAMA_FLAVOR", "ik")
    assert LlamaCppBackend._resolve_llama_flavor(None) == (None, False)


def test_probe_reports_supports_rtr(tmp_path, monkeypatch):
    fake = tmp_path / _ik_name()
    fake.write_text("stub")
    help_text = "  -rtr, --repurpose-tensor-cache  enable runtime tensor cache\n"
    monkeypatch.setattr(
        "core.inference.llama_cpp.child_env_without_native_path_secret", lambda: {}
    )
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda cmd, **kw: _types.SimpleNamespace(
            stdout = help_text, stderr = "", returncode = 0
        ),
    )
    LlamaCppBackend._capability_cache.clear()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["supports_rtr"] is True


def test_probe_without_rtr_reports_no_support(tmp_path, monkeypatch):
    fake = tmp_path / "llama-server"
    fake.write_text("stub")
    help_text = "  --fit-ctx N  fit context\n  --metrics  prometheus\n"
    monkeypatch.setattr(
        "core.inference.llama_cpp.child_env_without_native_path_secret", lambda: {}
    )
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda cmd, **kw: _types.SimpleNamespace(
            stdout = help_text, stderr = "", returncode = 0
        ),
    )
    LlamaCppBackend._capability_cache.clear()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["supports_rtr"] is False


def test_probe_missing_binary_supports_rtr_false():
    LlamaCppBackend._capability_cache.clear()
    caps = LlamaCppBackend.probe_server_capabilities("definitely/not/a/binary")
    assert caps["supports_rtr"] is False


# ── plan state lifecycle ───────────────────────────────────────────────


def test_streaming_plan_property_and_unload_reset():
    b = LlamaCppBackend()
    assert b.streaming_plan is None
    b._streaming_plan = LlamaCppBackend._streaming_plan_status_dict(_plan())
    assert b.streaming_plan["streaming"] is True
    assert b.streaming_plan["n_cpu_moe"] == 41
    b.unload_model()
    assert b.streaming_plan is None
    assert b._n_experts_active == 0
    assert b._expert_ffn_length == 0


# ── full load_model integration (fake Popen; no real server) ───────────

_REAL_POPEN = subprocess.Popen


def _integration_backend(tmp_path, memory):
    """A LlamaCppBackend mocked like test_llama_cpp_placement's harness:
    load_model runs end to end, only the spawn is faked."""
    backend = LlamaCppBackend()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(
        _gguf_blob(
            [
                ("general.architecture", "deepseek2"),
                ("deepseek2.block_count", 61),
                ("deepseek2.leading_dense_block_count", 3),
                ("deepseek2.expert_count", 256),
                ("deepseek2.expert_used_count", 8),
                ("deepseek2.expert_feed_forward_length", 2048),
                ("deepseek2.embedding_length", 7168),
            ]
        )
    )
    backend._get_gpu_memory = lambda _binary = None: list(memory)
    backend._get_gpu_free_memory = lambda _binary = None: [
        (index, free) for index, free, _total in memory
    ]
    backend._can_estimate_kv = lambda: False
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *args, **kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    return backend, gguf


def _flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_load_model_streams_oversize_moe(tmp_path, monkeypatch):
    backend, gguf = _integration_backend(tmp_path, [(0, 24 * 1024, 24 * 1024)])
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._get_gguf_size_bytes = lambda _path: int(1.5e12)
    backend._available_system_memory_mib = lambda: 110 * 1024
    # The harness GGUF blob carries the MoE keys, so the real
    # _read_gguf_metadata fills the planner inputs from the file.
    monkeypatch.setattr(disk_speed, "probe_sequential_read_mbps", lambda path: 7000.0)

    captured = {}
    expected_gguf = str(gguf)

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(gguf_path = expected_gguf, model_identifier = "test")

    cmd = captured["cmd"]
    assert "--fit" in cmd and _flag_value(cmd, "--fit") == "off"
    # 24 GiB VRAM holds ~1.7% of the file: planner pins 1 layer on GPU and
    # keeps all 58 MoE layers CPU-side (3 leading dense + 58 = 61).
    assert _flag_value(cmd, "--gpu-layers") == "1"
    assert _flag_value(cmd, "--n-cpu-moe") == "61"
    assert _flag_value(cmd, "--cache-type-k") == "q8_0"
    assert _flag_value(cmd, "--cache-type-v") == "q8_0"
    # ctx/slot caps from the plan (4096 requested, plan cap 4096; 1 slot).
    assert _flag_value(cmd, "-c") == "4096"
    assert _flag_value(cmd, "--parallel") == "1"
    # No expert profile: no static -ot pinning rules.
    assert "-ot" not in cmd
    assert "-rtr" not in cmd
    assert backend.streaming_plan is not None
    assert backend.streaming_plan["streaming"] is True
    assert backend.streaming_plan["n_cpu_moe"] == 58
    assert backend._n_cpu_moe == 58
    assert backend._gpu_layers == 1


def test_load_model_normal_model_unchanged(tmp_path, monkeypatch):
    backend, gguf = _integration_backend(tmp_path, [(0, 10_000, 16_000)])
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._get_gguf_size_bytes = lambda _path: 1024

    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(gguf_path = str(gguf), model_identifier = "test")

    cmd = captured["cmd"]
    # A model that fits never touches the streaming knobs.
    assert "--n-cpu-moe" not in cmd
    assert "-ot" not in cmd
    assert "-rtr" not in cmd
    assert "--cache-type-k" not in cmd
    assert backend.streaming_plan is None
    assert backend._n_cpu_moe == 0


def test_load_model_infeasible_streaming_raises(tmp_path):
    backend, gguf = _integration_backend(tmp_path, [(0, 24 * 1024, 24 * 1024)])
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._get_gguf_size_bytes = lambda _path: int(1.5e12)
    backend._plan_moe_streaming_for_load = lambda **kw: _plan(
        feasible = False, refusal_reason = "modelled throughput 0.010 tok/s is below the floor"
    )

    with pytest.raises(RuntimeError, match = "modelled throughput 0.010 tok/s"):
        backend.load_model(gguf_path = str(gguf), model_identifier = "test")
    assert backend.streaming_plan is None


def _ik_pair(tmp_path):
    mainline = tmp_path / "llama-server"
    mainline.write_text("stub")
    sibling = tmp_path / _ik_name()
    sibling.write_text("stub")
    return mainline, sibling


def test_load_model_ik_flavor_replaces_ot_with_rtr(tmp_path, monkeypatch):
    backend, gguf = _integration_backend(tmp_path, [(0, 24 * 1024, 24 * 1024)])
    mainline, sibling = _ik_pair(tmp_path)
    backend._find_llama_server_binary = lambda include_denied = False: str(mainline)
    backend._get_gguf_size_bytes = lambda _path: int(1.5e12)
    backend._plan_moe_streaming_for_load = lambda **kw: _plan()
    backend.probe_server_capabilities = lambda binary = None: {
        "found": True,
        "supports_kv_unified": False,
        "supports_rtr": True,
    }
    monkeypatch.setenv("UNSLOTH_STUDIO_LLAMA_FLAVOR", "ik")

    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != str(sibling):
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(gguf_path = str(gguf), model_identifier = "test")

    cmd = captured["cmd"]
    # The ik sibling binary is what actually launches.
    assert cmd[0] == str(sibling)
    assert "-rtr" in cmd
    assert "-ot" not in cmd
    assert _flag_value(cmd, "--n-cpu-moe") == "44"  # 3 leading dense + 41
    assert _flag_value(cmd, "--gpu-layers") == "20"


def test_load_model_ik_flavor_without_rtr_keeps_ot(tmp_path, monkeypatch):
    backend, gguf = _integration_backend(tmp_path, [(0, 24 * 1024, 24 * 1024)])
    mainline, sibling = _ik_pair(tmp_path)
    backend._find_llama_server_binary = lambda include_denied = False: str(mainline)
    backend._get_gguf_size_bytes = lambda _path: int(1.5e12)
    backend._plan_moe_streaming_for_load = lambda **kw: _plan()
    backend.probe_server_capabilities = lambda binary = None: {
        "found": True,
        "supports_kv_unified": False,
        "supports_rtr": False,
    }
    monkeypatch.setenv("UNSLOTH_STUDIO_LLAMA_FLAVOR", "ik")

    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != str(sibling):
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(gguf_path = str(gguf), model_identifier = "test")

    cmd = captured["cmd"]
    assert cmd[0] == str(sibling)
    assert "-rtr" not in cmd
    assert "-ot" in cmd
    assert _flag_value(cmd, "--n-cpu-moe") == "44"
