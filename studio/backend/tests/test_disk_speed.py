# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the sequential-read disk speed probe.

The probe feeds the MoE streaming planner its one measured host number. The
timing/IO seams are monkeypatched so the MB/s arithmetic, the result cache
and the OSError fallback are pinned deterministically without touching real
disks; one smoke test exercises a real small file end to end.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load disk_speed.py directly to avoid dragging in the full backend chain via
# utils/hardware/__init__.py. The probe is dependency-free. Registered in
# sys.modules before exec_module for symmetry with the dataclass-carrying
# planner loader (dataclasses resolves string annotations there).
_DS_PATH = Path(__file__).resolve().parent.parent / "utils" / "hardware" / "disk_speed.py"
_spec = importlib.util.spec_from_file_location("_ds_test_only", _DS_PATH)
_ds = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _ds
_spec.loader.exec_module(_ds)
probe_sequential_read_mbps = _ds.probe_sequential_read_mbps
clear_cache = _ds.clear_cache

_MIB = 1024 * 1024
_FAKE_SIZE = 16 * _MIB


@pytest.fixture(autouse = True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


class _FakeFile:
    """Zero-filled in-memory file with the seek/read surface the probe's
    portable (non-pread) path uses. The throwaway and measured passes read
    back as many zero bytes as requested."""

    def __init__(self, size):
        self._size = size
        self._pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def seek(self, offset):
        self._pos = offset

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        n = min(n, self._size - self._pos)
        self._pos += n
        return b"\0" * n


def _install_fakes(monkeypatch, times):
    """Patch the probe's IO + clock seams. ``times`` is the sequence of
    perf_counter values; it repeats its last value if asked again so stray
    calls from the harness cannot exhaust it. Returns the open-call counter.
    """
    opens = {"count": 0}

    def fake_open(path, mode="r", *args, **kwargs):
        opens["count"] += 1
        return _FakeFile(_FAKE_SIZE)

    clock = {"values": list(times)}

    def fake_perf_counter():
        if len(clock["values"]) > 1:
            return clock["values"].pop(0)
        return clock["values"][0]

    # Force the portable read path so the fake needs no fileno/pread.
    monkeypatch.setattr(_ds, "_HAS_PREAD", False)
    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr("os.path.getsize", lambda p: _FAKE_SIZE)
    monkeypatch.setattr(_ds.time, "perf_counter", fake_perf_counter)
    return opens


def test_deterministic_mbps(monkeypatch):
    _install_fakes(monkeypatch, [0.0, 0.5])
    # 4 MiB measured in 0.5 s => 4194304 bytes / 1e6 / 0.5 s = 8.388608 MB/s.
    result = probe_sequential_read_mbps("/fake/model.gguf", mib = 4, _chunk_mib = 1)
    assert result == pytest.approx(8.388608)


def test_second_call_hits_cache(monkeypatch):
    opens = _install_fakes(monkeypatch, [0.0, 0.5])
    first = probe_sequential_read_mbps("/fake/model.gguf", mib = 4, _chunk_mib = 1)
    second = probe_sequential_read_mbps("/fake/model.gguf", mib = 4, _chunk_mib = 1)
    assert first == second
    assert opens["count"] == 1


def test_clear_cache_forces_reprobe(monkeypatch):
    opens = _install_fakes(monkeypatch, [0.0, 0.5])
    probe_sequential_read_mbps("/fake/model.gguf", mib = 4, _chunk_mib = 1)
    clear_cache()
    probe_sequential_read_mbps("/fake/model.gguf", mib = 4, _chunk_mib = 1)
    assert opens["count"] == 2


def test_oserror_returns_zero(monkeypatch):
    def broken_open(path, mode="r", *args, **kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr("builtins.open", broken_open)
    assert probe_sequential_read_mbps("/fake/model.gguf", mib = 4) == 0.0


def test_missing_file_returns_zero(tmp_path):
    assert probe_sequential_read_mbps(tmp_path / "no-such-file.gguf", mib = 1) == 0.0


def test_real_file_smoke(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"\0" * (2 * _MIB))
    result = probe_sequential_read_mbps(payload, mib = 1)
    assert result > 0.0
