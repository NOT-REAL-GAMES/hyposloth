# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Portable sequential-read probe for NVMe streaming planning.

The MoE streaming planner (``core/inference/moe_streaming_planner.py``) needs
one number from the host: how fast can the system disk actually deliver
sequential reads? That number decides whether streaming a multi-hundred-GB
MoE GGUF from NVMe is usable or hopeless.

This probe trades accuracy for portability, deliberately:

- No ``O_DIRECT`` / ``FILE_FLAG_NO_BUFFERING``: those are platform-specific,
  need aligned buffers, and require admin rights on some systems. Instead we
  do a throwaway pass over a DIFFERENT 64 MiB region at the file's end before
  measuring, so the measured region is less likely to be served purely from
  page cache. The result is a conservative LOWER bound on cold-read speed --
  exactly the direction a feasibility planner should err in.
- No threads, no async, no third-party deps: the probe runs on the launcher
  path where importing psutil/pyserial-able natives is not an option.

Results are cached per (abspath, mib) because probing takes a noticeable
fraction of a second and callers re-plan often.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Tuple, Union

_MIB = 1024 * 1024

# Size of the throwaway cache-defeat pass at the file's end. 64 MiB is large
# enough to evict a small measured region from a typical readahead window
# without adding more than ~20 ms on any NVMe worth planning around.
_WARMUP_BYTES = 64 * _MIB

# os.pread exists on POSIX but not on Windows. Branch once at import into a
# module-level flag so tests can force the portable path deterministically.
_HAS_PREAD = hasattr(os, "pread")

# (abspath, mib) -> measured MB/s. Only successful (> 0) results are cached:
# a transient OSError must not poison later probes.
_CACHE: Dict[Tuple[str, int], float] = {}


def clear_cache() -> None:
    """Drop all cached probe results. Exists for tests and for callers that
    suspect the media changed (e.g. model moved to a different disk)."""
    _CACHE.clear()


def _read_region(fh, offset: int, total: int, chunk: int) -> int:
    """Read up to ``total`` bytes starting at ``offset``; return bytes read.

    Uses positioned reads (``os.pread``) where available so the two passes
    cannot interfere through the shared file offset; falls back to a plain
    seek + sequential read elsewhere (Windows). Either way the access pattern
    the kernel sees is a linear scan, which is what we want to measure.
    """
    if _HAS_PREAD:
        fd = fh.fileno()
        read = 0
        while read < total:
            data = os.pread(fd, min(chunk, total - read), offset + read)
            if not data:
                break
            read += len(data)
        return read
    fh.seek(offset)
    read = 0
    while read < total:
        data = fh.read(min(chunk, total - read))
        if not data:
            break
        read += len(data)
    return read


def probe_sequential_read_mbps(
    path: Union[str, Path], mib: int = 256, _chunk_mib: int = 8
) -> float:
    """Measure sequential read speed on the file at ``path``, in decimal MB/s.

    Reads ``mib`` MiB (or the whole file when smaller) in ``_chunk_mib``
    chunks from the file's start, timed with ``time.perf_counter``, after a
    throwaway pass over the file's last 64 MiB that warms nothing useful.
    Returns bytes/1e6 per second as a float. Returns 0.0 on any OSError
    (missing file, unreadable media, ...) -- callers treat 0 as "unknown"
    and fall back to a documented default. Positive results are cached per
    (abspath, mib); use ``clear_cache()`` to force re-measurement.
    """
    key = (os.path.abspath(os.fspath(path)), int(mib))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    try:
        size = os.path.getsize(key[0])
        if size <= 0:
            return 0.0
        chunk = max(1, int(_chunk_mib)) * _MIB
        warmup = min(_WARMUP_BYTES, size)
        measure = min(int(mib) * _MIB, size)
        with open(key[0], "rb") as fh:
            # Throwaway pass over the tail: pulls 64 MiB through the cache
            # that the measured pass will not touch, so the measurement below
            # is less likely to be served from page cache. Not timed.
            _read_region(fh, size - warmup, warmup, chunk)
            start = time.perf_counter()
            got = _read_region(fh, 0, measure, chunk)
            elapsed = time.perf_counter() - start
        if got <= 0 or elapsed <= 0.0:
            return 0.0
        result = (got / 1e6) / elapsed
    except OSError:
        return 0.0

    _CACHE[key] = result
    return result
