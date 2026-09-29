# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression test for named CMake build discovery in setup.ps1."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def _function_source(source: str, name: str) -> str:
    marker = f"function {name}"
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"Unclosed PowerShell function: {name}")


@pytest.mark.skipif(POWERSHELL is None, reason = "PowerShell is unavailable")
def test_setup_prefers_named_cuda_build(tmp_path):
    cpu = tmp_path / "build-vkv" / "bin" / "llama-server.exe"
    cuda = tmp_path / "build-vkv-cuda" / "bin" / "Release" / "llama-server.exe"
    cpu.parent.mkdir(parents = True)
    cuda.parent.mkdir(parents = True)
    cpu.write_bytes(b"")
    cuda.write_bytes(b"")

    helper = _function_source(SETUP_PS1.read_text(encoding = "utf-8"), "Get-LlamaServerCandidates")
    root = json.dumps(str(tmp_path))
    script = (
        f"{helper}\n"
        f"Get-LlamaServerCandidates -Root {root} | "
        "Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | "
        "Select-Object -First 1"
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
    )

    assert Path(result.stdout.strip()).resolve() == cuda.resolve()
