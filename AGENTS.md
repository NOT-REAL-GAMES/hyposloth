# AGENTS.md

Guidance for AI coding agents working in this repository. This file describes
the project as it actually is; trust the code over this file where they drift.

## Project overview

This is **Hyposloth** — an open-source project for 2–5x faster LLM training,
reinforcement learning and fine-tuning with ~70% less VRAM. The repository is a
monorepo containing two products plus a CLI:

- **Hyposloth Core** (`unsloth/` package): the Python fine-tuning/RL library.
  Patched model implementations, custom Triton kernels, and integrations with
  PyTorch, Hugging Face (transformers, TRL, PEFT, accelerate), bitsandbytes,
  xformers and the companion `unsloth_zoo` package. Licensed **Apache-2.0**
  (see `LICENSE`).
- **Hyposloth Studio** (`studio/`): a local web UI to run and train models.
  FastAPI backend + React/TypeScript frontend + an optional Tauri 2 (Rust)
  desktop shell. Licensed **AGPL-3.0-only** (see `studio/LICENSE.AGPL-3.0`).
- **`unsloth` CLI** (`unsloth_cli/`): a Typer-based command line
  (`unsloth = unsloth_cli:app` console script) covering `train`, `inference`,
  `chat`, `export`, `studio` and `start` (agent integrations).

Supported platforms: Windows, Linux, WSL and macOS (Apple Silicon via MLX).
NVIDIA CUDA, AMD ROCm, Intel XPU, Vulkan (GGUF inference only) and CPU paths
all exist. Python `>=3.9,<3.15`.

The package version lives in `unsloth/models/_utils.py` as `__version__`
(currently `2026.7.6`, calver-style) and is read by setuptools via
`[tool.setuptools.dynamic]`. `CHANGELOG.md` at the repo root feeds the Studio
"new version" popup; each release is a level-2 heading like
`## 2026.7.6 - 2026-07-22`.

## Repository layout

- `unsloth/` — Core library (Apache-2.0).
  - `models/` — per-architecture patches and loaders: `llama.py`, `gemma.py`,
    `gemma2.py`, `qwen2.py`, `qwen3.py`, `qwen3_moe.py`, `mistral.py`,
    `cohere.py`, `granite.py`, `llama4.py`, `glm4_moe.py`, `falcon_h1.py`,
    `vision.py`, `diffusion.py`, `sentence_transformer.py`, plus `loader.py`,
    `loader_utils.py`, `_utils.py`, `rl.py`, `rl_replacements.py`, `dpo.py`,
    `mapper.py` (model-name → implementation mapping; excluded from ruff).
  - `kernels/` — custom Triton kernels: `cross_entropy_loss.py`,
    `rms_layernorm.py`, `rope_embedding.py`, `swiglu.py`, `geglu.py`,
    `fast_lora.py`, `flex_attention.py`, `fp8.py`, `moe/`.
  - `registry/` — model registry (`registry.py` + per-family files, with
    `REGISTRY.md` docs).
  - `dataprep/`, `optimizers/`, `utils/` — synthetic data, Q-GaLore optimizer,
    packing/attention helpers.
  - Top level: `trainer.py`, `save.py` (GGUF/safetensors export),
    `chat_templates.py`, `tokenizer_utils.py`, `device_type.py`,
    `_gpu_init.py`, `import_fixes.py`.
- `unsloth_cli/` — CLI (Typer). `commands/{train,inference,chat,export,start,
  studio}.py`; subagent MCP bridges for Claude/Codex; its own test suite in
  `unsloth_cli/tests/`.
- `studio/` — Hyposloth Studio (AGPL-3.0).
  - `backend/` — FastAPI app. Entry `main.py`; `routes/` (chat, inference,
    training, export, datasets, models, auth, MCP servers, whisper, RAG…);
    `core/` (`inference/`, `training/`, `export/`, `data_recipe/`, `rag/`);
    `auth/`, `state/`, `storage/`, `hub/`, `plugins/`, `utils/`;
    `requirements/` (`base.txt`, `studio.txt`, `no-torch-runtime.txt`,
    `overrides.txt`, …); backend tests in `studio/backend/tests/` (~860 tests).
  - `frontend/` — React 19 + TypeScript + Vite + Tailwind CSS v4 +
    TanStack Router. Tests are `tests/**/*.test.ts` run by `node --test`.
  - `src-tauri/` — Tauri 2 desktop shell (Rust). `Cargo.toml`,
    `tauri.conf.json` + per-OS overrides.
  - Root installers/bootstrap: `setup.sh/.ps1/.bat`, `install_*_prebuilt.py`,
    `install_python_stack.py`, `MCP.md` (opt-in MCP control server docs).
- `tests/` — repo-level test tree (see Testing below).
- `scripts/` — dev/CI helpers: `run_ruff_format.py` + `enforce_kwargs_spacing.py`
  (formatting pipeline), `verify_import_hoist.py`, supply-chain scanners
  (`scan_packages.py`, `scan_npm_packages.py` + baseline JSONs),
  `lockfile_supply_chain_audit.py`, `notebook_validator.py`, `uninstall.sh/.ps1`.
- Root installers: `install.sh`, `install.ps1` — the user-facing one-liner
  installers (`curl | sh` / `irm | iex`). Tests under `tests/sh/` and
  `tests/studio_setup_ps1/` assert against them.
- `build.sh` — release build (frontend → wheel/sdist → optional publish).
- `.github/workflows/` — ~35 CI workflows (see CI section).
- `cli.py`, `unsloth-cli.py` — alternate/legacy CLI entry points.

## Technology stack

- **Python** 3.9–3.14. Build backend: setuptools (pinned `setuptools==80.9.0`,
  `setuptools-scm==9.2.0`). `pyproject.toml` is ~1400 lines, most of it a
  CUDA-version × torch-version extras matrix (`cu118`…`cu130` ×
  `torch211`…`torch2100`) pinning exact xformers wheels; the `studio` extra
  mirrors `studio/backend/requirements/studio.txt` (a test enforces parity).
- **Core ML**: torch, `unsloth_zoo` (companion package, much patching lives
  there), transformers (pinned range, currently `>=4.51.3,<=5.5.0` with
  exclusions), TRL, PEFT, accelerate, bitsandbytes, xformers, triton.
- **Studio backend**: FastAPI + uvicorn, pydantic, structlog, SQLAlchemy/sqlite
  (`studio.db`), llama.cpp for GGUF inference, MLX on Apple Silicon.
- **Studio frontend**: Node `^20.19 || >=22.12`, React 19, Vite, TypeScript,
  Tailwind v4, shadcn/radix; ESLint + Biome.
- **Desktop**: Tauri 2 (Rust edition 2021).

## Build and run

### Local development install (Studio, from source)

```bash
# macOS / Linux / WSL
./install.sh --local
unsloth studio -p 8888

# Windows (PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1 --local
unsloth studio -p 8888
```

`UNSLOTH_STUDIO_HOME=<dir>` isolates the venv, auth DB, caches and llama.cpp
build; pass it to both install and launch. Useful env vars:
`UNSLOTH_NO_TORCH=1` (GGUF-only install), `UNSLOTH_FORCE_VULKAN=1` (Vulkan
llama.cpp bundle), `UNSLOTH_NPM_REGISTRY=<url>` (npm mirror),
`UNSLOTH_COMPILE_DISABLE=1` (skip torch.compile in tests).

### Hyposloth Core (library)

```bash
uv venv unsloth_env --python 3.13 && source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto        # from PyPI
pip install -e ".[huggingface]"                    # editable from this repo
```

### Frontend

```bash
cd studio/frontend
npm ci                     # lockfile-pinned install (supply-chain locked)
npm run dev                # vite dev server
npm run build              # tsc -b && vite build → dist/
npm run typecheck          # tsc app + tests configs
npm run lint               # eslint
npm run biome:check        # biome (biome:fix to auto-fix)
npm test                   # node --test over tests/**/*.test.ts
npm run i18n:check         # locale parity check
```

### Release build / publish

```bash
./build.sh            # builds frontend (bun or npm), validates CSS output,
                      # stamps release metadata, builds wheel + sdist
./build.sh publish    # additionally verifies dist and uploads via twine
```

### Desktop app

`studio/package.json` exists only to pin `@tauri-apps/cli`
(`npm ci --prefix studio`); the Rust crate builds from `studio/src-tauri`
(see `release-desktop.yml` and `studio-tauri-smoke.yml`).

## Testing

pytest is configured in `pyproject.toml`: default `testpaths` is
**`tests/security` only**, and `pythonpath = ["."]`. Running bare `pytest`
from the repo root is the security suite, not the whole tree — the GPU-heavy
directories are excluded from default discovery on purpose.

The CI test commands (from `studio-backend-ci.yml` and friends) are:

- **Backend suite** (~860 CPU-friendly tests, matrix over Python 3.10–3.13):
  ```bash
  cd studio/backend
  pip install -r requirements/studio.txt
  python -m pytest tests/ -q --tb=short \
      --ignore=tests/test_studio_api.py \
      -k 'not llama_cpp_load_progress_live'
  ```
  (`test_studio_api.py` is end-to-end against a live model download;
  `llama_cpp_load_progress_live` spawns a real llama.cpp process.)
- **Repo CPU tests** (auto-discovered; needs `node` and `uv` installed for
  full coverage):
  ```bash
  python -m pytest tests/ -q --tb=short \
      --ignore=tests/qlora --ignore=tests/saving --ignore=tests/utils \
      --ignore=tests/sh --ignore=tests/vllm_compat --ignore=tests/version_compat \
      -m 'not server and not e2e'
  ```
  `tests/conftest.py` pre-loads `unsloth_zoo.device_type` under a mocked
  `torch.cuda.is_available()==True` so the unsloth import chain works on
  CPU-only machines (skipped when a real accelerator exists). The
  hardware-spoof files (`tests/studio/test_hardware_dispatch_matrix.py`,
  `test_is_mlx_dispatch_gate.py`, `test_xpu_spoof_pipeline.py`) mutate module
  globals and must run in their **own pytest invocation**.
- **CLI tests**: `python -m pytest unsloth_cli/tests -q --tb=short`
  (self-bootstrapping; imports neither unsloth nor torch).
- **Installer tests**: `tests/run_all.sh` — discovers and runs `tests/sh/test_*.sh`
  with bash (not sh — several rely on bashisms) plus the Python installer
  tests (`tests/python/test_install_*.py` etc.).
- **GPU tests**: `tests/qlora/`, `tests/saving/`, `tests/fast_inference/` and
  parts of `tests/python/` require a real GPU and are not run in CPU CI.
- **Core compatibility**: `consolidated-tests-ci.yml` runs the CPU suite
  against three (transformers, TRL) combinations, plus the `unsloth_zoo@main`
  CPU test suite — keep changes green across transformers 4.57.x and 5.x.

New tests in covered directories are auto-discovered; prefer adding tests next
to the area you change (backend → `studio/backend/tests/`, repo-level →
`tests/`, CLI → `unsloth_cli/tests/`).

## Code style guidelines

- **Ruff** is the single linter/formatter. `pyproject.toml`: line-length 100,
  target py311, lint rules deliberately narrow (`E9`, `F63`, `F7`, `F82` —
  syntax errors and undefined names only). Excluded everywhere:
  `chat_templates.py`, `ollama_template_mappers.py`, `_auto_install.py`,
  `mapper.py`.
- **Formatting is not plain `ruff format`.** The pre-commit hook runs
  `scripts/run_ruff_format.py`, which wraps `ruff format` with
  `scripts/enforce_kwargs_spacing.py`:
  - keyword arguments get spaces around `=`: `encoding = "utf-8"`;
  - redundant `pass`, blank lines after short import blocks, adjacent string
    literals and "magic" trailing commas are normalized.
  Match this style by hand or run `pre-commit run --all-files`.
  `.pre-commit-config.yaml` pins `ruff==0.6.9` for formatting; lint CI pins
  `ruff==0.15.12` for checking.
- **License headers**: core `unsloth/` files carry the Apache-2.0 header;
  Studio, CLI, scripts, tests and CI files carry
  `# SPDX-License-Identifier: AGPL-3.0-only` + the Unsloth AI copyright line.
  Keep headers consistent with the directory you edit.
- Comment style in CI/scripts is deliberately verbose "why" commentary —
  follow suit when editing those files.
- TypeScript: ESLint (`npm run lint`), Biome (`npm run biome:check`), strict
  `tsc` (`npm run typecheck`) are all gating in frontend CI.
- PR conventions (from `CONTRIBUTING.md`): one focused change per PR, concise
  description and motivation, link related issues. Branches: `main` (nightly
  source) and `pip`.

## CI and deployment

Key workflows in `.github/workflows/`:

- `lint-ci.yml` — runs on **every** PR: `compileall` on all Python, ruff
  check, `bash -n` on every `.sh`, YAML/JSON parse, codespell, shellcheck,
  and `scripts/verify_import_hoist.py` (a scope-aware AST check guarding
  import-hoisting refactors; run `python scripts/verify_import_hoist.py
  --self-test` locally if you touch it).
- `studio-backend-ci.yml` — backend pytest matrix + repo CPU tests + CLI
  tests, triggered by changes to `studio/`, `unsloth/`, `unsloth_cli/`,
  `tests/`, `scripts/`, root installers or `pyproject.toml`.
- `consolidated-tests-ci.yml` ("Core") — CPU suite × 3 transformers/TRL
  combos + unsloth_zoo suite.
- `studio-frontend-ci.yml` — lockfile audit, `npm ci --strict-allow-scripts`,
  typecheck, tests, build, biome.
- `studio-tauri-smoke.yml` / `release-desktop.yml` — Tauri build and signed
  desktop releases.
- `security-audit.yml`, `lockfile-audit.yml` — see Security section.
- `wheel-smoke.yml`, `clean-machine-install-ci.yml`, `cross-platform-parity-ci.yml`,
  `notebooks-ci.yml`, `version-compat-ci.yml`, various Studio smoke suites
  (API/UI/inference/update, per-OS).

Deployment: PyPI releases go through `./build.sh publish` (stamp → build →
verify-dist → twine). Desktop releases via `release-desktop.yml`. Docker image
`unsloth/unsloth` is published separately.

## Security considerations

- **Studio binds `127.0.0.1` by default.** Remote exposure options:
  `--secure` (Cloudflare HTTPS tunnel, fails closed — the raw port is never
  exposed) or `-H 0.0.0.0` (raw port on all interfaces; add `--cloudflare`
  for a public URL). On first public exposure with the auto-generated admin
  password, a password change is enforced before the link goes up.
- **Authentication**: the Studio API requires an API key/admin password
  (JWT, `cryptography`, diceware-generated secrets; sqlite `studio.db`).
  Anyone with the key can run code on the host: server-side tools (web
  search, Python/terminal execution) run as the launching user and are **on
  by default** — use `--disable-tools` when exposing the server. The
  `studio/auth/` code and the "pre-exposure gate" in `unsloth_cli` are
  heavily tested (`unsloth_cli/tests/`, `tests/studio/`); do not weaken them.
- **MCP control endpoint** is opt-in: `UNSLOTH_STUDIO_ENABLE_MCP=1` plus a
  bearer token (`UNSLOTH_STUDIO_MCP_TOKEN`); see `studio/MCP.md`. It can
  start/stop training and write model artifacts — keep it localhost-only.
- **Supply chain** is treated seriously; keep these mechanisms intact:
  - GitHub Actions are pinned to full commit SHAs.
  - `security-audit.yml` runs pip-audit + npm audit + cargo audit and
    `scripts/scan_packages.py` / `scan_npm_packages.py`, which download and
    pattern-scan every package in the transitive closure against
    `*_baseline.json` allowlists.
  - The frontend lockfile enforces a 7-day `min-release-age` and exact
    version pins; `npm ci --strict-allow-scripts` gates install scripts
    (`scripts/sync_allow_scripts_pins.py` re-pins the allowlist).
  - `tests/security/` (the default pytest target) tests the scanners and
    audit scripts themselves.
- Never commit secrets; installer and auth code writes secrets with strict
  permissions (0o600 etc.) — preserve that.

## Common gotchas

- `import unsloth` performs real work: device detection, `unsloth_zoo`
  patching, and on Apple Silicon an MLX path. Use `UNSLOTH_COMPILE_DISABLE=1`
  and the CPU-spoof conftest when testing on machines without GPUs.
- The backend pins `CUDA_DEVICE_ORDER=PCI_BUS_ID` before any torch import so
  GPU indices match `nvidia-smi`; keep index-based GPU logic consistent with
  PCI ordering.
- Windows consoles are forced to UTF-8 in `unsloth/__init__.py`,
  `unsloth_cli/__init__.py` and `studio/backend/main.py` — keep non-ASCII
  output behind those guards.
- `pyproject.toml`'s `studio` extra must stay in sync with
  `studio/backend/requirements/studio.txt` (a test fails on drift).
- Tailwind v4's scanner respects parent `.gitignore` files containing `*`;
  `build.sh` temporarily hides them during the frontend build. A CSS output
  under ~100 KB means the scan silently failed.
- Shell tests must run under `bash`, not `sh` (bashisms; see
  `tests/run_all.sh`).
