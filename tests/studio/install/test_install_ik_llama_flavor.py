import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
INSTALL_LLAMA_PREBUILT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INSTALL_LLAMA_PREBUILT
SPEC.loader.exec_module(INSTALL_LLAMA_PREBUILT)

install_ik_llama_flavor = INSTALL_LLAMA_PREBUILT.install_ik_llama_flavor
ik_llama_requested = INSTALL_LLAMA_PREBUILT.ik_llama_requested
ik_server_binary_name = INSTALL_LLAMA_PREBUILT.ik_server_binary_name
IK_PUBLISHED_REPO = INSTALL_LLAMA_PREBUILT.IK_PUBLISHED_REPO


def _release_payload(name: str | None) -> dict:
    assets = [] if name is None else [{"name": name}]
    return {"tag_name": "ik-b1", "assets": assets}


def test_missing_release_skips_cleanly(tmp_path, monkeypatch):
    # ik_llama.cpp upstream publishes no prebuilts: the fetch 404s and the
    # flavor install must skip, never raise or disturb the main install.
    def boom(url):
        raise RuntimeError("HTTP 404")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "fetch_json", boom)
    assert install_ik_llama_flavor(tmp_path) is False
    assert list(tmp_path.iterdir()) == []


def test_release_without_matching_asset_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "fetch_json",
        lambda url: _release_payload("llama-server-ik-linux-x64.tar.gz"),
    )
    assert install_ik_llama_flavor(tmp_path) is False
    assert list(tmp_path.iterdir()) == []


def test_release_without_tag_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "fetch_json",
        lambda url: {"assets": [{"name": ik_server_binary_name()}]},
    )
    assert install_ik_llama_flavor(tmp_path) is False
    assert list(tmp_path.iterdir()) == []


def test_matching_asset_downloads_next_to_server(tmp_path, monkeypatch):
    name = ik_server_binary_name()
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "fetch_json", lambda url: _release_payload(name)
    )
    downloaded = {}

    def fake_download(url, destination):
        downloaded["url"] = url
        Path(destination).write_bytes(b"ik-binary")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "download_file", fake_download)
    assert install_ik_llama_flavor(tmp_path) is True
    assert (tmp_path / name).read_bytes() == b"ik-binary"
    assert "unslothai/ik_llama.cpp" in downloaded["url"]
    assert "ik-b1" in downloaded["url"]
    assert name in downloaded["url"]


def test_download_failure_skips_and_cleans_up(tmp_path, monkeypatch):
    name = ik_server_binary_name()
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "fetch_json", lambda url: _release_payload(name)
    )

    def fake_download(url, destination):
        Path(destination).write_bytes(b"partial")
        raise RuntimeError("connection reset")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "download_file", fake_download)
    assert install_ik_llama_flavor(tmp_path) is False
    assert not (tmp_path / name).exists()


def test_ik_llama_requested_flag_and_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_IK_LLAMA", raising = False)
    assert ik_llama_requested(False) is False
    assert ik_llama_requested(True) is True
    monkeypatch.setenv("UNSLOTH_STUDIO_IK_LLAMA", "1")
    assert ik_llama_requested(False) is True
    monkeypatch.setenv("UNSLOTH_STUDIO_IK_LLAMA", "0")
    assert ik_llama_requested(False) is False


def test_ik_flag_defaults_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["install_llama_prebuilt.py"])
    args = INSTALL_LLAMA_PREBUILT.parse_args()
    assert args.ik is False


def test_default_ik_repo_is_fork():
    assert IK_PUBLISHED_REPO == "unslothai/ik_llama.cpp"
