import hashlib

import pytest

from splatgif import weights


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(weights.ENV_PATH, raising=False)
    monkeypatch.delenv(weights.ENV_ACCEPT, raising=False)


class FakeTty:
    def isatty(self):
        return True

    def write(self, _text):  # the licence notice is printed here
        return 0

    def flush(self):
        pass


class NoTty:
    def isatty(self):
        return False


def serve(tmp_path, monkeypatch, payload=b"pretend weights", digest=None):
    """Point the downloader at a local file (urllib handles file://) with a chosen checksum."""
    source = tmp_path / "remote.pt"
    source.write_bytes(payload)
    target = tmp_path / "cache" / "w.pt"
    monkeypatch.setattr(weights, "URL", source.as_uri())
    monkeypatch.setattr(weights, "SHA256", digest or hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(weights, "default_path", lambda: target)
    return target


def test_explicit_path_wins_and_must_exist(tmp_path, clean_env):
    f = tmp_path / "w.pt"
    f.write_bytes(b"x")
    assert weights.resolve_weights(f) == f
    with pytest.raises(FileNotFoundError):
        weights.resolve_weights(tmp_path / "missing.pt")


def test_env_var_is_honoured(tmp_path, monkeypatch, clean_env):
    f = tmp_path / "w.pt"
    f.write_bytes(b"x")
    monkeypatch.setenv(weights.ENV_PATH, str(f))
    assert weights.resolve_weights() == f


def test_cached_download_is_reused(tmp_path, monkeypatch, clean_env):
    target = serve(tmp_path, monkeypatch)
    target.parent.mkdir()
    target.write_bytes(b"already here")
    assert weights.resolve_weights() == target  # no consent needed, nothing fetched
    assert target.read_bytes() == b"already here"


def test_download_needs_consent_when_not_interactive(tmp_path, monkeypatch, clean_env):
    target = serve(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.stdin", NoTty())
    with pytest.raises(RuntimeError, match="research"):
        weights.resolve_weights()
    assert not target.exists()


def test_consent_flag_downloads_and_verifies(tmp_path, monkeypatch, clean_env):
    target = serve(tmp_path, monkeypatch)
    assert weights.resolve_weights(accept_license=True) == target
    assert target.read_bytes() == b"pretend weights"


def test_consent_env_var_downloads(tmp_path, monkeypatch, clean_env):
    target = serve(tmp_path, monkeypatch)
    monkeypatch.setenv(weights.ENV_ACCEPT, "1")
    assert weights.resolve_weights() == target and target.exists()


def test_checksum_mismatch_leaves_nothing_behind(tmp_path, monkeypatch, clean_env):
    target = serve(tmp_path, monkeypatch, digest="0" * 64)
    with pytest.raises(RuntimeError, match="hash"):
        weights.resolve_weights(accept_license=True)
    assert not target.exists()


def test_interactive_prompt_accepts_or_declines(tmp_path, monkeypatch, clean_env):
    target = serve(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.stdin", FakeTty())
    monkeypatch.setattr("sys.stderr", FakeTty())
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    with pytest.raises(SystemExit):
        weights.resolve_weights()
    assert not target.exists()
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    assert weights.resolve_weights() == target and target.exists()
