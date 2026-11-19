import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "download_model", Path(__file__).resolve().parents[1] / "scripts" / "download_model.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class Response(BytesIO):
    def geturl(self):
        return download.MODEL_URL


def test_verified_download_writes_only_matching_bytes(monkeypatch, tmp_path):
    data = b"fake official model for unit test"
    monkeypatch.setattr(download, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(data))
    target = tmp_path / "models" / "model.task"
    download.download_model(target)
    assert target.read_bytes() == data


def test_checksum_failure_does_not_create_model(monkeypatch, tmp_path):
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(b"bad data"))
    target = tmp_path / "model.task"
    with pytest.raises(ValueError, match="checksum"):
        download.download_model(target)
    assert not target.exists()


def test_existing_mismatched_file_never_overwritten(tmp_path):
    target = tmp_path / "existing.task"
    target.write_bytes(b"keep this file")
    with pytest.raises(ValueError, match="move it aside"):
        download.download_model(target)
    assert target.read_bytes() == b"keep this file"
