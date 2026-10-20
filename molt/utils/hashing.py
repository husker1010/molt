"""Content hashing for files, directories, data frames and configuration objects."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = ["sha256_file", "sha256_bytes", "stable_hash", "hash_directory", "hash_dataframe"]

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = _CHUNK) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    return str(obj)


def stable_hash(obj: Any, length: int | None = None) -> str:
    """Hash a JSON-serialisable object independently of dict ordering.

    Args:
        obj: Object to hash (dicts, lists, scalars, paths, NumPy values).
        length: Optional prefix length of the returned hex digest.
    """
    payload = json.dumps(obj, sort_keys=True, default=_default, separators=(",", ":"))
    digest = sha256_bytes(payload.encode("utf-8"))
    return digest[:length] if length else digest


def hash_directory(root: str | os.PathLike[str], patterns: Iterable[str] = ("**/*",)) -> str:
    """Hash relative paths and contents of all files under ``root`` matching ``patterns``."""
    root = Path(root)
    files = sorted({p for pat in patterns for p in root.glob(pat) if p.is_file()})
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def hash_dataframe(df: Any) -> str:
    """Hash a pandas ``DataFrame`` including its index and column labels."""
    import pandas as pd

    values = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    cols = json.dumps([str(c) for c in df.columns]).encode("utf-8")
    return sha256_bytes(cols + values)
