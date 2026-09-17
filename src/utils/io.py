"""Shared filesystem, hashing and JSON helpers.

These helpers are used by the Phase 01 parser (:mod:`src.parsing.parse_graphviz`)
to write the canonical ``parsed_graph.json`` artifact and to make that artifact
reproducible:

* :func:`write_json_atomic` never leaves a partially written file behind -- the
  payload is serialized first and then swapped into place with ``os.replace``.
* :func:`utc_timestamp` honours the ``SOURCE_DATE_EPOCH`` environment variable so
  that two runs over the same input produce byte-identical artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_ENCODING",
    "JSON_INDENT",
    "SHA256_CHUNK_SIZE",
    "SOURCE_DATE_EPOCH_ENV",
    "dumps_json",
    "ensure_dir",
    "read_json",
    "sha256_file",
    "utc_timestamp",
    "write_json_atomic",
]

DEFAULT_ENCODING = "utf-8"
JSON_INDENT = 2
SHA256_CHUNK_SIZE = 1 << 20  # 1 MiB
SOURCE_DATE_EPOCH_ENV = "SOURCE_DATE_EPOCH"


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create *path* (and its parents) if required and return it as a :class:`Path`."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sha256_file(path: str | os.PathLike[str], chunk_size: int = SHA256_CHUNK_SIZE) -> str:
    """Return the hexadecimal SHA-256 digest of the raw bytes of *path*."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp(now: datetime | None = None) -> str:
    """Return an RFC 3339 UTC timestamp (``YYYY-MM-DDThh:mm:ssZ``).

    Resolution order:

    1. an explicit *now* (must be timezone-aware; naive values are assumed UTC),
    2. the ``SOURCE_DATE_EPOCH`` environment variable (integer seconds), which
       makes output byte-reproducible across runs,
    3. the current wall-clock time.
    """
    if now is not None:
        moment = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        moment = moment.astimezone(timezone.utc)
    else:
        epoch = os.environ.get(SOURCE_DATE_EPOCH_ENV)
        if epoch is not None and epoch.strip():
            try:
                seconds = int(float(epoch))
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"{SOURCE_DATE_EPOCH_ENV} must be an integer number of seconds, got {epoch!r}"
                ) from exc
            moment = datetime.fromtimestamp(seconds, tz=timezone.utc)
        else:
            moment = datetime.now(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def dumps_json(payload: Any) -> str:
    """Serialize *payload* using the project's canonical JSON style.

    Keys are sorted, indentation is two spaces, non-ASCII characters are kept
    verbatim and the result ends with a single trailing newline.
    """
    return json.dumps(payload, sort_keys=True, indent=JSON_INDENT, ensure_ascii=False) + "\n"


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> Path:
    """Write *payload* to *path* atomically.

    The payload is serialized before any file is touched.  It is then written to
    a temporary file in the destination directory, flushed and ``fsync``-ed
    before being moved onto *path* with :func:`os.replace`, so readers either
    see the previous file or the complete new one -- never a partial write.
    """
    destination = Path(path)
    ensure_dir(destination.parent)
    text = dumps_json(payload)  # may raise (e.g. TypeError) before touching disk

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=DEFAULT_ENCODING,
        newline="\n",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read and decode a UTF-8 JSON document from *path*."""
    with open(path, encoding=DEFAULT_ENCODING) as handle:
        return json.load(handle)