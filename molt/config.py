"""YAML configuration loading with inheritance, interpolation and dotted overrides.

Configurations are plain nested dictionaries wrapped in :class:`Config`, which adds
attribute access and dotted-key lookup. Features:

* ``defaults: [other.yaml, ...]`` merges base files (relative to the including file) first.
* ``${a.b.c}`` string references are expanded against the merged config; environment
  variables are available as ``${env:NAME}`` (or ``${env:NAME,default}``).
* Overrides given as ``"a.b=value"`` strings are parsed with YAML semantics.
* Keys listed in :data:`PATH_KEYS` or living under ``paths`` are resolved to absolute paths.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "merge_dicts",
    "apply_overrides",
    "parse_override",
]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"
"""Location of the bundled default configuration."""

PATH_KEYS = frozenset(
    {
        "data_root",
        "runs_root",
        "output_dir",
        "checkpoint_dir",
        "cache_dir",
        "feature_dir",
        "swin_checkpoint",
    }
)
_REF = re.compile(r"\$\{([^${}]+)\}")
_MISSING = object()


class Config(dict):
    """A ``dict`` with attribute access and dotted-key helpers.

    Nested mappings are converted to :class:`Config` on construction so that
    ``cfg.paths.data_root`` and ``cfg["paths"]["data_root"]`` are equivalent.
    """

    def __init__(self, data: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        for key, value in {**(data or {}), **kwargs}.items():
            self[key] = _wrap(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def __deepcopy__(self, memo: dict) -> "Config":
        return Config(copy.deepcopy(dict(self), memo))

    def get_path(self, dotted: str, default: Any = _MISSING) -> Any:
        """Return the value at ``dotted`` (e.g. ``"cv.outer_folds"``).

        Args:
            dotted: Dot-separated key path.
            default: Value returned when the key is absent; raises ``KeyError`` if omitted.
        """
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.lstrip("-").isdigit():
                node = node[int(part)]
            else:
                if default is _MISSING:
                    raise KeyError(dotted)
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """Set ``dotted`` to ``value``, creating intermediate mappings as needed."""
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Mapping):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = _wrap(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain (recursively unwrapped) dictionary."""
        return _unwrap(self)

    def to_yaml(self) -> str:
        """Serialise the configuration to YAML."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the configuration as YAML to ``path`` and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def merge_dicts(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into a copy of ``base``; lists are replaced, not merged."""
    out = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(item: str) -> tuple[str, Any]:
    """Split ``"a.b=value"`` into the key and a YAML-parsed value."""
    if "=" not in item:
        raise ValueError(f"Override must look like key=value, got {item!r}")
    key, raw = item.split("=", 1)
    key = key.strip().lstrip("+")
    if not key:
        raise ValueError(f"Empty key in override {item!r}")
    return key, yaml.safe_load(raw) if raw.strip() else None


def apply_overrides(cfg: Config, overrides: Iterable[str] | Mapping[str, Any] | None) -> Config:
    """Apply dotted overrides (``["cv.outer_folds=3"]`` or ``{"cv.outer_folds": 3}``) in place."""
    if not overrides:
        return cfg
    items = overrides.items() if isinstance(overrides, Mapping) else map(parse_override, overrides)
    for key, value in items:
        cfg.set_path(key, value)
    return cfg


def _read_yaml(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular 'defaults' include involving {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Top level of {path} must be a mapping")
    bases = data.pop("defaults", None) or []
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = merge_dicts(merged, _read_yaml(base_path, _seen | {path}))
    return merge_dicts(merged, data)


def _interpolate(cfg: Config, max_passes: int = 10) -> None:
    def resolve(value: Any) -> Any:
        if isinstance(value, Mapping):
            for k in list(value.keys()):
                value[k] = resolve(value[k])
            return value
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if not isinstance(value, str) or "${" not in value:
            return value
        full = _REF.fullmatch(value)
        if full:
            return _lookup(full.group(1))
        return _REF.sub(lambda m: str(_lookup(m.group(1))), value)

    def _lookup(ref: str) -> Any:
        ref = ref.strip()
        if ref.startswith("env:"):
            name, _, default = ref[4:].partition(",")
            if name in os.environ:
                return os.environ[name]
            if default:
                return default
            raise KeyError(f"Environment variable {name!r} is not set")
        return cfg.get_path(ref)

    for _ in range(max_passes):
        before = cfg.to_yaml()
        resolve(cfg)
        if cfg.to_yaml() == before:
            return
    raise ValueError("Configuration interpolation did not converge (cyclic ${...} references?)")


def _resolve_paths(cfg: Config, base_dir: Path) -> None:
    def to_abs(value: Any) -> Any:
        if isinstance(value, str) and value and "://" not in value:
            p = Path(os.path.expandvars(os.path.expanduser(value)))
            return str(p if p.is_absolute() else (base_dir / p).resolve())
        return value

    paths = cfg.get("paths")
    if isinstance(paths, Mapping):
        for key in list(paths.keys()):
            paths[key] = to_abs(paths[key])

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key in list(node.keys()):
                if key in PATH_KEYS and node is not paths:
                    node[key] = to_abs(node[key])
                else:
                    walk(node[key])
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(cfg)


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Iterable[str] | Mapping[str, Any] | None = None,
    *,
    with_defaults: bool = True,
    base_dir: str | os.PathLike[str] | None = None,
) -> Config:
    """Load a YAML configuration.

    Args:
        path: Config file. ``None`` loads only the bundled default configuration.
        overrides: Dotted ``key=value`` overrides applied after merging.
        with_defaults: Merge ``path`` on top of :data:`DEFAULT_CONFIG_PATH`.
        base_dir: Directory used to resolve relative paths (defaults to the config's directory).

    Returns:
        The merged, interpolated and path-resolved :class:`Config`.
    """
    data: dict[str, Any] = {}
    if with_defaults or path is None:
        data = _read_yaml(DEFAULT_CONFIG_PATH)
    src = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if path is not None and src.resolve() != DEFAULT_CONFIG_PATH.resolve():
        data = merge_dicts(data, _read_yaml(src))
    cfg = Config(data)
    apply_overrides(cfg, overrides)
    _interpolate(cfg)
    _resolve_paths(cfg, Path(base_dir) if base_dir else src.resolve().parent)
    return cfg
