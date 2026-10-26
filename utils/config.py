"""Strict runtime configuration layering and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import ml_collections
import yaml


def normalize_packaged_file_flag(
    flag_name: str,
    *,
    package_root: str | Path,
    argv: list[str] | None = None,
) -> list[str]:
    """Resolve a packaged relative file flag without changing user paths.

    Editable source invocations normally resolve ``configs/...`` from the
    repository root.  A console script installed from a wheel can be invoked
    from any directory, so fall back to the installed package root only when
    the caller's relative path does not already exist.
    """

    arguments = sys.argv if argv is None else argv
    option = f'--{flag_name}'
    root = Path(package_root)

    def resolved(value: str) -> str:
        if not value or (value.startswith('<') and value.endswith('>')):
            return value
        path = Path(value).expanduser()
        if path.is_absolute() or path.exists():
            return str(path)
        packaged = root / path
        return str(packaged) if packaged.is_file() else value

    for index, argument in enumerate(arguments):
        prefix = option + '='
        if argument.startswith(prefix):
            arguments[index] = prefix + resolved(argument[len(prefix) :])
        elif argument == option and index + 1 < len(arguments):
            arguments[index + 1] = resolved(arguments[index + 1])
    return arguments


def _plain(value: Any) -> Any:
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _coerce_like(value: Any, template: Any, *, key: str) -> Any:
    """Preserve tuple-valued research settings when YAML uses a list."""

    if isinstance(template, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f'{key} must be a sequence, got {type(value).__name__}.')
        return tuple(value)
    return value


def _set_known(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split('.')
    if not parts or any(not part for part in parts):
        raise ValueError(f'Invalid configuration key {key!r}.')
    target: dict[str, Any] = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], Mapping):
            raise KeyError(f'Unknown configuration key {key!r}.')
        target = target[part]
    leaf = parts[-1]
    if leaf not in target:
        raise KeyError(f'Unknown configuration key {key!r}.')
    target[leaf] = _coerce_like(value, target[leaf], key=key)


def _merge_known(target: dict[str, Any], overlay: Mapping[str, Any], prefix: str = '') -> None:
    for key, value in overlay.items():
        key = str(key)
        dotted = f'{prefix}.{key}' if prefix else key
        if key not in target:
            raise KeyError(f'Unknown configuration key {dotted!r}.')
        if isinstance(value, Mapping):
            if not isinstance(target[key], Mapping):
                raise TypeError(f'{dotted} is not a mapping and cannot be overlaid.')
            child = dict(target[key])
            _merge_known(child, value, dotted)
            target[key] = child
        else:
            target[key] = _coerce_like(value, target[key], key=dotted)


def resolve_config(
    base: Mapping[str, Any],
    *,
    yaml_path: str | Path | None = None,
    cli_overrides: Sequence[str] = (),
) -> ml_collections.ConfigDict:
    """Resolve one config as preset/default, then YAML, then CLI overrides."""

    resolved = _plain(base)
    if not isinstance(resolved, dict):
        raise TypeError('Base configuration must be a mapping.')

    if yaml_path is not None and str(yaml_path).strip():
        path = Path(yaml_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'Configuration YAML not found: {path}')
        with path.open(encoding='utf-8') as file:
            overlay = yaml.safe_load(file)
        if overlay is None:
            overlay = {}
        if not isinstance(overlay, Mapping):
            raise TypeError('Configuration YAML root must be a mapping.')
        _merge_known(resolved, overlay)

    for raw in cli_overrides:
        if '=' not in raw:
            raise ValueError(
                f'CLI config override must use key=value syntax, got {raw!r}.'
            )
        key, text = raw.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f'CLI config override has an empty key: {raw!r}.')
        _set_known(resolved, key, yaml.safe_load(text))

    return ml_collections.ConfigDict(resolved)


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _plain(config), sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def write_resolved_config(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    sources: Mapping[str, Any] | None = None,
) -> None:
    """Write the exact effective config and its content hash as JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'config': _plain(config),
        'config_sha256': config_sha256(config),
        'precedence': ['preset/default', 'legacy_agent_cli', 'yaml', 'cli'],
        'sources': _plain(sources or {}),
    }
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write('\n')
    temporary.replace(output)


__all__ = [
    'config_sha256',
    'normalize_packaged_file_flag',
    'resolve_config',
    'write_resolved_config',
]
