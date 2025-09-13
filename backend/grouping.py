from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Union
import json

from .config import STORE_DIR


_CFG_PATH = STORE_DIR / "grouping.json"


@dataclass
class GroupingConfig:
    roles: List[str]
    # role -> list of suffix patterns (case-insensitive), matched with endswith
    suffix_map: Dict[str, List[str]]
    # If True, a "complete" group requires all roles to exist on disk
    require_all_roles: bool = False
    # Hint for UI: how many roles constitute a typical unit item
    unit_size: int = 3

    def to_dict(self) -> Dict[str, object]:
        return {
            "roles": list(self.roles),
            "suffix_map": {k: list(v) for k, v in self.suffix_map.items()},
            "require_all_roles": bool(self.require_all_roles),
            "unit_size": int(self.unit_size),
        }


def _default_config() -> GroupingConfig:
    return GroupingConfig(
        roles=["target", "ref", "diff"],
        suffix_map={
            "target": ["_target.fits"],
            "ref": ["_ref.fits"],
            "diff": ["_diff.fits"],
        },
        require_all_roles=False,
        unit_size=3,
    )


def load_config() -> GroupingConfig:
    try:
        if _CFG_PATH.exists():
            data = json.loads(_CFG_PATH.read_text(encoding="utf-8"))
            roles = data.get("roles") or []
            suffix_map = data.get("suffix_map") or {}
            require_all_roles = bool(data.get("require_all_roles", False))
            unit_size = int(data.get("unit_size", len(roles) or 3))
            # basic validation / fallback
            if not roles or not isinstance(roles, list):
                return _default_config()
            # ensure every role has suffix list
            for r in roles:
                if r not in suffix_map or not isinstance(suffix_map[r], list) or len(suffix_map[r]) == 0:
                    # fallback to default FITS naming if missing
                    default_suf = f"_{r}.fits" if r in {"target", "ref", "diff"} else f"_{r}"
                    suffix_map[r] = [default_suf]
            return GroupingConfig(roles=roles, suffix_map=suffix_map, require_all_roles=require_all_roles, unit_size=unit_size)
    except Exception:
        pass
    return _default_config()


def save_config(cfg: GroupingConfig) -> None:
    _CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CFG_PATH.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def get_config_dict() -> Dict[str, object]:
    return load_config().to_dict()


def set_config(payload: Dict[str, object]) -> Dict[str, object]:
    roles = payload.get("roles") or []
    suffix_map = payload.get("suffix_map") or {}
    require_all_roles = bool(payload.get("require_all_roles", False))
    unit_size = int(payload.get("unit_size", len(roles) or 3))
    if not isinstance(roles, list) or len(roles) == 0:
        raise ValueError("roles must be a non-empty list")
    if not isinstance(suffix_map, dict):
        raise ValueError("suffix_map must be a dict[role -> list[str]]")
    # normalize and validate
    norm_map: Dict[str, List[str]] = {}
    for r in roles:
        vals = suffix_map.get(r)
        if not isinstance(vals, list) or len(vals) == 0:
            # default to _{role}.fits if missing
            vals = [f"_{r}.fits"]
        norm_map[r] = [str(s).lower() for s in vals]
    cfg = GroupingConfig(roles=roles, suffix_map=norm_map, require_all_roles=require_all_roles, unit_size=unit_size)
    save_config(cfg)
    return cfg.to_dict()


def match_role_and_base(p: Path, cfg: Optional[GroupingConfig] = None) -> Tuple[Optional[str], str]:
    cfg = cfg or load_config()
    name_lower = p.name.lower()
    for role in cfg.roles:
        for suf in cfg.suffix_map.get(role, []):
            s = suf.lower()
            if name_lower.endswith(s):
                base = p.name[: -len(s)]
                return role, base
    # fallback: no recognized role suffix; use stem
    return None, p.stem


def expected_paths(parent: Path, base: str, cfg: Optional[GroupingConfig] = None) -> Dict[str, Path]:
    cfg = cfg or load_config()
    out: Dict[str, Path] = {}
    for role in cfg.roles:
        # First suffix is the canonical expected filename
        suf_list = cfg.suffix_map.get(role, [])
        suf = suf_list[0] if suf_list else f"_{role}"
        out[role] = parent / f"{base}{suf}"
    return out


def resolve_existing_role_file(parent: Path, base: str, role: str, cfg: Optional[GroupingConfig] = None) -> Optional[Path]:
    cfg = cfg or load_config()
    for suf in cfg.suffix_map.get(role, []):
        cand = parent / f"{base}{suf}"
        if cand.exists():
            return cand
    return None


def group_key_for_path(p: Path, cfg: Optional[GroupingConfig] = None) -> Tuple[Path, str]:
    cfg = cfg or load_config()
    _, base = match_role_and_base(p, cfg)
    return p.parent, base


def group_items(paths_or_items: Iterable[Union[Path, object]], get_path=lambda x: Path(x.path) if hasattr(x, "path") else x, cfg: Optional[GroupingConfig] = None) -> Dict[Tuple[Path, str], List[object]]:
    cfg = cfg or load_config()
    from collections import defaultdict
    groups: Dict[Tuple[Path, str], List[object]] = defaultdict(list)
    for it in paths_or_items:
        p = get_path(it)
        parent, base = group_key_for_path(p, cfg)
        groups[(parent, base)].append(it)
    return groups


