"""Analysis settings, redesigned: specs, the panel that draws them, and the
report style every analysis shares. See spec.py for how a spec works."""

from __future__ import annotations

from .spec import SettingsPanel, Spec

_SPECS: dict[str, Spec] | None = None


def spec_map() -> dict[str, Spec]:
    """Every analysis's spec, by registry key."""
    global _SPECS
    if _SPECS is None:
        from .specs import all_specs

        _SPECS = {spec.key: spec for spec in all_specs()}
    return _SPECS


def has_spec(key: str) -> bool:
    return key in spec_map()


def panel_for(key: str, parent=None, title: str = "", embedded: bool = False) -> SettingsPanel | None:
    spec = spec_map().get(key)
    if spec is None:
        return None
    if not title:
        try:
            from ..catalog import BY_KEY

            title = BY_KEY[key].title if key in BY_KEY else ""
        except Exception:  # noqa: BLE001
            title = ""
    return SettingsPanel(spec, title=title, parent=parent, embedded=embedded)
