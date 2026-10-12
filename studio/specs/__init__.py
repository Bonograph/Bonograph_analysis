"""Every analysis's settings spec, gathered by goal."""

from __future__ import annotations


def all_specs() -> list:
    import importlib

    out = []
    for name in ("compare", "describe", "fit", "relate", "counts", "time_to_event", "agreement", "patterns",
                 "reshape", "make"):
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.{name}":
                continue  # a goal not written yet
            raise
        out.extend(module.SPECS)
    return out
