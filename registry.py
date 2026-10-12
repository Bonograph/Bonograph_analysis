"""
Maps an analysis's key (a plain identifier, the name of its backend
module) to the backend function that actually computes it. What the user
reads comes from analysis/catalog.py, never from the key. Each backend lives in its own file under
analysis/backends/ and registers itself with ``@register("module_name")``
so this module never needs to know the full list up front.
"""

from __future__ import annotations

import re

import inspect
from typing import Any, Callable

import numpy as np

from .backend_types import AnalysisError, AnalysisResult
from .data_source import SheetData

Backend = Callable[..., AnalysisResult]
OptionsDialogFactory = Callable[..., Any]  # (parent) -> a QDialog with .exec()

_REGISTRY: dict[str, Backend] = {}
_OPTIONS_DIALOGS: dict[str, OptionsDialogFactory] = {}


def register(name: str):
    """Decorator: ``@register("descriptive_statistics")`` on a backend
    function of signature ``(SheetData) -> AnalysisResult`` or, for a
    backend with user-configurable options, ``(SheetData, options) ->
    AnalysisResult`` (paired with ``register_options_dialog``)."""

    def decorator(fn: Backend) -> Backend:
        _REGISTRY[name] = fn
        return fn

    return decorator


def register_options_dialog(name: str):
    """Decorator for a factory ``(parent) -> QDialog`` that lets the user
    configure ``name`` before it runs - e.g. which statistics to compute,
    how to treat replicate subcolumns. The dialog must expose the chosen
    settings as a ``.options`` attribute once accepted. Optional: most
    analyses don't need one and just run with their built-in defaults."""

    def decorator(factory: OptionsDialogFactory) -> OptionsDialogFactory:
        _OPTIONS_DIALOGS[name] = factory
        return factory

    return decorator


def get_options_dialog_factory(name: str) -> OptionsDialogFactory | None:
    """The settings panel for ``name``: drawn from its spec (analysis/studio),
    or - for anything without one - the panel its backend registered."""
    _ensure_backends_loaded()
    try:
        from . import studio

        if studio.has_spec(name):
            return lambda parent=None, _n=name: studio.panel_for(_n, parent=parent)
    except Exception:  # noqa: BLE001 - a broken spec falls back to the registered panel
        import traceback

        traceback.print_exc()
    return _OPTIONS_DIALOGS.get(name)


def available_analyses() -> list[str]:
    _ensure_backends_loaded()
    return sorted(_REGISTRY)


def has_backend(name: str) -> bool:
    _ensure_backends_loaded()
    return name in _REGISTRY


_loaded = False


def _ensure_backends_loaded() -> None:
    # Imported lazily (not at module import time) so importing the
    # registry alone never pulls in the numerical libraries just to check a name.
    global _loaded
    if not _loaded:
        from . import backends  # noqa: F401 - import registers every backend

        _loaded = True


def preload_backends_async() -> None:
    """Kick off ``_ensure_backends_loaded`` on a background thread.

    Registering all 70 backends pulls in the whole analysis package (and
    on the very first call in a process, big chunks of SciPy with
    it) - roughly a one-time 1-1.5s hit. Calling this once right after the
    main window is shown pays that cost while the user is looking at an
    empty sheet instead of during their first "Analyze Data" click.
    CPython's import lock makes this safe to race against a real click
    that triggers the same import from the main thread.
    """
    import threading

    def _worker() -> None:
        try:
            _ensure_backends_loaded()
        except Exception:
            pass  # a real call site will surface any genuine import error

    threading.Thread(target=_worker, name="backend-preload", daemon=True).start()


def prepare_settings(dialog, sheet: SheetData | None) -> None:
    """Give a settings panel the sheet it is about to run on.

    Some settings list the sheet's own columns (Correlation, ROC,
    Bland-Altman...) and some switch choices on or off by the sheet's shape
    (Two-way ANOVA with or without repeats). Both the Analyze window and the
    stand-alone settings path call this, so the two can't disagree.
    """
    if sheet is None:
        return
    if hasattr(dialog, "set_variable_names"):
        dialog.set_variable_names(["X"] + list(sheet.group_labels()))
    if hasattr(dialog, "set_sheet_data"):
        dialog.set_sheet_data(sheet)


def display_name(name: str) -> str:
    """What the user reads for the analysis registered as ``name``."""
    try:
        from .catalog import BY_KEY

        entry = BY_KEY.get(name)
    except Exception:  # noqa: BLE001
        entry = None
    return entry.title if entry is not None else name.replace("_", " ").capitalize()


def run_analysis(name: str, data: SheetData, options: Any = None) -> AnalysisResult:
    """Run the backend registered for ``name``, or return a clear
    "not implemented yet" result if nothing is registered for it - this
    never raises just because a menu entry doesn't have code behind it
    yet, only if the backend itself fails on the given data (callers
    should still expect AnalysisError/other exceptions from real runs).

    ``options`` (from that analysis's options dialog, if it has one) is
    only forwarded to backends that actually declare a second parameter,
    so plain single-argument backends are unaffected."""
    _ensure_backends_loaded()
    backend = _REGISTRY.get(name)
    if backend is None:
        return AnalysisResult(
            title=display_name(name),
            summary="This analysis isn't wired up to a backend yet.",
            notes=[
                "Its entry in the picker works, but the statistical "
                "routine behind it hasn't been implemented in "
                "analysis/backends/ yet.",
            ],
        )
    try:
        if options is not None and _accepts_options(backend):
            result = backend(data, options)
        else:
            result = backend(data)
    except AnalysisError:
        raise
    except MemoryError:
        raise AnalysisError("This sheet is too large for this analysis to hold in memory. "
                            "Try it on fewer rows or columns.") from None
    except (OverflowError, FloatingPointError):
        raise AnalysisError("Some values are too extreme to calculate with (numbers this large or small "
                            "overflow). Rescale them - e.g. a log transform or different units - and run it "
                            "again.") from None
    except np.linalg.LinAlgError:
        raise AnalysisError("The calculation couldn't be solved for this data - usually because a column "
                            "has no variation, or columns repeat each other. Check for constant or "
                            "duplicated columns.") from None
    # results are named as the Analyze window names the analysis
    if isinstance(result, AnalysisResult):
        result.title = display_name(name)
        _tidy_undefined(result)
    return result


_UNDEFINED_SPAN = re.compile(r"-?(?:nan|inf) to -?(?:nan|inf)")
_UNDEFINED = re.compile(r"(?<![\w.])-?nan(?![\w])")


def _tidy_undefined(result: AnalysisResult) -> None:
    """A number that could not be worked out (no spread, nothing to divide
    by) reaches here as "nan". Nobody should be shown that word: a table
    cell gets a dash, a sentence gets "n/a"."""
    for table in result.tables:
        for row in table.rows:
            for i, cell in enumerate(row):
                if isinstance(cell, str) and ("nan" in cell or "inf to" in cell):
                    cell = _UNDEFINED_SPAN.sub("–", cell)
                    row[i] = "" if cell.strip() in ("nan", "-nan") else _UNDEFINED.sub("–", cell)
    if "nan" in result.summary or "inf to" in result.summary:
        result.summary = _UNDEFINED.sub("n/a", _UNDEFINED_SPAN.sub("n/a", result.summary))
    for i, note in enumerate(result.notes):
        if "nan" in note:
            result.notes[i] = _UNDEFINED.sub("n/a", _UNDEFINED_SPAN.sub("n/a", note))


def _accepts_options(backend: Backend) -> bool:
    try:
        params = inspect.signature(backend).parameters
    except (TypeError, ValueError):
        return False
    return len(params) >= 2
