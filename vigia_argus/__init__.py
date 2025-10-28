# vigia_argus/__init__.py
"""
VIGIA Argus-V8X plugin
- Registers custom layers (SimAM, SwinBlock) into Ultralytics namespaces
- Provides a helper to locate packaged YAMLs
Usage:
    import vigia_argus
    from ultralytics import YOLO
    m = YOLO(vigia_argus.model_yaml("argus_v8x.yaml"))
"""

from __future__ import annotations
from importlib.resources import files

# 1) Import your custom blocks (must exist in the same package)
from .argus_blocks import SimAM, SwinBlock

__all__ = ["SimAM", "SwinBlock", "model_yaml", "ensure_patched"]

_PATCH_DONE = False


def _append_to_all(obj, names: list[str]) -> None:
    """Safely append exported names to a module's __all__ if present."""
    try:
        cur = getattr(obj, "__all__", ())
        obj.__all__ = tuple(list(cur) + names)  # tuple to tuple (safe across versions)
    except Exception:
        pass


def ensure_patched() -> None:
    """
    Idempotently patch Ultralytics namespaces so YAML can resolve SimAM/SwinBlock.
    Call once on import; safe to call again.
    """
    global _PATCH_DONE
    if _PATCH_DONE:
        return
    # a) Base building blocks namespace
    try:
        import ultralytics.nn.modules as mods
        mods.SimAM = SimAM
        mods.SwinBlock = SwinBlock
        _append_to_all(mods, ["SimAM", "SwinBlock"])
    except Exception:
        # If Ultralytics isn't installed yet, user will patch after install
        return

    # b) Tasks namespace (parse_model() looks up symbols here)
    try:
        import ultralytics.nn.tasks as tasks
        tasks.SimAM = SimAM
        tasks.SwinBlock = SwinBlock
        _append_to_all(tasks, ["SimAM", "SwinBlock"])
    except Exception:
        # Some older/newer versions may structure tasks differently; ignore silently
        pass

    _PATCH_DONE = True


# Patch immediately on import so users don't have to remember to call anything
ensure_patched()


def model_yaml(name: str = "argus_v8x.yaml") -> str:
    """
    Return an absolute path to a packaged YAML under vigia_argus/cfg/.
    Example:
        YOLO(model_yaml("argus_v8x.yaml"))
    """
    return str(files("vigia_argus.cfg") / name)