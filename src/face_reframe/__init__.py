"""Face-centred automatic reframing for DaVinci Resolve (a Smart Reframe alternative)."""

__all__ = ["CropPlan", "plan_crops", "parse_aspect", "TrackSettings", "build_camera_path"]
__version__ = "0.1.0"


def __getattr__(name):
    """Lazy re-exports (PEP 562): audio-only tools then don't pull in cv2."""
    if name in ("CropPlan", "plan_crops", "parse_aspect"):
        from . import geometry
        return getattr(geometry, name)
    if name in ("TrackSettings", "build_camera_path"):
        from . import track
        return getattr(track, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
