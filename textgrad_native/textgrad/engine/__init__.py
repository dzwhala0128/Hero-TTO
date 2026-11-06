"""HTTP-only delivery registry. Native graph algorithms are unchanged."""
from .base import EngineLM, CachedEngine

def get_engine(engine_name, **kwargs):
    raise ValueError("Pass an explicit pipeline.engine.TraceEngine; vendor factories are excluded")

def validate_multimodal_engine(engine):
    raise ValueError("This reproduction package supports text only")
