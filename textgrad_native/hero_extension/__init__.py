"""Thin HerO adapters around the vendored, source-native TextGrad package."""

from .hero_objective import build_objective
from .hero_solution import parse_solution, serialize_solution
from .hero_validator import validate_solution

__all__ = ["build_objective", "parse_solution", "serialize_solution", "validate_solution"]
