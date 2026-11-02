"""Evaluation units and spatial aggregation: grid cells, polygons (admin units) and regions, one implementation.

`ids` names units as strings at the edges, `regrid` holds the sparse transform and the weights that build it,
`units` the table of units and the transform of each source grid onto them.
"""
from .ids import GridIdConvention
from .regrid import SparseTransform
from .units import AreaError, EvaluationUnits

__all__ = ["AreaError", "EvaluationUnits", "GridIdConvention", "SparseTransform"]
