"""Evaluation units and spatial aggregation: grid cells, polygons (admin units) and regions, one implementation.

`ids` names units as strings at the edges, `regrid` holds the sparse transform and the weights that build it,
`units` the table of units and the transform of each source grid onto them. `axes` tells grid cells (``lat``, ``lon``)
from evaluation units (``unit``) in arrays and tables, `masks` builds the cell and polygon masks, `select` picks cells
and units from the config and `onto_units` moves arrays onto units. The package re-exports the main classes.
"""
from .ids import GridIdConvention
from .regrid import SparseTransform
from .units import EvaluationUnits, UnitsSpec

__all__ = ["EvaluationUnits", "GridIdConvention", "SparseTransform", "UnitsSpec"]
