"""Shared kernel parameters; executable IR lives in region and extended."""
from .ir import ComputeKind, LoopKind, DataType, TileKernel

__all__ = ['ComputeKind', 'LoopKind', 'DataType', 'TileKernel']
