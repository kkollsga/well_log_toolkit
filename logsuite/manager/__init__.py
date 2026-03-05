"""
Multi-well management and orchestration.

Submodules
----------
data_manager : WellDataManager class for loading and managing wells
proxy : Property proxy classes for broadcasting operations across wells
"""

from .data_manager import WellDataManager
from .proxy import _ManagerMultiPropertyProxy, _ManagerPropertyProxy

__all__ = [
    "WellDataManager",
    "_ManagerPropertyProxy",
    "_ManagerMultiPropertyProxy",
]
