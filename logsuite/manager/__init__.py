"""
Multi-well management and orchestration.

Submodules
----------
data_manager : WellDataManager class for loading and managing wells
proxy : Property proxy classes for broadcasting operations across wells
view : ManagerView read-only subset of a manager
"""

from .data_manager import WellDataManager
from .proxy import _ManagerMultiPropertyProxy, _ManagerPropertyProxy
from .view import ManagerView

__all__ = [
    "WellDataManager",
    "ManagerView",
    "_ManagerPropertyProxy",
    "_ManagerMultiPropertyProxy",
]
