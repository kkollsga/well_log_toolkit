"""
ManagerView — a filtered, read-only view over a WellDataManager.

Returned by :meth:`WellDataManager.filter`. Exposes the same property and
well attribute-access surface as the manager, restricted to a subset of
wells and (optionally) to specific property values.

The view duck-types the manager via its ``_wells`` attribute so that
existing property proxies iterate it transparently and require no changes.
When value filters are present, attribute access returns a thin wrapper
proxy that post-filters ``.data()`` outputs by those values.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..utils import suggest_similar_names
from .proxy import _ManagerMultiPropertyProxy, _ManagerPropertyProxy

if TYPE_CHECKING:
    import pandas as pd

    from .data_manager import WellDataManager


_STAT_METHODS_BLOCKED_BY_WHERE: frozenset[str] = frozenset(
    {
        "mean",
        "median",
        "mode",
        "min",
        "max",
        "std",
        "sum",
        "percentile",
        "stats",
        "sums_avg",
    }
)


def _coerce_allowed(values: Any) -> list:
    """Accept a scalar or iterable as a list of allowed values."""
    if isinstance(values, (list, tuple, set)):
        return list(values)
    return [values]


class ManagerView:
    """
    Read-only subset view of a :class:`WellDataManager`.

    A view holds a reference to the underlying manager, a filtered
    ``_wells`` dict, and an optional ``value_filters`` dict mapping
    column names to allowed values. Property proxies created from the
    view iterate only the subset; when value filters are present, the
    ``.data()`` output is post-filtered to keep only matching rows.

    Views are immutable: :meth:`filter` returns a new, further-narrowed
    view rather than mutating the original.

    Parameters
    ----------
    manager : WellDataManager
        Underlying manager. The view does not own data.
    well_keys : Iterable[str], optional
        Manager dict keys (e.g. ``"well_Well_A"``) to include. ``None``
        includes every well in the manager.
    value_filters : dict[str, list], optional
        Mapping of column name -> allowed values. Applied to ``.data()``
        outputs as a row mask. Keys typically match a filter property name
        (e.g. ``"Zone"``); the special key ``"well"`` is handled by the
        ``WellDataManager.filter`` / :meth:`filter` interface and not
        stored here. Stat methods (mean, stats, ...) raise
        :class:`NotImplementedError` while value filters are active —
        use ``.data()`` and compute statistics yourself, or
        ``.filter("Zone")`` for grouped stats.

    Examples
    --------
    >>> view = manager.filter(wells=["Well_A", "Well_B"])
    >>> view.PHIE.mean()
    {'well_Well_A': 0.182, 'well_Well_B': 0.205}
    >>> sub = manager.filter(where={"Zone": "Reservoir"})
    >>> sub.PHIE.data().head()
    """

    def __init__(
        self,
        manager: WellDataManager,
        well_keys: Iterable[str] | None = None,
        value_filters: dict[str, list] | None = None,
    ):
        object.__setattr__(self, "_source_manager", manager)
        if well_keys is None:
            wells = dict(manager._wells)
        else:
            wells = {k: manager._wells[k] for k in well_keys if k in manager._wells}
        object.__setattr__(self, "_wells", wells)
        object.__setattr__(self, "_value_filters", dict(value_filters or {}))

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        if name.startswith("well_"):
            if name in self._wells:
                return self._wells[name]
            available = list(self._wells.keys())
            suggestions = suggest_similar_names(name, available)
            msg = f"Well '{name}' not found in this view."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}?"
            msg += f" Available: {', '.join(available) or 'none'}"
            raise AttributeError(msg)

        proxy = _ManagerPropertyProxy(self, name)
        if not self._value_filters:
            return proxy
        return _ValueFilteringProxy(proxy, self._value_filters)

    def properties(self, property_names: list[str]):
        """Multi-property proxy scoped to this view."""
        proxy = _ManagerMultiPropertyProxy(self, property_names)
        if not self._value_filters:
            return proxy
        return _ValueFilteringProxy(proxy, self._value_filters)

    def filter(
        self,
        *,
        wells: Iterable[str] | str | None = None,
        where: dict[str, Any] | None = None,
    ) -> ManagerView:
        """
        Narrow the view by selecting wells and/or restricting property values.

        Returns a new view; the original is unchanged.

        Parameters
        ----------
        wells : str or list of str, optional
            Well names (original, sanitized, or manager dict key) to keep.
            Names not present in the current view are ignored silently.
        where : dict, optional
            Mapping of column name -> allowed value(s). Each value can be
            a scalar or list. The special key ``"well"`` selects wells
            (intersected with ``wells``). Other keys post-filter
            ``.data()`` outputs to rows where the named column is in the
            allowed set. New filters compose with any already on the view.

        Returns
        -------
        ManagerView
            Further-filtered view.
        """
        if wells is None and where is None:
            return self

        new_well_keys = list(self._wells.keys())
        new_value_filters = dict(self._value_filters)

        if wells is not None:
            wells_list = [wells] if isinstance(wells, str) else list(wells)
            new_well_keys = self._resolve_well_keys(wells_list)

        if where is not None:
            for key, vals in where.items():
                if key == "well":
                    well_names = _coerce_allowed(vals)
                    new_well_keys = self._resolve_well_keys(well_names)
                else:
                    new_value_filters[key] = _coerce_allowed(vals)

        return ManagerView(self._source_manager, new_well_keys, new_value_filters)

    def _resolve_well_keys(self, names: list[str]) -> list[str]:
        """Map user-supplied names to manager dict keys, restricted to the current subset."""
        out: list[str] = []
        for n in names:
            for key, well in self._wells.items():
                if n == key or n == well.name or n == getattr(well, "sanitized_name", None):
                    if key not in out:
                        out.append(key)
                    break
        return out

    @property
    def wells(self) -> list[str]:
        """Original well names included in this view."""
        return [w.name for w in self._wells.values()]

    def __repr__(self) -> str:
        n = len(self._wells)
        names = ", ".join(w.name for w in self._wells.values())
        wf = f", where={self._value_filters}" if self._value_filters else ""
        return f"ManagerView({n} well{'s' if n != 1 else ''}: [{names}]{wf})"

    def __len__(self) -> int:
        return len(self._wells)

    def __iter__(self):
        return iter(self._wells.values())

    def __getitem__(self, name: str):
        """Get a well by name (original, sanitized, or sanitized-with-prefix)."""
        if not isinstance(name, str):
            raise TypeError(f"ManagerView indices must be str, got {type(name).__name__}")
        if name in self._wells:
            return self._wells[name]
        for well in self._wells.values():
            if well.name == name:
                return well
        from ..utils import sanitize_well_name

        sanitized_key = f"well_{sanitize_well_name(name)}"
        if sanitized_key in self._wells:
            return self._wells[sanitized_key]
        available = [w.name for w in self._wells.values()]
        msg = f"Well '{name}' not in this view."
        msg += f" Available: {available}" if available else " View is empty."
        raise KeyError(msg)

    def __contains__(self, name: str) -> bool:
        if name in self._wells:
            return True
        return any(w.name == name for w in self._wells.values())


class _ValueFilteringProxy:
    """
    Wraps a property proxy to apply value filters on ``.data()`` outputs.

    Used internally by :class:`ManagerView` when ``where=`` filters are
    active. ``.data()`` calls run on the underlying proxy and the result
    is masked to rows matching every filter. ``.filter`` /
    ``filter_intervals`` chain on the underlying proxy and return a new
    wrapper preserving the value filters. Statistical methods raise
    :class:`NotImplementedError` — use ``.data()`` and compute externally,
    or use ``.filter("Zone")`` for grouped stats.
    """

    def __init__(self, proxy, value_filters: dict[str, list]):
        self._proxy = proxy
        self._value_filters = dict(value_filters)

    def data(self, *args, **kwargs) -> pd.DataFrame:
        df = self._proxy.data(*args, **kwargs)
        if df.empty:
            return df
        for col, allowed in self._value_filters.items():
            if col in df.columns:
                df = df[df[col].isin(allowed)]
        return df.reset_index(drop=True)

    def filter(self, *args, **kwargs):
        new_proxy = self._proxy.filter(*args, **kwargs)
        return _ValueFilteringProxy(new_proxy, self._value_filters)

    def filter_intervals(self, *args, **kwargs):
        new_proxy = self._proxy.filter_intervals(*args, **kwargs)
        return _ValueFilteringProxy(new_proxy, self._value_filters)

    def fit(self, model, **kwargs):
        """Fit a regression on the where-filtered data; returns a RegressionFit."""
        from ..analysis.regression_fit import RegressionFit

        property_names = getattr(self._proxy, "_property_names", None)
        if property_names is None or len(property_names) != 2:
            raise ValueError(
                "fit() requires a multi-property proxy with exactly 2 properties; "
                "use manager.properties([x, y]).fit(...)."
            )
        df = self.data()
        if df.empty:
            raise ValueError("No data available to fit; pooled DataFrame is empty.")
        x_col, y_col = property_names
        model.fit(df[x_col].to_numpy(), df[y_col].to_numpy())
        return RegressionFit(
            model,
            name=kwargs.pop("name", f"{x_col}-{y_col}"),
            **kwargs,
        )

    def __getattr__(self, name: str):
        if name in _STAT_METHODS_BLOCKED_BY_WHERE:
            raise NotImplementedError(
                f"{name}() is not supported on a view with value filters "
                f"(where=). Use .data() and compute statistics externally, "
                f"or use .filter('property_name') for grouped stats."
            )
        return getattr(self._proxy, name)
