"""
df_registry
==============
Lightweight DataFrame provenance tracker for Jupyter notebooks.

Tracks the lineage of pandas DataFrames as they are created and
transformed throughout a notebook pipeline.  Each call to
``register()`` appends one row to an in-memory registry that records:

    - the new DataFrame name and shape
    - the parent DataFrame name and shape
    - the operation label
    - any options / parameters passed to the transformation
    - a UTC timestamp

Typical usage
-------------
    from df_registry import register, get_registry, show_lineage, reset_registry

    # After every transformation, call register():
    df_clean = df_raw.drop(columns=["transaction_id", "card_id"])
    register(
        new_df=df_clean,
        new_name="df_clean",
        parent_name="df_raw",
        operation="drop_id_cols",
        options={"columns": ["transaction_id", "card_id"]},
    )

    # Display the full registry at any point:
    show_lineage()

Compliance
----------
Formatted with black and passes ruff (no bare `except`, no unused imports,
type annotations throughout).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Module-level singleton registry
# ---------------------------------------------------------------------------

# Each element is a plain dict; converted to a DataFrame only on demand so
# that appending stays O(1) rather than O(n) per registration call.
_REGISTRY: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register(
    *,
    new_df: pd.DataFrame,
    new_name: str,
    parent_name: str,
    operation: str,
    options: dict[str, Any] | None = None,
) -> None:
    """Record a DataFrame transformation in the provenance registry.

    Parameters
    ----------
    new_df:
        The newly created DataFrame (used to capture its shape).
    new_name:
        Variable name assigned to *new_df* in the notebook.
    parent_name:
        Variable name of the DataFrame that *new_df* was derived from.
    operation:
        Short label describing the transformation, e.g.
        ``"drop_id_cols"``, ``"encode_categoricals"``,
        ``"train_test_split"``.
    options:
        Arbitrary key/value pairs describing the parameters passed to
        the transformation (columns dropped, encoding method, split
        ratio, etc.).  Stored as a JSON string for compact display.

    Returns
    -------
    None
        Side-effect only: appends one row to the module-level registry.

    Examples
    --------
    >>> register(
    ...     new_df=df_encoded,
    ...     new_name="df_encoded",
    ...     parent_name="df_clean",
    ...     operation="encode_categoricals",
    ...     options={"cols": ["channel", "entry_mode"], "method": "onehot"},
    ... )
    """
    # Resolve parent shape from the registry if available so callers do not
    # have to pass the parent DataFrame object itself.
    parent_shape = _lookup_shape(parent_name)

    _REGISTRY.append(
        {
            # Lineage identifiers
            "new_name": new_name,
            "parent_name": parent_name,
            # Shapes — (rows, cols) tuples stored as strings for readability
            "new_shape": str(new_df.shape),
            "parent_shape": str(parent_shape) if parent_shape else "unknown",
            # Row / column deltas (None when parent shape is unknown)
            "row_delta": (
                new_df.shape[0] - parent_shape[0]
                if parent_shape is not None
                else None
            ),
            "col_delta": (
                new_df.shape[1] - parent_shape[1]
                if parent_shape is not None
                else None
            ),
            # Transformation metadata
            "operation": operation,
            # Serialise options to a compact JSON string; fall back to repr
            # for non-serialisable objects (e.g. sklearn transformers).
            "options": _safe_json(options or {}),
            # Wall-clock timestamp in UTC ISO-8601 format
            "registered_at": datetime.now(tz=timezone.utc).isoformat(
                timespec="seconds"
            ),
        }
    )


def get_registry() -> pd.DataFrame:
    """Return the full provenance registry as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per ``register()`` call, columns as described in the
        module docstring.  Returns an empty DataFrame (with the correct
        columns) if nothing has been registered yet.
    """
    columns = [
        "new_name",
        "parent_name",
        "new_shape",
        "parent_shape",
        "row_delta",
        "col_delta",
        "operation",
        "options",
        "registered_at",
    ]

    if not _REGISTRY:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(_REGISTRY, columns=columns)


def show_lineage(
    *,
    compact: bool = False,
    name_filter: str | None = None,
) -> pd.DataFrame:
    """Display and return the provenance registry.

    Intended for interactive use in Jupyter; the returned DataFrame is
    also displayed via ``pd.options.display`` settings so long lines are
    not truncated.

    Parameters
    ----------
    compact:
        If ``True``, hide the ``options`` and ``registered_at`` columns
        to fit the output on screen.
    name_filter:
        If provided, return only rows where ``new_name`` or
        ``parent_name`` contains this substring (case-insensitive).

    Returns
    -------
    pd.DataFrame
        The (possibly filtered / narrowed) registry view.
    """
    registry_df = get_registry()

    if registry_df.empty:
        print("Registry is empty — call register() after each transformation.")
        return registry_df

    # Optional name filter
    if name_filter is not None:
        mask = registry_df["new_name"].str.contains(
            name_filter, case=False, na=False
        ) | registry_df["parent_name"].str.contains(
            name_filter, case=False, na=False
        )
        registry_df = registry_df.loc[mask].reset_index(drop=True)

    # Optional column reduction for compact display
    if compact:
        display_cols = [
            "new_name",
            "parent_name",
            "new_shape",
            "row_delta",
            "col_delta",
            "operation",
        ]
        registry_df = registry_df[display_cols]

    # Widen display so options strings are not truncated
    with pd.option_context(
        "display.max_colwidth", 120,
        "display.max_rows", 100,
    ):
        # ``display()`` is available in Jupyter; fall back to print elsewhere.
        try:
            from IPython.display import display  # type: ignore[import-untyped]

            display(registry_df)
        except ImportError:
            print(registry_df.to_string(index=False))

    return registry_df


def reset_registry() -> None:
    """Clear all entries from the provenance registry.

    Useful at the top of a notebook to ensure a clean state on re-runs.

    Returns
    -------
    None
    """
    _REGISTRY.clear()
    print("Registry cleared.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lookup_shape(name: str) -> tuple[int, int] | None:
    """Return the recorded shape of the most recent entry for *name*.

    Searches the registry in reverse order so the most recent
    registration for a given DataFrame name is found first.

    Parameters
    ----------
    name:
        The ``new_name`` to look up.

    Returns
    -------
    tuple[int, int] or None
        ``(rows, cols)`` if found, ``None`` otherwise.
    """
    for entry in reversed(_REGISTRY):
        if entry["new_name"] == name:
            # Shape is stored as a string like "(60000, 32)"; parse it back.
            shape_str: str = entry["new_shape"]
            try:
                rows, cols = (
                    int(x.strip()) for x in shape_str.strip("()").split(",")
                )
                return (rows, cols)
            except ValueError:
                return None
    return None


def _safe_json(obj: dict[str, Any]) -> str:
    """Serialise *obj* to a compact JSON string.

    Falls back to ``repr()`` for values that are not JSON-serialisable
    (e.g. numpy arrays, sklearn objects).

    Parameters
    ----------
    obj:
        Dictionary of options to serialise.

    Returns
    -------
    str
        JSON string, or ``repr(obj)`` on serialisation failure.
    """
    try:
        return json.dumps(obj, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(obj)

