"""DuckDB persistence helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def store_frame(frame: pd.DataFrame, table: str, database: str | Path = "data/quant.duckdb") -> None:
    """Replace a DuckDB table with a DataFrame."""
    if not table.replace("_", "").isalnum():
        raise ValueError("Invalid table name")
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as conn:
        conn.register("incoming_frame", frame)
        conn.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM incoming_frame')
        conn.unregister("incoming_frame")
