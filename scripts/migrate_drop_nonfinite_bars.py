# scripts/migrate_drop_nonfinite_bars.py

"""
One-time migration: delete non-finite OHLCV bars (NaN / +/-inf close or adj_close).

Root cause this fixes
---------------------
Every one of the 17 etf_basket symbols carried exactly one corrupt trailing bar
— the 2026-06-10 row — whose close and adj_close were both NaN.  Such a row
clears the NOT NULL / PRIMARY KEY constraints (NaN is a valid DOUBLE), so it was
written straight through by DuckDBStore.write_bars, which has no finiteness
guard.  Downstream, np.log(closes[1:] / closes[:-1]) in the backtester turns a
single NaN close into a poisoned returns series.

The Part A audit confirmed:
  - 17 non-finite bars total, exactly 1 per etf_basket symbol.
  - All are the FINAL (trailing) bar; no interior bars are affected.
  - All are NaN; NO infinities are present today.

What this script does
---------------------
Deletes every row from ohlcv_bars whose close OR adj_close is non-finite, and
nothing else.  The predicate also includes isinf() arms on both columns so that
any future infinite garbage is swept too — durability beyond today's NaN-only
defect.  isnan() and isinf() were verified to exist in the installed DuckDB
build (1.5.2) by a probe before this predicate was written; see the WHERE_NONFINITE
comment below.

Why a plain DELETE (not a table swap)
-------------------------------------
Unlike the dedup migration, this removes whole rows without touching the kept
rows' values or the schema, so an in-place DELETE is sufficient.  DuckDB's ACID
guarantees make the single DELETE statement durable; if interrupted before it
commits, no rows are removed and the script can simply be re-run.

Run from the repo root:
    uv run python scripts/migrate_drop_nonfinite_bars.py

Prerequisites
-------------
  - Run with no other process holding the DuckDB file open.
  - Backup must exist: data/quant_trader.duckdb.bak-pre-nonfinite
    (the operator has already backed up the database — see the guard below).
"""

# pathlib for cross-platform path handling — mirrors the dedup migration.
from pathlib import Path

# sys.exit() to propagate failure to the caller (CI / shell script).
import sys

# duckdb is the embedded database.  We open the live file for writing here —
# the only place outside DuckDBStore that writes SQL directly, intentional for
# a one-off migration that does not fit the store API.
import duckdb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Same default path as DuckDBStore().__init__ — relative to repo root, so run
# from there.  If you need a different path, override here.
DB_PATH = Path("data/quant_trader.duckdb")

# The canonical production table name, mirroring DUCKDB_TABLE_NAME in schema.py.
PROD_TABLE = "ohlcv_bars"

# The number of garbage rows the operator expects to remove (Part A audit:
# 17 etf_basket symbols x 1 trailing NaN bar each).  Used as a sanity gate.
EXPECTED_DELETIONS = 17

# The non-finite predicate, reused by the pre-count, the DELETE, and the
# post-count so all three are guaranteed to test exactly the same rows.
#
# Function choice is VERIFIED, not assumed: a probe against the installed
# DuckDB (1.5.2) confirmed both isnan() and isinf() exist and behave as
# expected (isnan(nan)->True, isinf(inf)->True).  Therefore:
#   - NaN test uses isnan() on both columns (the only defect present today).
#   - isinf() arms are added on both columns for durability against any future
#     infinite values, even though the audit found none today.
# The predicate covers BOTH close and adj_close.
WHERE_NONFINITE = (
    "isnan(close) OR isnan(adj_close) "
    "OR isinf(close) OR isinf(adj_close)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_scalar(conn: duckdb.DuckDBPyConnection, sql: str, params=None) -> any:
    """Execute a scalar query and return the single value."""
    if params:
        return conn.execute(sql, params).fetchone()[0]
    return conn.execute(sql).fetchone()[0]


def abort(msg: str) -> None:
    """Print a prefixed error message and exit with code 1."""
    print(f"\nABORT: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------

def main() -> int:
    # ------------------------------------------------------------------
    # Guard: backup must exist — belt-and-suspenders before touching the DB.
    # ------------------------------------------------------------------
    backup_path = DB_PATH.with_suffix(".duckdb.bak-pre-nonfinite")
    if not backup_path.exists():
        abort(
            f"Backup not found at {backup_path}. "
            "Create it with: cp data/quant_trader.duckdb data/quant_trader.duckdb.bak-pre-nonfinite"
        )

    if not DB_PATH.exists():
        abort(f"Database file not found at {DB_PATH}.")

    print(f"Database : {DB_PATH} ({DB_PATH.stat().st_size:,} bytes)")
    print(f"Backup   : {backup_path} ({backup_path.stat().st_size:,} bytes)")
    print()

    # ------------------------------------------------------------------
    # Open connection (read-write — we will modify the DB).
    # ------------------------------------------------------------------
    conn = duckdb.connect(str(DB_PATH))

    try:
        # ------------------------------------------------------------------
        # Step 1: Confirm the verified non-finite functions are usable on THIS
        # connection/file too — fail loudly rather than silently matching no
        # rows if a function were somehow unavailable here.
        # ------------------------------------------------------------------
        probe = run_scalar(
            conn,
            "SELECT isnan(CAST('nan' AS DOUBLE)) AND isinf(CAST('inf' AS DOUBLE))",
        )
        if probe is not True:
            abort("isnan()/isinf() did not behave as expected on this connection.")
        print("OK: isnan()/isinf() verified available on this connection.\n")

        # ------------------------------------------------------------------
        # Step 2: Pre-migration counts — baseline for verification.
        # ------------------------------------------------------------------
        pre_total = run_scalar(conn, f"SELECT COUNT(*) FROM {PROD_TABLE}")

        # Total garbage rows the predicate matches before deleting anything.
        pre_nonfinite = run_scalar(
            conn, f"SELECT COUNT(*) FROM {PROD_TABLE} WHERE {WHERE_NONFINITE}"
        )

        # Per-symbol breakdown of exactly which rows will be removed, so the
        # operator can eyeball that the deletion is the expected 17 trailing bars.
        per_symbol = conn.execute(
            f"""
            SELECT symbol, COUNT(*) AS n
            FROM {PROD_TABLE}
            WHERE {WHERE_NONFINITE}
            GROUP BY symbol
            ORDER BY symbol
            """
        ).fetchall()

        print("=== PRE-MIGRATION ===")
        print(f"  Total rows                : {pre_total:,}")
        print(f"  Non-finite rows to remove : {pre_nonfinite:,}")
        print(f"  Per-symbol breakdown      :")
        # One line per affected symbol — symbol and count of rows to be removed.
        for symbol, n in per_symbol:
            print(f"      {symbol:<6} {n}")
        print(f"  Grand total to remove     : {pre_nonfinite:,}  (expected {EXPECTED_DELETIONS})")
        print()

        # Sanity gate: the matched total must equal the operator's expectation.
        # Diverging means the data is not what the audit described — STOP before
        # deleting, so we never silently remove more (or fewer) rows than intended.
        if pre_nonfinite != EXPECTED_DELETIONS:
            abort(
                f"Non-finite row count {pre_nonfinite:,} != expected {EXPECTED_DELETIONS}. "
                f"Refusing to DELETE — '{PROD_TABLE}' is unchanged. Re-run the audit."
            )
        print(f"Sanity check passed: {pre_nonfinite:,} == expected {EXPECTED_DELETIONS}.\n")

        # ------------------------------------------------------------------
        # Step 3: DELETE every row matching the verified non-finite predicate.
        # After this statement commits the migration is done.
        # ------------------------------------------------------------------
        conn.execute(f"DELETE FROM {PROD_TABLE} WHERE {WHERE_NONFINITE}")
        print(f"Deleted non-finite rows from '{PROD_TABLE}'.")

        # ------------------------------------------------------------------
        # Step 4: Post-migration verification.
        # ------------------------------------------------------------------
        print("\n=== POST-MIGRATION VERIFICATION ===")

        # V1: Zero non-finite rows remain (same predicate as before).
        post_nonfinite = run_scalar(
            conn, f"SELECT COUNT(*) FROM {PROD_TABLE} WHERE {WHERE_NONFINITE}"
        )
        print(f"  Non-finite rows remaining : {post_nonfinite:,}  (expected 0)")
        if post_nonfinite != 0:
            abort(f"POST-CHECK FAILED: {post_nonfinite:,} non-finite rows remain")

        # V2: Exactly pre_nonfinite rows were removed — total dropped by that much.
        post_total = run_scalar(conn, f"SELECT COUNT(*) FROM {PROD_TABLE}")
        rows_deleted = pre_total - post_total
        print(f"  Total rows                : {post_total:,}  (was {pre_total:,})")
        print(f"  Rows deleted              : {rows_deleted:,}  (expected {pre_nonfinite:,})")
        if rows_deleted != pre_nonfinite:
            abort(
                f"POST-CHECK FAILED: deleted {rows_deleted:,} rows != expected {pre_nonfinite:,}"
            )

        print("\nAll verifications passed.")

    finally:
        # Always close the connection so the file lock is released, even if
        # an abort() raised SystemExit partway through.
        conn.close()

    print(f"\nMigration complete.")
    print(f"Backup retained at: {backup_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
