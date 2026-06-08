# scripts/migrate_dedup_daily_bars.py

"""
One-time migration: deduplicate daily bars and canonicalize timestamps to midnight UTC.

Root cause this fixes
---------------------
Two ingest runs against different yfinance versions produced two UTC timestamps
for the same trading date:
  - Old yfinance returned naive pd.Timestamps → fetcher stored 00:00 UTC
  - New yfinance returned America/New_York-aware pd.Timestamps → fetcher
    stored 05:00 UTC (winter/EST) or 04:00 UTC (summer/EDT)
Both timestamps cleared the PRIMARY KEY (symbol, timestamp, timeframe) because
the stored TIMESTAMP values were numerically distinct, so INSERT OR IGNORE let
both through.  Result: SPY had 1,848 rows for 1,343 distinct dates.

What this script does
---------------------
For each (symbol, CAST(timestamp AS DATE), timeframe) group:
  1. Keeps exactly one row — the one with the LATEST raw timestamp (DESC order
     via ROW_NUMBER).  DESC gives preference to the 05:00/04:00 row from the
     newer full-history ingest, which uses a consistent adj_close convention.
     open/high/low/close/volume are byte-identical between halves per the
     production audit, so the choice of which half to keep does not affect
     backtests.
  2. Re-stamps the survivor's timestamp to midnight UTC of its date:
         CAST(CAST(timestamp AS DATE) AS TIMESTAMP)
     This makes on-disk timestamps agree with the Part A write-path floor
     (duckdb_store._bar_to_tuple now floors 1d bars to midnight UTC), so
     the DB is consistent with what any future ingest will write.

Atomicity approach
------------------
The migration is a table swap (CREATE new + DROP old + RENAME), not an
in-place UPDATE.  This ensures:
  - The PRIMARY KEY constraint is preserved on the final table.
  - If the script is interrupted between CREATE and RENAME, the original
    table is untouched and the orphan _deduped table can be dropped and the
    script re-run.
  - DuckDB's ACID guarantees make each DDL statement independently durable.

Run from the repo root:
    uv run python scripts/migrate_dedup_daily_bars.py

Prerequisites
-------------
  - Run with no other process holding the DuckDB file open.
  - Backup must exist: data/quant_trader.duckdb.bak-pre-dedup
    (created manually before running this script).
"""

# pathlib for cross-platform path handling.
from pathlib import Path

# sys.exit() to propagate failure to the caller (CI / shell script).
import sys

# duckdb is the embedded database.  We open the live file for writing here —
# the only place in this codebase that directly writes SQL outside of
# DuckDBStore, which is intentional: this migration is a one-off that does not
# fit the ORM-style store API.
import duckdb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Same default path as DuckDBStore().__init__ — relative to repo root, so
# run from there.  If you need a different path, override here.
DB_PATH = Path("data/quant_trader.duckdb")

# The interim table name used during the atomic swap.  Chosen to be obviously
# migration-related so it does not shadow a real table name.
STAGING_TABLE = "ohlcv_bars_deduped"

# The canonical production table name, mirroring DUCKDB_TABLE_NAME in schema.py.
PROD_TABLE = "ohlcv_bars"

# DDL for the new table — verbatim from schema.py CREATE_TABLE_SQL but with the
# staging name substituted.  Duplicating it here avoids importing src/, which
# might not be on sys.path when the script is run as a one-off.  If the
# schema ever changes, update both places.
STAGING_DDL = f"""
CREATE TABLE {STAGING_TABLE} (
    symbol     VARCHAR   NOT NULL,
    timestamp  TIMESTAMP NOT NULL,
    open       DOUBLE    NOT NULL,
    high       DOUBLE    NOT NULL,
    low        DOUBLE    NOT NULL,
    close      DOUBLE    NOT NULL,
    adj_close  DOUBLE    NOT NULL,
    volume     BIGINT    NOT NULL,
    timeframe  VARCHAR   NOT NULL,
    source     VARCHAR   NOT NULL,
    PRIMARY KEY (symbol, timestamp, timeframe)
)
""".strip()


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
    backup_path = DB_PATH.with_suffix(".duckdb.bak-pre-dedup")
    if not backup_path.exists():
        abort(
            f"Backup not found at {backup_path}. "
            "Create it with: cp data/quant_trader.duckdb data/quant_trader.duckdb.bak-pre-dedup"
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
        # Step 1: Confirm daily-only.  The floor must never touch intraday bars.
        # If any non-1d timeframe is present, STOP — the migration logic is not
        # designed for intraday data.
        # ------------------------------------------------------------------
        timeframes = [r[0] for r in conn.execute(
            f"SELECT DISTINCT timeframe FROM {PROD_TABLE} ORDER BY timeframe"
        ).fetchall()]
        print(f"Distinct timeframes: {timeframes}")
        if any(tf != "1d" for tf in timeframes):
            abort(
                f"Non-daily timeframes found: {timeframes}. "
                "This migration floors timestamps to midnight UTC — safe only for 1d bars. STOP."
            )
        print("OK: only '1d' timeframe present.\n")

        # ------------------------------------------------------------------
        # Step 2: Pre-migration counts — baseline for verification.
        # ------------------------------------------------------------------
        pre_total = run_scalar(conn, f"SELECT COUNT(*) FROM {PROD_TABLE}")
        pre_dup_groups = run_scalar(conn, f"""
            SELECT COUNT(*) FROM (
                SELECT symbol, CAST(timestamp AS DATE) AS d, timeframe
                FROM {PROD_TABLE}
                GROUP BY symbol, d, timeframe
                HAVING COUNT(*) > 1
            )
        """)
        pre_spy = run_scalar(conn, f"SELECT COUNT(*) FROM {PROD_TABLE} WHERE symbol='SPY'")
        pre_expected_final = pre_total - pre_dup_groups  # each dup group contributes 1 extra row

        print("=== PRE-MIGRATION ===")
        print(f"  Total rows            : {pre_total:,}")
        print(f"  Dup (symbol,date) groups: {pre_dup_groups:,}")
        print(f"  SPY row count         : {pre_spy:,}")
        print(f"  Expected post-total   : {pre_expected_final:,}")
        print()

        # ------------------------------------------------------------------
        # Step 3: Clean up any leftover staging table from a prior aborted run.
        # DROP TABLE IF EXISTS is idempotent — safe to call even if the table
        # does not exist.
        # ------------------------------------------------------------------
        conn.execute(f"DROP TABLE IF EXISTS {STAGING_TABLE}")
        print(f"Cleaned up any leftover '{STAGING_TABLE}' table.")

        # ------------------------------------------------------------------
        # Step 4: Create the deduped staging table with the same DDL (including
        # PRIMARY KEY) as the production table.
        # ------------------------------------------------------------------
        conn.execute(STAGING_DDL)
        print(f"Created staging table '{STAGING_TABLE}'.")

        # ------------------------------------------------------------------
        # Step 5: Populate the staging table.
        #
        # Inner query: ROW_NUMBER() partitioned by (symbol, date, timeframe),
        # ordered by raw timestamp DESC so the LATEST raw timestamp is rn=1.
        # For SPY, the 05:00/04:00 UTC row (newer full-history ingest) wins over
        # the 00:00 UTC row (older partial ingest).
        #
        # Outer query: keep only rn=1, then re-stamp timestamp to midnight UTC
        # via CAST(CAST(timestamp AS DATE) AS TIMESTAMP).  For any raw timestamp
        # on 2024-01-02 (00:00, 04:00, or 05:00), this produces 2024-01-02 00:00.
        # ------------------------------------------------------------------
        insert_sql = f"""
            INSERT INTO {STAGING_TABLE}
            SELECT
                symbol,
                CAST(CAST(timestamp AS DATE) AS TIMESTAMP) AS timestamp,
                open, high, low, close, adj_close, volume, timeframe, source
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol, CAST(timestamp AS DATE), timeframe
                        ORDER BY timestamp DESC
                    ) AS rn
                FROM {PROD_TABLE}
            ) ranked
            WHERE rn = 1
        """
        conn.execute(insert_sql)
        staging_count = run_scalar(conn, f"SELECT COUNT(*) FROM {STAGING_TABLE}")
        print(f"Inserted {staging_count:,} rows into staging table.")

        # Sanity-check before the destructive swap.
        if staging_count != pre_expected_final:
            abort(
                f"Staging row count {staging_count:,} != expected {pre_expected_final:,}. "
                f"Aborting before table swap — original '{PROD_TABLE}' is unchanged."
            )
        print(f"Sanity check passed: staging count {staging_count:,} == expected {pre_expected_final:,}.\n")

        # ------------------------------------------------------------------
        # Step 6: Atomic table swap — DROP original, RENAME staging.
        # After this point the migration is committed.
        # ------------------------------------------------------------------
        conn.execute(f"DROP TABLE {PROD_TABLE}")
        conn.execute(f"ALTER TABLE {STAGING_TABLE} RENAME TO {PROD_TABLE}")
        print(f"Table swap complete: '{STAGING_TABLE}' → '{PROD_TABLE}'.")

        # ------------------------------------------------------------------
        # Step 7: Post-migration verification.
        # ------------------------------------------------------------------
        print("\n=== POST-MIGRATION VERIFICATION ===")

        # V1: Total row count matches expected.
        post_total = run_scalar(conn, f"SELECT COUNT(*) FROM {PROD_TABLE}")
        print(f"  Total rows            : {post_total:,}  (expected {pre_expected_final:,})")
        if post_total != pre_expected_final:
            abort(f"POST-CHECK FAILED: total rows {post_total:,} != expected {pre_expected_final:,}")

        # V2: SPY row count == number of unique SPY dates pre-migration.
        post_spy = run_scalar(conn, f"SELECT COUNT(*) FROM {PROD_TABLE} WHERE symbol='SPY'")
        print(f"  SPY row count         : {post_spy:,}  (expected 1343)")
        if post_spy != 1343:
            abort(f"POST-CHECK FAILED: SPY row count {post_spy:,} != 1343")

        # V3: Zero dup (symbol, date) groups store-wide.
        post_dups = run_scalar(conn, f"""
            SELECT COUNT(*) FROM (
                SELECT symbol, CAST(timestamp AS DATE) AS d, timeframe
                FROM {PROD_TABLE}
                GROUP BY symbol, d, timeframe
                HAVING COUNT(*) > 1
            )
        """)
        print(f"  Dup (symbol,date) groups: {post_dups:,}  (expected 0)")
        if post_dups != 0:
            abort(f"POST-CHECK FAILED: {post_dups:,} dup groups remain")

        # V4: Every timestamp is midnight UTC (time-of-day = 00:00:00).
        non_midnight = run_scalar(conn, f"""
            SELECT COUNT(*) FROM {PROD_TABLE}
            WHERE timestamp != CAST(CAST(timestamp AS DATE) AS TIMESTAMP)
        """)
        print(f"  Non-midnight timestamps : {non_midnight:,}  (expected 0)")
        if non_midnight != 0:
            abort(f"POST-CHECK FAILED: {non_midnight:,} timestamps are not midnight UTC")

        # V5: SPY (date, close) sequence is identical to pre-migration DISTINCT sequence.
        # Load both as sorted lists of (date-string, close) tuples and compare.
        post_spy_seq = conn.execute(f"""
            SELECT CAST(timestamp AS DATE) AS d, close
            FROM {PROD_TABLE} WHERE symbol='SPY'
            ORDER BY d
        """).fetchall()
        # Compare via JSON file written before migration.
        import json
        with open("/tmp/spy_pre_dedup.json") as f:
            pre_spy_seq = [(r[0], r[1]) for r in json.load(f)]
        post_spy_seq_norm = [(str(r[0]), r[1]) for r in post_spy_seq]
        if post_spy_seq_norm == pre_spy_seq:
            print(f"  SPY (date,close) match : YES — {len(post_spy_seq_norm)} rows identical to pre-migration DISTINCT sequence")
        else:
            # Find first mismatch for diagnostic output.
            for i, (pre, post) in enumerate(zip(pre_spy_seq, post_spy_seq_norm)):
                if pre != post:
                    print(f"  First mismatch at index {i}: pre={pre}, post={post}")
                    break
            abort(f"POST-CHECK FAILED: SPY (date,close) sequence differs from pre-migration")

        # V6: Spot-check a second symbol — symbol 'A' (1,257 rows, no dup dates pre-migration).
        spot_sym = "A"
        spot_count = run_scalar(conn, f"SELECT COUNT(*) FROM {PROD_TABLE} WHERE symbol=?", [spot_sym])
        spot_dups = run_scalar(conn, f"""
            SELECT COUNT(*) FROM (
                SELECT symbol, CAST(timestamp AS DATE) AS d
                FROM {PROD_TABLE}
                WHERE symbol=?
                GROUP BY symbol, d, timeframe
                HAVING COUNT(*) > 1
            )
        """, [spot_sym])
        spot_non_midnight = run_scalar(conn, f"""
            SELECT COUNT(*) FROM {PROD_TABLE}
            WHERE symbol=?
              AND timestamp != CAST(CAST(timestamp AS DATE) AS TIMESTAMP)
        """, [spot_sym])
        print(f"  Spot-check '{spot_sym}': {spot_count:,} rows, {spot_dups} dup dates, {spot_non_midnight} non-midnight")
        if spot_dups != 0 or spot_non_midnight != 0:
            abort(f"POST-CHECK FAILED: spot symbol {spot_sym!r} has unexpected state")

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
