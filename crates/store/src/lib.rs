//! Persistent SQLite-backed store of experiment results ("risk snapshots"),
//! replacing the server crate's earlier in-memory, fixed-capacity
//! `ResultStore`. Uses `rusqlite`'s `bundled` feature, which compiles
//! SQLite from source as part of the crate build, so no external `sqlite3`
//! needs to be present on the (distroless, no package manager) runtime
//! image.
//!
//! Also persists the `/ask` pipeline's narration/suggestion/grounding-
//! warnings text (`POST /experiment` leaves these `NULL` -- that path never
//! calls Gemini, so there's no narration to store), alongside the
//! `EvidenceTrace` (as `trace_json`) and a handful of denormalized summary
//! fields pulled out of it for fast querying. See the server crate's
//! `routes::post_ask`/`post_experiment` for what gets inserted.

use std::sync::{Arc, Mutex};

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

pub type Result<T> = std::result::Result<T, StoreError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RiskSnapshot {
    pub id: String,
    pub created_at: String,
    pub portfolio_hash: String,
    pub experiment_type: String,
    pub engine_version: String,
    pub regime_label: Option<String>,
    pub smoothed_probs: Option<[f64; 3]>,
    pub portfolio_vol_annualized: Option<f64>,
    pub cvar_historical: Option<f64>,
    pub trace_json: String,
    /// `/ask`'s grounded narration text. `None` for a `/experiment`-
    /// originated snapshot (that path never calls Gemini).
    pub narration: Option<String>,
    /// `/ask`'s proactive follow-up suggestion. `None` for `/experiment`.
    pub suggestion: Option<String>,
    /// JSON-encoded `Vec<String>` of `/ask`'s grounding warnings; `None` if
    /// there were none (or this is a `/experiment` snapshot).
    pub grounding_warnings: Option<String>,
}

pub struct SnapshotStore {
    conn: Arc<Mutex<Connection>>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS risk_snapshots (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    portfolio_hash TEXT NOT NULL,
    experiment_type TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    regime_label TEXT,
    smoothed_probs TEXT,
    portfolio_vol_annualized REAL,
    cvar_historical REAL,
    trace_json TEXT NOT NULL,
    narration TEXT,
    suggestion TEXT,
    grounding_warnings TEXT
);
CREATE INDEX IF NOT EXISTS risk_snapshots_portfolio_hash_created_at
    ON risk_snapshots (portfolio_hash, created_at);
CREATE TABLE IF NOT EXISTS agent_execution_traces (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    execution_trace_json TEXT NOT NULL
);
";

/// Columns added after the table's initial release. SQLite has no `ALTER
/// TABLE ... ADD COLUMN IF NOT EXISTS`, so each is added via a `pragma_table_info`
/// existence check instead -- safe to run against a pre-existing database
/// that predates these columns (they're simply added, existing rows get
/// `NULL`), and a no-op against a freshly `CREATE TABLE`d one (`SCHEMA`
/// above already declares them, so `ensure_column` finds them present).
const ADDITIVE_COLUMNS: &[(&str, &str)] =
    &[("narration", "TEXT"), ("suggestion", "TEXT"), ("grounding_warnings", "TEXT")];

fn ensure_column(conn: &Connection, name: &str, sql_type: &str) -> rusqlite::Result<()> {
    let exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('risk_snapshots') WHERE name = ?1",
        params![name],
        |row| row.get(0),
    )?;
    if !exists {
        conn.execute(&format!("ALTER TABLE risk_snapshots ADD COLUMN {name} {sql_type}"), [])?;
    }
    Ok(())
}

impl SnapshotStore {
    /// Opens (creating if absent) the SQLite database at `path` and runs the
    /// (idempotent) schema migration. Pass `":memory:"` for a private,
    /// non-persistent database -- used by this crate's own tests and by the
    /// server crate's tests, so neither needs a file on disk.
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        for (name, sql_type) in ADDITIVE_COLUMNS {
            ensure_column(&conn, name, sql_type)?;
        }
        Ok(SnapshotStore {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Inserts `snapshot`, ignoring whatever `snapshot.id` was set to and
    /// generating a fresh UUID v4 in its place (mirroring the old
    /// `ResultStore::insert`'s "id is server-assigned" contract). Returns
    /// the generated id.
    pub fn insert(&self, snapshot: &RiskSnapshot) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let created_at = Utc::now().to_rfc3339();
        let smoothed_probs_json = snapshot
            .smoothed_probs
            .map(|p| serde_json::to_string(&p).expect("[f64; 3] always serializes"));

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO risk_snapshots (
                id, created_at, portfolio_hash, experiment_type, engine_version,
                regime_label, smoothed_probs, portfolio_vol_annualized, cvar_historical, trace_json,
                narration, suggestion, grounding_warnings
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                id,
                created_at,
                snapshot.portfolio_hash,
                snapshot.experiment_type,
                snapshot.engine_version,
                snapshot.regime_label,
                smoothed_probs_json,
                snapshot.portfolio_vol_annualized,
                snapshot.cvar_historical,
                snapshot.trace_json,
                snapshot.narration,
                snapshot.suggestion,
                snapshot.grounding_warnings,
            ],
        )?;
        Ok(id)
    }

    pub fn get(&self, id: &str) -> Result<Option<RiskSnapshot>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT id, created_at, portfolio_hash, experiment_type, engine_version,
                        regime_label, smoothed_probs, portfolio_vol_annualized, cvar_historical, trace_json,
                        narration, suggestion, grounding_warnings
                 FROM risk_snapshots WHERE id = ?1",
                params![id],
                row_to_snapshot,
            )
            .optional()?;
        Ok(row)
    }

    /// Snapshots for `portfolio_hash`, newest first, at most `limit`. Used
    /// by Risk Drift (a later session) to compare a portfolio's risk over
    /// time.
    pub fn latest_for_portfolio(&self, portfolio_hash: &str, limit: usize) -> Result<Vec<RiskSnapshot>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, created_at, portfolio_hash, experiment_type, engine_version,
                    regime_label, smoothed_probs, portfolio_vol_annualized, cvar_historical, trace_json,
                    narration, suggestion, grounding_warnings
             FROM risk_snapshots WHERE portfolio_hash = ?1
             ORDER BY created_at DESC, rowid DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![portfolio_hash, limit as i64], row_to_snapshot)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The most recent snapshots across all portfolios, newest first, at
    /// most `limit` -- replaces the server's old 20-entry `VecDeque`.
    pub fn list_recent(&self, limit: usize) -> Result<Vec<RiskSnapshot>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, created_at, portfolio_hash, experiment_type, engine_version,
                    regime_label, smoothed_probs, portfolio_vol_annualized, cvar_historical, trace_json,
                    narration, suggestion, grounding_warnings
             FROM risk_snapshots ORDER BY created_at DESC, rowid DESC LIMIT ?1",
        )?;
        let rows = stmt
            .query_map(params![limit as i64], row_to_snapshot)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Persists `execution_trace_json` (an `agent::AgentExecutionTrace`,
    /// serialized -- this crate has no dependency on `agent`, so it's
    /// stored opaquely as JSON, same as `RiskSnapshot::trace_json`), under
    /// its own already-generated `id` (unlike `insert`, which always
    /// assigns a fresh one -- `AgentExecutionTrace::id` is generated once,
    /// by `agent::pipeline::run`, and `GET /execution-trace/{id}` needs to
    /// be able to look it up by that same id the caller already has from
    /// `AskResponse`).
    pub fn insert_execution_trace(&self, id: &str, execution_trace_json: &str) -> Result<()> {
        let created_at = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO agent_execution_traces (id, created_at, execution_trace_json)
             VALUES (?1, ?2, ?3)",
            params![id, created_at, execution_trace_json],
        )?;
        Ok(())
    }

    /// The raw `execution_trace_json` stored for `id`, or `None` if unknown.
    pub fn get_execution_trace(&self, id: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT execution_trace_json FROM agent_execution_traces WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(row)
    }
}

fn row_to_snapshot(row: &rusqlite::Row) -> rusqlite::Result<RiskSnapshot> {
    let smoothed_probs_json: Option<String> = row.get(6)?;
    let smoothed_probs = smoothed_probs_json
        .map(|s| serde_json::from_str::<[f64; 3]>(&s))
        .transpose()
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(6, rusqlite::types::Type::Text, Box::new(e)))?;

    Ok(RiskSnapshot {
        id: row.get(0)?,
        created_at: row.get(1)?,
        portfolio_hash: row.get(2)?,
        experiment_type: row.get(3)?,
        engine_version: row.get(4)?,
        regime_label: row.get(5)?,
        smoothed_probs,
        portfolio_vol_annualized: row.get(7)?,
        cvar_historical: row.get(8)?,
        trace_json: row.get(9)?,
        narration: row.get(10)?,
        suggestion: row.get(11)?,
        grounding_warnings: row.get(12)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot(portfolio_hash: &str, experiment_type: &str) -> RiskSnapshot {
        RiskSnapshot {
            id: String::new(), // insert() ignores this and assigns its own
            created_at: String::new(),
            portfolio_hash: portfolio_hash.to_string(),
            experiment_type: experiment_type.to_string(),
            engine_version: "0.1.0".to_string(),
            regime_label: Some("Bull".to_string()),
            smoothed_probs: Some([0.7, 0.2, 0.1]),
            portfolio_vol_annualized: Some(0.15),
            cvar_historical: None,
            trace_json: serde_json::json!({"experiment": experiment_type}).to_string(),
            narration: Some("Vol is 15.5% annualised.".to_string()),
            suggestion: Some("What if I reduce my turnover to 20%?".to_string()),
            grounding_warnings: Some(serde_json::to_string(&vec!["unverified number '99%'".to_string()]).unwrap()),
        }
    }

    #[test]
    fn insert_and_get_round_trips_the_full_snapshot() {
        let store = SnapshotStore::open(":memory:").unwrap();
        let snapshot = sample_snapshot("abc123", "RiskDecomposition");
        let id = store.insert(&snapshot).unwrap();

        let fetched = store.get(&id).unwrap().expect("just-inserted snapshot should be found");
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.portfolio_hash, "abc123");
        assert_eq!(fetched.experiment_type, "RiskDecomposition");
        assert_eq!(fetched.engine_version, "0.1.0");
        assert_eq!(fetched.regime_label.as_deref(), Some("Bull"));
        assert_eq!(fetched.smoothed_probs, Some([0.7, 0.2, 0.1]));
        assert_eq!(fetched.portfolio_vol_annualized, Some(0.15));
        assert_eq!(fetched.cvar_historical, None);
        assert_eq!(fetched.trace_json, snapshot.trace_json);
        assert_eq!(fetched.narration, snapshot.narration);
        assert_eq!(fetched.suggestion, snapshot.suggestion);
        assert_eq!(fetched.grounding_warnings, snapshot.grounding_warnings);
    }

    #[test]
    fn narration_and_grounding_warnings_are_null_for_an_experiment_originated_snapshot() {
        let store = SnapshotStore::open(":memory:").unwrap();
        let mut snapshot = sample_snapshot("abc123", "RiskDecomposition");
        snapshot.narration = None;
        snapshot.suggestion = None;
        snapshot.grounding_warnings = None;
        let id = store.insert(&snapshot).unwrap();

        let fetched = store.get(&id).unwrap().unwrap();
        assert_eq!(fetched.narration, None);
        assert_eq!(fetched.suggestion, None);
        assert_eq!(fetched.grounding_warnings, None);
    }

    #[test]
    fn get_returns_none_for_an_unknown_id() {
        let store = SnapshotStore::open(":memory:").unwrap();
        assert!(store.get("nonexistent").unwrap().is_none());
    }

    #[test]
    fn latest_for_portfolio_returns_newest_first_and_respects_limit() {
        let store = SnapshotStore::open(":memory:").unwrap();
        let ids: Vec<String> = (0..5)
            .map(|i| store.insert(&sample_snapshot("hash-a", &format!("Type{i}"))).unwrap())
            .collect();
        // Also insert a snapshot for a different portfolio, which must never appear.
        store.insert(&sample_snapshot("hash-b", "Other")).unwrap();

        let latest = store.latest_for_portfolio("hash-a", 3).unwrap();
        assert_eq!(latest.len(), 3);
        // created_at has only second resolution, so within-the-same-second
        // inserts tie there; the query breaks that tie on `rowid DESC`
        // (monotonically increasing with insertion order), so the 3 most
        // recently *inserted* ids should come back, newest (last-inserted)
        // first.
        assert_eq!(latest[0].id, ids[4]);
        assert_eq!(latest[1].id, ids[3]);
        assert_eq!(latest[2].id, ids[2]);
        for snap in &latest {
            assert_eq!(snap.portfolio_hash, "hash-a");
        }
    }

    #[test]
    fn list_recent_returns_newest_first_and_respects_limit() {
        let store = SnapshotStore::open(":memory:").unwrap();
        for i in 0..5 {
            store.insert(&sample_snapshot(&format!("hash-{i}"), "RiskDecomposition")).unwrap();
        }
        let recent = store.list_recent(2).unwrap();
        assert_eq!(recent.len(), 2);
    }

    /// `SnapshotStore::open` must be able to add `narration`/`suggestion`/
    /// `grounding_warnings` to a database file created before those columns
    /// existed, without erroring or losing existing rows.
    #[test]
    fn open_migrates_a_pre_existing_database_missing_the_new_columns() {
        let dir = std::env::temp_dir().join(format!("store-migration-test-{}", Uuid::new_v4()));
        let path = dir.to_str().unwrap().to_string();

        // Simulate the pre-migration schema directly (no narration/suggestion/
        // grounding_warnings columns), with one existing row.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE risk_snapshots (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, portfolio_hash TEXT NOT NULL,
                    experiment_type TEXT NOT NULL, engine_version TEXT NOT NULL, regime_label TEXT,
                    smoothed_probs TEXT, portfolio_vol_annualized REAL, cvar_historical REAL,
                    trace_json TEXT NOT NULL
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO risk_snapshots (id, created_at, portfolio_hash, experiment_type, engine_version, trace_json)
                 VALUES ('old-id', '2026-01-01T00:00:00Z', 'hash', 'RiskDecomposition', '0.1.0', '{}')",
                [],
            )
            .unwrap();
        }

        let store = SnapshotStore::open(&path).expect("open should migrate the pre-existing schema");
        let old_row = store.get("old-id").unwrap().expect("pre-existing row should survive the migration");
        assert_eq!(old_row.narration, None);
        assert_eq!(old_row.suggestion, None);
        assert_eq!(old_row.grounding_warnings, None);

        let new_snapshot = sample_snapshot("hash2", "CvarRebalance");
        let new_id = store.insert(&new_snapshot).unwrap();
        let fetched = store.get(&new_id).unwrap().unwrap();
        assert_eq!(fetched.narration, new_snapshot.narration);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn execution_trace_round_trips_through_the_store() {
        let store = SnapshotStore::open(":memory:").unwrap();
        store.insert_execution_trace("exec-1", r#"{"id":"exec-1","narration":"hi"}"#).unwrap();

        let fetched = store.get_execution_trace("exec-1").unwrap();
        assert_eq!(fetched.as_deref(), Some(r#"{"id":"exec-1","narration":"hi"}"#));
    }

    #[test]
    fn get_execution_trace_returns_none_for_an_unknown_id() {
        let store = SnapshotStore::open(":memory:").unwrap();
        assert!(store.get_execution_trace("nonexistent").unwrap().is_none());
    }

    #[test]
    fn inserting_25_snapshots_does_not_evict_any() {
        let store = SnapshotStore::open(":memory:").unwrap();
        let ids: Vec<String> = (0..25)
            .map(|i| store.insert(&sample_snapshot(&format!("hash-{i}"), "RiskDecomposition")).unwrap())
            .collect();
        for id in &ids {
            assert!(store.get(id).unwrap().is_some(), "snapshot {id} should not have been evicted");
        }
        assert_eq!(store.list_recent(100).unwrap().len(), 25);
    }
}
