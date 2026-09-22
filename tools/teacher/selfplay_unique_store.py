"""Store exact canonical self-play states and admit validated shards."""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

POLICY_DIM = 209
V3_RECORD_SIZE = 64
META_RECORD_SIZE = 20
QUALITY_MISSING_ROOT_VALUE = 1
SCHEMA_VERSION = 1
V3_DTYPE = np.dtype([
    ("own_pawn", "u1"), ("opp_pawn", "u1"), ("walls_h", "<u8"), ("walls_v", "<u8"),
    ("walls_left_own", "i1"), ("walls_left_opp", "i1"), ("nnue_eval", "<u2"),
    ("game_result", "i1"), ("policy_target", "<u2"), ("own_dist", "u1"),
    ("opp_dist", "u1"), ("mover", "u1"), ("own_cat_total", "<i2"),
    ("opp_cat_total", "<i2"), ("policy_top_idx", "<u2", (8,)),
    ("policy_top_prob", "<u2", (8,)),
])


@dataclass(frozen=True)
class RecordRef:
    shard: str
    offset: int
    source: str = "unknown"
    family: str | None = None


@dataclass(frozen=True)
class Decision:
    inserted: bool = False
    replaced: bool = False
    rejected: bool = False
    equal: bool = False


class UniqueStateStore:
    """Persist exact state keys and the best reference for each key."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self._in_transaction = False
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS states (
                key BLOB PRIMARY KEY, reliability REAL NOT NULL,
                shard TEXT NOT NULL, offset INTEGER NOT NULL,
                source TEXT NOT NULL, family TEXT, quality_flags INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS equal_refs (
                key BLOB NOT NULL, reliability REAL NOT NULL,
                shard TEXT NOT NULL, offset INTEGER NOT NULL,
                source TEXT NOT NULL, family TEXT,
                FOREIGN KEY(key) REFERENCES states(key)
            );
            CREATE TABLE IF NOT EXISTS state_metadata (
                key BLOB PRIMARY KEY, quality_flags INTEGER NOT NULL,
                FOREIGN KEY(key) REFERENCES states(key)
            );
            CREATE TABLE IF NOT EXISTS counters (
                scope TEXT NOT NULL, name TEXT NOT NULL, value INTEGER NOT NULL,
                PRIMARY KEY(scope, name)
            );
            """
        )
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self.db.commit()

    @property
    def unique_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM states").fetchone()[0])

    def _inc(self, scope: str, name: str, value: int = 1) -> None:
        self.db.execute(
            "INSERT INTO counters VALUES (?, ?, ?) ON CONFLICT(scope,name) DO UPDATE SET value=value+excluded.value",
            (scope, name, value),
        )

    def _reject(self, ref: RecordRef | None) -> Decision:
        scopes = ["global"]
        if ref is not None:
            scopes += ["source:" + ref.source, "family:" + (ref.family or "unassigned")]
        for scope in scopes:
            self._inc(scope, "rejected")
        if not self._in_transaction:
            self.db.commit()
        return Decision(rejected=True)

    def accept(self, key: bytes, reliability: float, ref: RecordRef,
               quality_flags: int = 0) -> Decision:
        if not isinstance(key, bytes) or not key:
            return self._reject(ref)
        if not np.isfinite(reliability):
            return self._reject(ref)
        self._inc("global", "raw")
        for scope in ("source:" + ref.source, "family:" + (ref.family or "unassigned")):
            self._inc(scope, "raw")
        row = self.db.execute(
            "SELECT reliability, source, family FROM states WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            self.db.execute("INSERT INTO states(key,reliability,shard,offset,source,family,quality_flags) VALUES (?, ?, ?, ?, ?, ?, 0)", (key, reliability, ref.shard, ref.offset, ref.source, ref.family))
            self.db.execute("INSERT OR REPLACE INTO state_metadata VALUES (?, ?)", (key, quality_flags))
            self._inc("global", "unique")
            for scope in ("source:" + ref.source, "family:" + (ref.family or "unassigned")):
                self._inc(scope, "unique")
            if not self._in_transaction:
                self.db.commit()
            return Decision(inserted=True)
        old = float(row[0])
        if reliability > old:
            # Keep the first accepted source and family as the composition
            # label. A stronger duplicate updates the selected record only.
            # Otherwise a duplicate from another pool could move a state
            # across the central or broad quota during reconciliation.
            self.db.execute(
                "UPDATE states SET reliability=?, shard=?, offset=? WHERE key=?",
                (reliability, ref.shard, ref.offset, key),
            )
            self.db.execute("INSERT OR REPLACE INTO state_metadata VALUES (?, ?)", (key, quality_flags))
            self._inc("global", "replaced")
            for scope in ("source:" + ref.source, "family:" + (ref.family or "unassigned")):
                self._inc(scope, "replaced")
            if not self._in_transaction:
                self.db.commit()
            return Decision(replaced=True)
        if reliability == old:
            self.db.execute("INSERT INTO equal_refs VALUES (?, ?, ?, ?, ?, ?)", (key, reliability, ref.shard, ref.offset, ref.source, ref.family))
            self._inc("global", "equal")
            for scope in ("source:" + ref.source, "family:" + (ref.family or "unassigned")):
                self._inc(scope, "equal")
            if not self._in_transaction:
                self.db.commit()
            return Decision(equal=True)
        self._inc("global", "rejected")
        for scope in ("source:" + ref.source, "family:" + (ref.family or "unassigned")):
            self._inc(scope, "rejected")
        if not self._in_transaction:
            self.db.commit()
        return Decision(rejected=True)

    @staticmethod
    def canonical_key(row: np.void) -> bytes:
        """Build the exact mover-relative identity and exclude labels."""
        if not 0 <= int(row["own_pawn"]) < 81 or not 0 <= int(row["opp_pawn"]) < 81:
            raise ValueError("pawn cell is outside the board")
        if int(row["mover"]) not in (0, 1):
            raise ValueError("invalid side to move")
        if not 0 <= int(row["walls_left_own"]) <= 10 or not 0 <= int(row["walls_left_opp"]) <= 10:
            raise ValueError("remaining walls are outside the legal range")
        return struct.pack("<BBQQbbB", int(row["own_pawn"]), int(row["opp_pawn"]),
                           int(row["walls_h"]), int(row["walls_v"]),
                           int(row["walls_left_own"]), int(row["walls_left_opp"]),
                           int(row["mover"]))

    def lookup(self, key: bytes) -> RecordRef | None:
        row = self.db.execute("SELECT shard,offset,source,family FROM states WHERE key=?", (key,)).fetchone()
        return None if row is None else RecordRef(row[0], row[1], row[2], row[3])

    def counts(self, scope: str = "global") -> dict[str, int]:
        return {k: int(v) for k, v in self.db.execute("SELECT name,value FROM counters WHERE scope=?", (scope,))}

    def scopes(self) -> dict[str, dict[str, int]]:
        return {s: self.counts(s) for s, in self.db.execute("SELECT DISTINCT scope FROM counters")}

    def admit_shard(self, v3_path: str | Path, metadata_path: str | Path | None = None,
                    keys: Iterable[bytes] | None = None, reliability: float = 1.0,
                    source: str = "unknown", family: str | None = None,
                    missing_metadata: bool = False) -> int:
        v3 = Path(v3_path)
        if v3.stat().st_size % V3_RECORD_SIZE:
            raise ValueError("V3 shard size is not 64-byte aligned")
        n = v3.stat().st_size // V3_RECORD_SIZE
        if metadata_path is None:
            meta = [None] * n
        else:
            m = Path(metadata_path)
            if m.stat().st_size % META_RECORD_SIZE:
                raise ValueError("metadata size is not 20-byte aligned")
            if m.stat().st_size // META_RECORD_SIZE != n:
                raise ValueError("V3 and metadata record counts differ")
            meta = np.memmap(m, dtype=np.dtype([("game", "<u8"), ("root", "<f4"), ("plies", "<u2"), ("length", "<u2"), ("source", "u1"), ("flags", "u1"), ("reserved", "<u2")]), mode="r")
        arr = np.memmap(v3, dtype=V3_DTYPE, mode="r")
        key_list = list(keys) if keys is not None else [self.canonical_key(row) for row in arr]
        if len(key_list) != n:
            raise ValueError("key count differs from V3 record count")
        if any(not isinstance(key, bytes) or not key for key in key_list):
            raise ValueError("canonical state key is empty or invalid")
        for i, row in enumerate(arr):
            action = int(row["policy_target"])
            if action >= POLICY_DIM:
                raise ValueError("illegal action index")
            for idx, prob in zip(row["policy_top_idx"], row["policy_top_prob"]):
                if int(prob) and int(idx) >= POLICY_DIM:
                    raise ValueError("illegal populated top policy index")
            if meta is not None and meta[i] is not None:
                rec = meta[i]
                if not (rec["flags"] & QUALITY_MISSING_ROOT_VALUE) and not np.isfinite(rec["root"]):
                    raise ValueError("root value is not finite")
                if rec["length"] and rec["plies"] > rec["length"]:
                    raise ValueError("terminal distance exceeds game length")
        self.db.execute("BEGIN")
        self._in_transaction = True
        try:
            for i, key in enumerate(key_list):
                flags = QUALITY_MISSING_ROOT_VALUE if missing_metadata else 0
                self.accept(key, reliability, RecordRef(str(v3), i, source, family), flags)
                if missing_metadata:
                    pass
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        finally:
            self._in_transaction = False
        return n

    def import_weakness_loss_100ms(self, root: str | Path | None = None,
                                   reliability: float = 1.0) -> int:
        root = Path(root or "data/selfplay_canonical_v3/weakness-loss-100ms-20260920")
        total = 0
        for path in sorted(root.rglob("*.bin")):
            if path.stat().st_size % V3_RECORD_SIZE:
                raise ValueError(f"unaligned V3 shard: {path}")
            arr = np.memmap(path, dtype=V3_DTYPE, mode="r")
            keys = [self.canonical_key(row) for row in arr]
            before = self.unique_count
            self.admit_shard(path, keys=keys, reliability=reliability,
                             source="weakness-loss-100ms-20260920", family=None,
                             missing_metadata=True)
            total += len(keys)
        return total

    def close(self) -> None:
        self.db.close()
