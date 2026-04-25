"""Write-ahead log with pending/committed operation records."""
import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.config import get_config


@dataclass
class WALEntry:
    entry_id: int
    tx_id: str
    operation: str
    doc_id: str
    timestamp: str
    status: str
    details: Dict[str, Any]
    checksum: str = ""

    def payload_for_checksum(self) -> Dict[str, Any]:
        data = asdict(self)
        data["checksum"] = ""
        return data

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WALService:
    """Append-only WAL used to detect interrupted writes on startup."""

    def __init__(self):
        self.config = get_config()
        self.path = self.config.storage.wal_path()
        self._lock = threading.RLock()
        self._entries: List[WALEntry] = []
        self._entry_counter = 0
        self._load_wal()

    def _checksum(self, data: Dict[str, Any]) -> str:
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _fsync_append(self, entry: WALEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _load_wal(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    # Backward compatibility with the older Phase 2 draft WAL.
                    if "document_id" in raw and "doc_id" not in raw:
                        raw = {
                            "entry_id": raw.get("entry_id", 0),
                            "tx_id": f"legacy-{raw.get('entry_id', 0)}",
                            "operation": raw.get("operation", "unknown"),
                            "doc_id": raw.get("document_id", ""),
                            "timestamp": raw.get("timestamp", datetime.utcnow().isoformat()),
                            "status": raw.get("status", "committed"),
                            "details": raw.get("data", {}),
                            "checksum": raw.get("checksum", ""),
                        }
                    entry = WALEntry(**raw)
                    self._entries.append(entry)
                    self._entry_counter = max(self._entry_counter, entry.entry_id)
                except Exception:
                    continue

    def begin(self, operation: str, doc_id: str, details: Dict[str, Any] | None = None) -> str:
        """Append a pending WAL entry and return its transaction id."""
        with self._lock:
            self._entry_counter += 1
            tx_id = f"{int(datetime.utcnow().timestamp() * 1000000)}-{self._entry_counter}"
            entry = WALEntry(
                entry_id=self._entry_counter,
                tx_id=tx_id,
                operation=operation,
                doc_id=doc_id,
                timestamp=datetime.utcnow().isoformat(),
                status="pending",
                details=details or {},
            )
            entry.checksum = self._checksum(entry.payload_for_checksum())
            self._entries.append(entry)
            self._fsync_append(entry)
            return tx_id

    def commit(self, tx_id: str, details: Dict[str, Any] | None = None) -> None:
        self._finish(tx_id, "committed", details)

    def fail(self, tx_id: str, details: Dict[str, Any] | None = None) -> None:
        self._finish(tx_id, "failed", details)

    def _finish(self, tx_id: str, status: str, details: Dict[str, Any] | None) -> None:
        with self._lock:
            prior = next((e for e in reversed(self._entries) if e.tx_id == tx_id), None)
            self._entry_counter += 1
            entry = WALEntry(
                entry_id=self._entry_counter,
                tx_id=tx_id,
                operation=prior.operation if prior else "unknown",
                doc_id=prior.doc_id if prior else "",
                timestamp=datetime.utcnow().isoformat(),
                status=status,
                details=details or {},
            )
            entry.checksum = self._checksum(entry.payload_for_checksum())
            self._entries.append(entry)
            self._fsync_append(entry)

    def recover(self) -> Dict[str, Any]:
        """Report transactions that were pending without a terminal record."""
        terminal = {}
        pending = {}
        for entry in self._entries:
            if entry.status == "pending":
                pending[entry.tx_id] = entry
            elif entry.status in {"committed", "failed"}:
                terminal[entry.tx_id] = entry.status
        unresolved = [e for tx, e in pending.items() if tx not in terminal]
        return {
            "pending": len(unresolved),
            "recovered": 0,
            "failed": len([e for e in self._entries if e.status == "failed"]),
            "pending_transactions": [
                {"tx_id": e.tx_id, "operation": e.operation, "doc_id": e.doc_id, "timestamp": e.timestamp}
                for e in unresolved
            ],
        }

    def verify_wal(self) -> Dict[str, Any]:
        valid = 0
        invalid = []
        for entry in self._entries:
            if entry.checksum == self._checksum(entry.payload_for_checksum()):
                valid += 1
            else:
                invalid.append(entry.entry_id)
        return {
            "total_entries": len(self._entries),
            "valid": valid,
            "invalid": len(invalid),
            "invalid_entry_ids": invalid,
        }

    def get_stats(self) -> Dict[str, Any]:
        recovery = self.recover()
        operations: Dict[str, int] = {}
        statuses: Dict[str, int] = {}
        for entry in self._entries:
            operations[entry.operation] = operations.get(entry.operation, 0) + 1
            statuses[entry.status] = statuses.get(entry.status, 0) + 1
        return {
            "total_entries": len(self._entries),
            "entry_counter": self._entry_counter,
            "operations": operations,
            "statuses": statuses,
            "pending": recovery["pending"],
        }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._entry_counter = 0
            if self.path.exists():
                self.path.unlink()


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write JSON to disk using temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
