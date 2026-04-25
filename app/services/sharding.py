"""Vector shard management for LiteVectorDB."""
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from app.config import get_config
from app.services.wal import atomic_write_json


@dataclass
class ShardInfo:
    shard_id: int
    path: str
    vector_count: int = 0
    chunk_ids: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    checksum: str = ""
    health: str = "ok"


class ShardingService:
    """Stores vectors in data/shards/shard_N.npy and tracks chunk placement."""

    def __init__(self):
        self.config = get_config()
        self.shard_count = max(1, self.config.sharding.shard_count)
        self.shards_dir = self.config.storage.shards_path()
        self.index_path = self.config.storage.shard_index_path()
        self._shards: Dict[int, ShardInfo] = {}
        self._chunk_to_shard: Dict[str, int] = {}
        self._cache: Dict[int, np.ndarray] = {}
        self._ensure_shards()
        self._load_index()

    def _ensure_shards(self) -> None:
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        for shard_id in range(self.shard_count):
            path = self.shards_dir / f"shard_{shard_id}.npy"
            self._shards[shard_id] = ShardInfo(shard_id=shard_id, path=str(path))

    def _stable_hash(self, value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big")

    def shard_for_doc(self, doc_id: str) -> int:
        return self._stable_hash(doc_id) % self.shard_count

    def _checksum_file(self, path: Path) -> str:
        if not path.exists():
            return ""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def _load_index(self) -> None:
        if not self.index_path.exists():
            self._save_index()
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for raw in data.get("shards", []):
                info = ShardInfo(**raw)
                self._shards[int(info.shard_id)] = info
                for chunk_id in info.chunk_ids:
                    self._chunk_to_shard[chunk_id] = int(info.shard_id)
        except Exception:
            self._save_index()

    def _save_index(self) -> None:
        data = {
            "version": 2,
            "shard_count": self.shard_count,
            "updated_at": datetime.utcnow().isoformat(),
            "shards": [asdict(self._shards[i]) for i in sorted(self._shards)],
        }
        atomic_write_json(self.index_path, data)

    def validate_shards(self) -> Dict[str, object]:
        warnings = []
        for shard_id, info in self._shards.items():
            path = Path(info.path)
            if not path.exists():
                info.vector_count = 0
                info.checksum = ""
                info.health = "missing" if info.chunk_ids else "empty"
                continue
            try:
                arr = np.load(path, mmap_mode="r")
                checksum = self._checksum_file(path)
                info.health = "ok" if not info.checksum or info.checksum == checksum else "checksum_mismatch"
                info.vector_count = int(arr.shape[0]) if arr.ndim == 2 else 0
                info.checksum = checksum
                if info.health != "ok":
                    warnings.append({"shard_id": shard_id, "health": info.health})
            except Exception as exc:
                info.health = "corrupt"
                warnings.append({"shard_id": shard_id, "health": "corrupt", "error": str(exc)})
        self._save_index()
        return {"warnings": warnings, "healthy": not warnings}

    def load_shard(self, shard_id: int) -> Optional[np.ndarray]:
        if shard_id in self._cache:
            return self._cache[shard_id]
        info = self._shards.get(shard_id)
        if not info:
            return None
        path = Path(info.path)
        if not path.exists():
            return None
        try:
            arr = np.load(path, mmap_mode="r")
            self._cache[shard_id] = arr
            return arr
        except Exception:
            info.health = "corrupt"
            return None

    def _drop_cache(self, shard_id: int) -> None:
        arr = self._cache.pop(shard_id, None)
        mmap_obj = getattr(arr, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def append_vectors(self, doc_id: str, chunk_ids: List[str], vectors: np.ndarray) -> List[Dict[str, int]]:
        shard_id = self.shard_for_doc(doc_id)
        info = self._shards[shard_id]
        path = Path(info.path)
        existing = None
        if path.exists():
            existing = np.load(path)
        start = int(existing.shape[0]) if existing is not None and existing.ndim == 2 else 0
        new_vectors = vectors if existing is None else np.vstack([existing, vectors])

        tmp = path.with_suffix(".npy.tmp")
        with open(tmp, "wb") as f:
            np.save(f, new_vectors)
            f.flush()
            os.fsync(f.fileno())
        self._drop_cache(shard_id)
        os.replace(tmp, path)

        refs = []
        for offset, chunk_id in enumerate(chunk_ids, start=start):
            refs.append({"shard_id": shard_id, "shard_offset": offset})
            if chunk_id not in info.chunk_ids:
                info.chunk_ids.append(chunk_id)
            self._chunk_to_shard[chunk_id] = shard_id
        info.vector_count = int(new_vectors.shape[0])
        info.last_updated = datetime.utcnow().isoformat()
        info.checksum = self._checksum_file(path)
        info.health = "ok"
        self._save_index()
        return refs

    def get_vectors_by_refs(self, refs: Iterable[Dict[str, int]]) -> np.ndarray:
        grouped: Dict[int, List[tuple[int, int]]] = {}
        refs_list = list(refs)
        for pos, ref in enumerate(refs_list):
            grouped.setdefault(int(ref["shard_id"]), []).append((pos, int(ref["shard_offset"])))
        result: List[Optional[np.ndarray]] = [None] * len(refs_list)
        for shard_id, positions in grouped.items():
            arr = self.load_shard(shard_id)
            if arr is None:
                continue
            for original_pos, offset in positions:
                if offset < len(arr):
                    result[original_pos] = arr[offset].astype(np.float32)
        return np.array([v for v in result if v is not None], dtype=np.float32)

    def remove_chunk_from_shard(self, chunk_id: str) -> None:
        shard_id = self._chunk_to_shard.pop(chunk_id, None)
        if shard_id is None or shard_id not in self._shards:
            return
        info = self._shards[shard_id]
        info.chunk_ids = [cid for cid in info.chunk_ids if cid != chunk_id]
        info.last_updated = datetime.utcnow().isoformat()
        self._save_index()

    def rebuild_from_metadata(self, metadata: Dict[str, dict]) -> None:
        for info in self._shards.values():
            info.chunk_ids = []
        self._chunk_to_shard.clear()
        for chunk_id, entry in metadata.items():
            shard_id = int(entry.get("shard_id", self.shard_for_doc(entry.get("doc_id", chunk_id))))
            if shard_id in self._shards:
                self._shards[shard_id].chunk_ids.append(chunk_id)
                self._chunk_to_shard[chunk_id] = shard_id
        self.validate_shards()

    def get_shard_for_chunk(self, chunk_id: str) -> Optional[int]:
        return self._chunk_to_shard.get(chunk_id)

    def get_all_shards(self) -> Dict[int, ShardInfo]:
        return self._shards

    def get_stats(self) -> dict:
        total_vectors = sum(s.vector_count for s in self._shards.values())
        return {
            "shard_count": self.shard_count,
            "total_vectors": total_vectors,
            "avg_vectors_per_shard": total_vectors / self.shard_count if self.shard_count else 0,
            "shards": [asdict(self._shards[i]) for i in sorted(self._shards)],
        }

    def shard_details(self) -> List[dict]:
        details = []
        for shard_id in sorted(self._shards):
            info = self._shards[shard_id]
            path = Path(info.path)
            details.append({
                **asdict(info),
                "file_size": path.stat().st_size if path.exists() else 0,
            })
        return details
