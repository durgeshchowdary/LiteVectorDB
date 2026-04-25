"""Legacy Phase 1 -> Phase 2 shard migration."""
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.services.sharding import ShardInfo


class LegacyMigrationService:
    def __init__(self, persistence_service, sharding_service):
        self.persistence = persistence_service
        self.sharding = sharding_service

    def migrate(self) -> Dict[str, Any]:
        metadata = self.persistence.metadata
        legacy_vectors = self.persistence.vectors

        if not metadata:
            return {"status": "no_data", "migrated": 0}

        if legacy_vectors is None:
            return {"status": "no_vectors", "migrated": 0}

        shard_groups: Dict[int, List[tuple[str, dict, np.ndarray]]] = {}

        for chunk_id, entry in metadata.items():
            vector_idx = entry.get("vector_idx")

            if vector_idx is None:
                continue

            vector_idx = int(vector_idx)

            if vector_idx < 0 or vector_idx >= len(legacy_vectors):
                continue

            doc_id = entry.get("doc_id", chunk_id)
            shard_id = self.sharding.shard_for_doc(doc_id)
            vector = np.asarray(legacy_vectors[vector_idx], dtype=np.float32)

            shard_groups.setdefault(shard_id, []).append((chunk_id, entry, vector))

        migrated = 0

        for shard_id, items in shard_groups.items():
            shard_path = self.sharding.shards_dir / f"shard_{shard_id}.npy"
            tmp_path = self.sharding.shards_dir / f"shard_{shard_id}.npy.tmp"

            vectors = np.vstack([item[2] for item in items]).astype(np.float16)

            with open(tmp_path, "wb") as f:
                np.save(f, vectors)
                f.flush()
                os.fsync(f.fileno())

            if shard_path.exists():
                shard_path.unlink()

            os.replace(tmp_path, shard_path)

            chunk_ids = []

            for offset, (chunk_id, entry, _) in enumerate(items):
                entry["shard_id"] = shard_id
                entry["shard_offset"] = offset
                chunk_ids.append(chunk_id)
                migrated += 1

            checksum = self.sharding._checksum_file(shard_path)

            self.sharding._shards[shard_id] = ShardInfo(
                shard_id=shard_id,
                path=str(shard_path),
                vector_count=len(items),
                chunk_ids=chunk_ids,
                last_updated=str(time.time()),
                checksum=checksum,
                health="ok",
            )

        self._rewrite_metadata_jsonl()
        self.sharding._save_index()
        self.sharding._cache.clear()

        return {
            "status": "ok",
            "migrated": migrated,
            "shards_rebuilt": len(shard_groups),
            "total_metadata": len(metadata),
        }

    def _rewrite_metadata_jsonl(self) -> None:
        metadata_path = self.persistence.metadata_path
        tmp_path = Path(str(metadata_path) + ".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            for entry in self.persistence.metadata.values():
                f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, metadata_path)