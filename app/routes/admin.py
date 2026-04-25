"""Admin inspection and maintenance routes."""
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.legacy_migration import LegacyMigrationService
from app.services.vector_store import get_vector_store
from app.services.locks import get_lock_manager

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats")
async def admin_stats():
    return get_vector_store().get_stats()


@router.get("/shards")
async def admin_shards():
    store = get_vector_store()
    store.sharding_service.validate_shards()
    return {"shards": store.sharding_service.shard_details()}


@router.post("/rebuild-index")
async def rebuild_index():
    return get_vector_store().rebuild_index()


@router.post("/migrate-legacy")
async def migrate_legacy():
    """
    Migrates old Phase 1 vectors.npy + metadata.jsonl data into Phase 2 shards.
    Safe behavior:
    - does not delete legacy vectors.npy
    - skips chunks already migrated
    - adds shard_id and shard_offset into metadata memory
    """
    try:
        store = get_vector_store()

        result = LegacyMigrationService(
            store.persistence_service,
            store.sharding_service,
        ).migrate()

        store.rebuild_index(save=True)

        return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats/shards")
async def get_shard_stats():
    return get_vector_store().sharding_service.get_stats()


@router.get("/stats/wal")
async def get_wal_stats():
    return get_vector_store().wal_service.get_stats()


@router.get("/stats/deletions")
async def get_deletion_stats():
    return {
        "deleted_count": len(get_vector_store().deletion_service.get_deleted_chunks())
    }


@router.get("/stats/system")
async def get_system_stats():
    return get_vector_store().metrics_service.get_all_stats()


@router.post("/index/rebuild")
async def rebuild_index_legacy():
    return get_vector_store().rebuild_index()


@router.post("/shards/rebuild")
async def rebuild_shards():
    store = get_vector_store()
    store.sharding_service.rebuild_from_metadata(store.persistence_service.metadata)
    return {"status": "ok", "shards": store.sharding_service.get_stats()}


@router.post("/wal/verify")
async def verify_wal():
    return get_vector_store().wal_service.verify_wal()


@router.post("/wal/clear")
async def clear_wal():
    get_vector_store().wal_service.clear()
    return {"status": "ok", "message": "WAL cleared"}


@router.post("/deletions/purge")
async def purge_old_deletions(days: int = 30):
    purged = get_vector_store().deletion_service.purge_old_tombstones(days)
    return {"status": "ok", "purged_count": purged}


@router.post("/data/clear")
async def clear_all_data(confirm: bool = False):
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to clear all data",
        )

    store = get_vector_store()
    store.persistence_service.clear()
    store.wal_service.clear()
    store.deletion_service.clear_all()

    for shard in store.sharding_service.get_all_shards().values():
        path = Path(shard.path)
        if path.exists():
            path.unlink()

    store.sharding_service.rebuild_from_metadata({})
    store.index_service.clear()

    return {"status": "ok", "message": "All data cleared"}


@router.get("/health/detailed")
async def detailed_health():
    store = get_vector_store()
    return {
        "status": "ok",
        "startup": store.startup_summary,
        "stats": store.get_stats(),
        "index": store.index_service.get_stats(),
        "wal": store.wal_service.get_stats(),
    }


@router.get("/locks")
async def get_locks():
    manager = get_lock_manager()
    return {"locked_resources": manager.get_locked_resources()}