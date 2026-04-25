"""File lock service for concurrency control in LiteVectorDB."""
import os
import time
import threading
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from app.config import get_config


class FileLock:
    """A simple file-based lock implementation."""
    
    def __init__(self, lock_path: Path, timeout: float = 30.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._local_lock = threading.Lock()
        self._acquired = False
    
    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the lock."""
        start_time = time.time()
        
        while True:
            with self._local_lock:
                if not self._acquired:
                    # Try to acquire the lock
                    if self._try_acquire():
                        self._acquired = True
                        return True
                
                if not blocking:
                    return False

            # Check timeout and sleep outside the local mutex so another
            # thread in this process can release the lock.
            if time.time() - start_time > self.timeout:
                return False
            time.sleep(0.1)
    
    def _try_acquire(self) -> bool:
        """Try to acquire the lock file."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try to create the lock file exclusively
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            # Check if the lock is stale (older than timeout)
            try:
                mtime = os.path.getmtime(self.lock_path)
                if time.time() - mtime > self.timeout:
                    # Remove stale lock and retry
                    try:
                        self.lock_path.unlink()
                        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                        return True
                    except FileExistsError:
                        pass
            except OSError:
                pass
            return False
    
    def release(self) -> bool:
        """Release the lock."""
        with self._local_lock:
            if self._acquired:
                try:
                    self.lock_path.unlink()
                    self._acquired = False
                    return True
                except FileNotFoundError:
                    self._acquired = False
                    return True
            return False
    
    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock: {self.lock_path}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


class LockManager:
    """Manages file locks for different resources."""
    
    def __init__(self):
        self.config = get_config()
        self._locks: dict[str, FileLock] = {}
        self._lock_creation_lock = threading.Lock()
    
    def _get_lock(self, resource: str) -> FileLock:
        """Get or create a lock for a resource."""
        with self._lock_creation_lock:
            if resource not in self._locks:
                lock_path = self.config.storage.data_dir / "locks" / f"{resource}.lock"
                self._locks[resource] = FileLock(
                    lock_path=lock_path,
                    timeout=self.config.concurrency.lock_timeout_seconds
                )
            return self._locks[resource]
    
    @contextmanager
    def lock(self, resource: str = "default", blocking: bool = True):
        """Context manager for acquiring a lock."""
        file_lock = self._get_lock(resource)
        acquired = file_lock.acquire(blocking=blocking)
        
        if not acquired:
            raise TimeoutError(f"Could not acquire lock for resource: {resource}")
        
        try:
            yield file_lock
        finally:
            file_lock.release()
    
    def is_locked(self, resource: str = "default") -> bool:
        """Check if a resource is currently locked."""
        lock = self._get_lock(resource)
        return lock._acquired
    
    def get_locked_resources(self) -> list[str]:
        """Get list of currently locked resources."""
        return [resource for resource, lock in self._locks.items() if lock._acquired]


# Global lock manager instance
_lock_manager: Optional[LockManager] = None


def get_lock_manager() -> LockManager:
    """Get the global lock manager instance."""
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = LockManager()
    return _lock_manager


@contextmanager
def file_lock(resource: str = "default", blocking: bool = True):
    """Convenience function for acquiring a file lock."""
    manager = get_lock_manager()
    with manager.lock(resource, blocking):
        yield
