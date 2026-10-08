"""Nonblocking conversation locks using each operating system's native API."""
from contextlib import contextmanager
import errno
import sys


def acquire_lock(stream, *, shared=False):
    """Lock an independent descriptor; contention raises BlockingIOError."""
    if sys.platform == 'win32':
        import msvcrt
        # CRT locks a byte range. Every participant uses byte zero; a shared
        # snapshot conservatively takes an exclusive lock on Windows.
        if stream.seek(0, 2) == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK} or getattr(exc, 'winerror', None) == 33:
                raise BlockingIOError('Conversation lock is busy') from exc
            raise
    else:
        import fcntl
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(stream, mode | fcntl.LOCK_NB)


def release_lock(stream):
    if sys.platform == 'win32':
        import msvcrt
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def conversation_lock(path, *, shared=False):
    # A descriptor per call excludes concurrent calls in the same process too.
    with open(path, 'a+b') as stream:
        acquire_lock(stream, shared=shared)
        try:
            yield
        finally:
            release_lock(stream)
