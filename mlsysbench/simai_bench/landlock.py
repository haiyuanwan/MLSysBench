"""Small, dependency-free Linux Landlock wrapper for agent processes."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path
from typing import Iterable


class LandlockError(RuntimeError):
    """Raised when a requested Landlock sandbox cannot be installed."""


# Landlock uses the generic Linux syscall table on supported architectures.
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446

_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14

_READ_ONLY_DIR_ACCESS = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
_READ_ONLY_FILE_ACCESS = _FS_EXECUTE | _FS_READ_FILE


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def landlock_abi_version() -> int | None:
    """Return the supported Landlock ABI, or ``None`` when unavailable."""

    if sys.platform != "linux":
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    result = libc.syscall(
        ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    if result >= 0:
        return int(result)
    if ctypes.get_errno() in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
        return None
    return None


def restrict_current_process(
    *,
    read_only_paths: Iterable[str | Path],
    read_write_paths: Iterable[str | Path],
) -> int:
    """Apply a deny-by-default Landlock policy to this process and descendants.

    The caller should invoke this immediately before ``exec``. Open file
    descriptors are outside Landlock's scope, so the parent must avoid passing
    descriptors that expose private benchmark inputs.
    """

    abi = landlock_abi_version()
    if abi is None:
        raise LandlockError("Landlock is not supported or enabled on this host")

    handled_access = (
        _FS_EXECUTE
        | _FS_WRITE_FILE
        | _FS_READ_FILE
        | _FS_READ_DIR
        | _FS_REMOVE_DIR
        | _FS_REMOVE_FILE
        | _FS_MAKE_CHAR
        | _FS_MAKE_DIR
        | _FS_MAKE_REG
        | _FS_MAKE_SOCK
        | _FS_MAKE_FIFO
        | _FS_MAKE_BLOCK
        | _FS_MAKE_SYM
    )
    if abi >= 2:
        handled_access |= _FS_REFER
    if abi >= 3:
        handled_access |= _FS_TRUNCATE

    libc = ctypes.CDLL(None, use_errno=True)
    ruleset_attr = _RulesetAttr(handled_access_fs=handled_access)
    ctypes.set_errno(0)
    ruleset_fd = libc.syscall(
        ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
        ctypes.byref(ruleset_attr),
        ctypes.c_size_t(ctypes.sizeof(ruleset_attr)),
        ctypes.c_uint(0),
    )
    if ruleset_fd < 0:
        _raise_errno("landlock_create_ruleset")

    try:
        for path in _unique_existing_paths(read_only_paths):
            _add_path_rule(libc, ruleset_fd, path, handled_access, writable=False)
        for path in _unique_existing_paths(read_write_paths):
            _add_path_rule(libc, ruleset_fd, path, handled_access, writable=True)

        ctypes.set_errno(0)
        if libc.prctl(
            ctypes.c_int(_PR_SET_NO_NEW_PRIVS),
            ctypes.c_ulong(1),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        ) != 0:
            _raise_errno("prctl(PR_SET_NO_NEW_PRIVS)")

        ctypes.set_errno(0)
        result = libc.syscall(
            ctypes.c_long(_SYS_LANDLOCK_RESTRICT_SELF),
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint(0),
        )
        if result != 0:
            _raise_errno("landlock_restrict_self")
    finally:
        os.close(ruleset_fd)
    return abi


def default_system_read_paths() -> tuple[Path, ...]:
    """System trees commonly needed by dynamically linked CLI programs."""

    candidates = (
        "/bin",
        "/dev",
        "/etc",
        "/lib",
        "/lib64",
        "/opt",
        "/run",
        "/sbin",
        "/sys",
        "/usr",
        "/var",
    )
    return tuple(Path(path) for path in candidates if Path(path).exists())


def _add_path_rule(
    libc: ctypes.CDLL,
    ruleset_fd: int,
    path: Path,
    handled_access: int,
    *,
    writable: bool,
) -> None:
    is_dir = path.is_dir()
    if writable:
        allowed_access = handled_access if is_dir else (
            _FS_READ_FILE | _FS_WRITE_FILE | _FS_EXECUTE | _FS_TRUNCATE
        ) & handled_access
    else:
        allowed_access = _READ_ONLY_DIR_ACCESS if is_dir else _READ_ONLY_FILE_ACCESS

    path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        rule_attr = _PathBeneathAttr(
            allowed_access=allowed_access,
            parent_fd=path_fd,
        )
        ctypes.set_errno(0)
        result = libc.syscall(
            ctypes.c_long(_SYS_LANDLOCK_ADD_RULE),
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(rule_attr),
            ctypes.c_uint(0),
        )
        if result != 0:
            _raise_errno(f"landlock_add_rule({path})")
    finally:
        os.close(path_fd)


def _unique_existing_paths(paths: Iterable[str | Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            continue
        unique[str(path)] = path
    return list(unique.values())


def _raise_errno(operation: str) -> None:
    error_number = ctypes.get_errno()
    raise LandlockError(f"{operation} failed: {os.strerror(error_number)}")
