"""Seccomp filter used by every Linux runner (re-exported from the standalone launcher).

Codex's Linux sandbox pairs bubblewrap with seccomp and ``PR_SET_NO_NEW_PRIVS`` (research §6.2).
The filter lives in :mod:`harness_sandbox._launcher` because that file runs inside the sandbox with
``python -I -S`` and cannot import this package; this module is the importable, documented face.
"""

from ._launcher import (
    AUDIT_ARCH,
    CLONE_NS_FLAGS,
    DENY_ALWAYS,
    DENY_NAMESPACES,
    SYSCALLS,
    build_seccomp_program,
    install_seccomp,
    machine,
    set_no_new_privs,
)

__all__ = ["AUDIT_ARCH", "CLONE_NS_FLAGS", "DENY_ALWAYS", "DENY_NAMESPACES", "SYSCALLS", "build_seccomp_program",
           "install_seccomp", "machine", "set_no_new_privs"]
