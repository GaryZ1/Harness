"""In-sandbox launcher: rlimits, ``PR_SET_NO_NEW_PRIVS``, a seccomp filter and the proxy bridge.

This file is executed *inside* the sandbox as ``python3 -I -S _launcher.py --spec JSON -- argv...``,
so it must stay standalone: stdlib only, no package-relative imports. The package also imports it
(:mod:`harness_sandbox.seccomp` re-exports the BPF builder) so there is a single source of truth.

Why a launcher (research §6.3 technique 1, checklist §6.7 items 2 and 11):

* Codex's Linux helper pairs bubblewrap with seccomp and ``PR_SET_NO_NEW_PRIVS``. Applying the
  filter here, after bubblewrap finished its own namespace setup, lets bwrap keep working while the
  agent's process tree can no longer create nested namespaces, mount, ptrace, load BPF or modules.
* Resource limits are set on the launcher itself right before ``exec``, so they bind the command and
  everything it spawns, never the harness.
* srt removes the network namespace and reaches a host proxy over a bind-mounted Unix socket. The
  bridge below listens on ``127.0.0.1:<port>`` inside the empty network namespace and forwards each
  connection to that socket, so ``HTTP(S)_PROXY=http://127.0.0.1:3128`` works with no network.

When a bridge is needed the launcher stays resident (it must keep serving), runs the command as a
child through a second, bridge-less copy of itself (so no ``preexec_fn`` ever runs in a threaded
process), forwards SIGTERM/SIGINT/SIGHUP and exits with the child's status. Otherwise it ``exec``s
the command directly and disappears from the process tree.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import fcntl
import json
import os
import platform
import resource
import signal
import socket
import struct
import subprocess
import sys
import threading

# ----------------------------------------------------------------------------- seccomp

AUDIT_ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}
X32_SYSCALL_BIT = 0x40000000

#: Always-denied syscalls (EPERM): mounting, module/kexec loading, tracing, BPF, keyrings,
#: io_uring (its operations bypass seccomp), userfaultfd and swap/reboot.
DENY_ALWAYS = (
    "mount", "umount2", "pivot_root", "open_tree", "move_mount", "fsopen", "fsconfig", "fsmount", "fspick",
    "mount_setattr", "swapon", "swapoff", "reboot", "init_module", "finit_module", "delete_module",
    "kexec_load", "kexec_file_load", "bpf", "perf_event_open", "userfaultfd", "keyctl", "add_key",
    "request_key", "ptrace", "process_vm_readv", "process_vm_writev", "open_by_handle_at",
    "io_uring_setup", "io_uring_enter", "io_uring_register",
)
#: Denied unless nested namespaces are allowed (checklist item 19: "spawn a nested user namespace").
DENY_NAMESPACES = ("unshare", "setns")

SYSCALLS = {
    "x86_64": {
        "ptrace": 101, "mount": 165, "umount2": 166, "pivot_root": 155, "swapon": 167, "swapoff": 168,
        "reboot": 169, "init_module": 175, "delete_module": 176, "kexec_load": 246, "finit_module": 313,
        "kexec_file_load": 320, "bpf": 321, "perf_event_open": 298, "userfaultfd": 323, "keyctl": 250,
        "add_key": 248, "request_key": 249, "process_vm_readv": 310, "process_vm_writev": 311,
        "unshare": 272, "setns": 308, "clone": 56, "clone3": 435, "open_by_handle_at": 304,
        "open_tree": 428, "move_mount": 429, "fsopen": 430, "fsconfig": 431, "fsmount": 432, "fspick": 433,
        "mount_setattr": 442, "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
    },
    "aarch64": {
        "ptrace": 117, "mount": 40, "umount2": 39, "pivot_root": 41, "swapon": 224, "swapoff": 225,
        "reboot": 142, "init_module": 105, "delete_module": 106, "kexec_load": 104, "finit_module": 273,
        "kexec_file_load": 294, "bpf": 280, "perf_event_open": 241, "userfaultfd": 282, "keyctl": 219,
        "add_key": 217, "request_key": 218, "process_vm_readv": 270, "process_vm_writev": 271,
        "unshare": 97, "setns": 268, "clone": 220, "clone3": 435, "open_by_handle_at": 265,
        "open_tree": 428, "move_mount": 429, "fsopen": 430, "fsconfig": 431, "fsmount": 432, "fspick": 433,
        "mount_setattr": 442, "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
    },
}

CLONE_NS_FLAGS = 0x00020000 | 0x02000000 | 0x04000000 | 0x08000000 | 0x10000000 | 0x20000000 | 0x40000000

_BPF_LD_W_ABS = 0x00 | 0x00 | 0x20
_BPF_JEQ_K = 0x05 | 0x10 | 0x00
_BPF_JGE_K = 0x05 | 0x30 | 0x00
_BPF_JSET_K = 0x05 | 0x40 | 0x00
_BPF_RET_K = 0x06 | 0x00
RET_KILL_PROCESS = 0x80000000
RET_ALLOW = 0x7FFF0000
RET_ERRNO = 0x00050000


def _ins(code: int, jt: int, jf: int, k: int) -> bytes:
    return struct.pack("HBBI", code, jt, jf, k & 0xFFFFFFFF)


def machine() -> str:
    m = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(m, m)


def build_seccomp_program(arch: str | None = None, *, block_namespaces: bool = True) -> bytes:
    """Classic-BPF seccomp filter as raw ``struct sock_filter`` bytes.

    Wrong architecture -> kill (stops i386/x32 ABI bypasses); x32 syscall numbers -> EPERM;
    denied syscalls -> EPERM; ``clone3`` -> ENOSYS (its flags live in user memory BPF cannot read, so
    libc falls back to ``clone``); ``clone`` with any ``CLONE_NEW*`` flag -> EPERM; else allow.
    """
    arch = arch or machine()
    if arch not in SYSCALLS:
        raise ValueError(f"seccomp filter not available for architecture {arch!r}")
    table = SYSCALLS[arch]
    prog = [_ins(_BPF_LD_W_ABS, 0, 0, 4), _ins(_BPF_JEQ_K, 1, 0, AUDIT_ARCH[arch]), _ins(_BPF_RET_K, 0, 0, RET_KILL_PROCESS),
            _ins(_BPF_LD_W_ABS, 0, 0, 0)]
    if arch == "x86_64":
        prog += [_ins(_BPF_JGE_K, 0, 1, X32_SYSCALL_BIT), _ins(_BPF_RET_K, 0, 0, RET_ERRNO | errno.EPERM)]
    names = list(DENY_ALWAYS) + (list(DENY_NAMESPACES) if block_namespaces else [])
    for name in names:
        prog += [_ins(_BPF_JEQ_K, 0, 1, table[name]), _ins(_BPF_RET_K, 0, 0, RET_ERRNO | errno.EPERM)]
    if block_namespaces:
        prog += [_ins(_BPF_JEQ_K, 0, 1, table["clone3"]), _ins(_BPF_RET_K, 0, 0, RET_ERRNO | errno.ENOSYS)]
        prog += [
            _ins(_BPF_JEQ_K, 0, 3, table["clone"]),
            _ins(_BPF_LD_W_ABS, 0, 0, 16),  # low 32 bits of args[0] (little-endian) = clone flags
            _ins(_BPF_JSET_K, 0, 1, CLONE_NS_FLAGS),
            _ins(_BPF_RET_K, 0, 0, RET_ERRNO | errno.EPERM),
        ]
    prog.append(_ins(_BPF_RET_K, 0, 0, RET_ALLOW))
    return b"".join(prog)


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2


def _libc():
    return ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def set_no_new_privs() -> None:
    if sys.platform.startswith("linux"):
        if _libc().prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def install_seccomp(program: bytes) -> None:
    """Install ``program`` for this process and all descendants (requires no_new_privs)."""
    buf = ctypes.create_string_buffer(program, len(program))
    fprog = _SockFprog(len(program) // 8, ctypes.cast(buf, ctypes.c_void_p))
    if _libc().prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP) failed")


# ----------------------------------------------------------------------------- rlimits

RLIMITS = {
    "cpu": "RLIMIT_CPU", "data": "RLIMIT_DATA", "as": "RLIMIT_AS", "nproc": "RLIMIT_NPROC",
    "fsize": "RLIMIT_FSIZE", "nofile": "RLIMIT_NOFILE", "core": "RLIMIT_CORE",
}


def apply_rlimits(limits: dict) -> None:
    """``limits`` maps short names to ``[soft, hard]``; never raises a hard limit above the current one."""
    for short, (soft, hard) in limits.items():
        res = getattr(resource, RLIMITS[short], None)
        if res is None:
            continue
        cur_soft, cur_hard = resource.getrlimit(res)
        if cur_hard != resource.RLIM_INFINITY:
            hard = min(hard, cur_hard)
            soft = min(soft, hard)
        resource.setrlimit(res, (soft, hard))  # failure propagates: fail closed, never run unlimited


# ----------------------------------------------------------------------------- network bridge

SIOCGIFFLAGS, SIOCSIFFLAGS, IFF_UP = 0x8913, 0x8914, 0x1


def bring_up_loopback() -> bool:
    """A fresh network namespace has ``lo`` down (bwrap raises it; ``unshare`` does not)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            ifr = struct.pack("16sH14s", b"lo", 0, b"\0" * 14)
            flags = struct.unpack("16sH", fcntl.ioctl(s, SIOCGIFFLAGS, ifr)[:18])[1]
            if not flags & IFF_UP:
                fcntl.ioctl(s, SIOCSIFFLAGS, struct.pack("16sH14s", b"lo", flags | IFF_UP, b"\0" * 14))
        return True
    except OSError:
        return False


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _serve_conn(client: socket.socket, target: str) -> None:
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.connect(target)
    except OSError:
        client.close()
        upstream.close()
        return
    t = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
    t.start()
    _pump(client, upstream)
    t.join(timeout=300)
    client.close()
    upstream.close()


def start_bridge(port: int, unix_socket: str, host: str = "127.0.0.1") -> socket.socket:
    """Listen on ``host:port`` and forward each connection to ``unix_socket`` (the host proxy)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(128)

    def loop() -> None:
        while True:
            try:
                client, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=_serve_conn, args=(client, unix_socket), daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()
    return srv


# ----------------------------------------------------------------------------- main


def _confine(spec: dict) -> None:
    if spec.get("umask") is not None:
        os.umask(int(spec["umask"]))
    if spec.get("rlimits"):
        apply_rlimits(spec["rlimits"])
    if spec.get("no_new_privs", True) or spec.get("seccomp"):
        set_no_new_privs()
    if spec.get("seccomp"):
        install_seccomp(build_seccomp_program(block_namespaces=spec.get("block_namespaces", True)))


def _exec(argv: list[str]) -> None:
    try:
        os.execvp(argv[0], argv)
    except FileNotFoundError:
        sys.stderr.write(f"harness-sandbox: command not found: {argv[0]}\n")
        os._exit(127)
    except PermissionError:
        sys.stderr.write(f"harness-sandbox: permission denied: {argv[0]}\n")
        os._exit(126)


def main(args: list[str]) -> int:
    if len(args) < 3 or args[0] != "--spec" or "--" not in args:
        sys.stderr.write("usage: _launcher.py --spec JSON -- argv...\n")
        return 2
    spec = json.loads(args[1])
    argv = args[args.index("--") + 1:]
    if not argv:
        sys.stderr.write("harness-sandbox: empty command\n")
        return 2
    bridge = spec.get("bridge")
    if not bridge:
        try:
            _confine(spec)
        except OSError as exc:
            sys.stderr.write(f"harness-sandbox: confinement failed: {exc}\n")
            return 126
        _exec(argv)
        return 127  # pragma: no cover - exec does not return
    if spec.get("loopback"):
        bring_up_loopback()
    try:
        srv = start_bridge(int(bridge["port"]), bridge["socket"])
    except OSError as exc:
        sys.stderr.write(f"harness-sandbox: proxy bridge failed: {exc}\n")
        return 126
    inner = {k: v for k, v in spec.items() if k not in ("bridge", "loopback")}
    child = subprocess.Popen([sys.executable, "-I", "-S", os.path.abspath(__file__), "--spec", json.dumps(inner), "--", *argv])

    def forward(signum, _frame):  # pragma: no cover - signal timing
        try:
            child.send_signal(signum)
        except OSError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)
    rc = child.wait()
    srv.close()
    if rc < 0:  # re-raise the child's fatal signal so the caller sees the true cause
        signal.signal(-rc, signal.SIG_DFL)
        os.kill(os.getpid(), -rc)
        return 128 - rc  # pragma: no cover - only reached for non-fatal signals
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
