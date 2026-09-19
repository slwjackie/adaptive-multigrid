"""Optional FP64 C++/OpenMP stencil backend with an explicit, reproducible build.

No compiler is invoked by a solve. ``auto`` can fall back to the existing CSR
backend; ``native`` requires this library. All gains and diagonal factors are
folded into coefficients at setup, using the *source column* diagonal.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sysconfig
from time import perf_counter

import numpy as np

from .transfer import OFFSETS_9

_SOURCE = Path(__file__).with_name("native") / "stencil9.cpp"
_CACHE: dict[str, ctypes.CDLL] = {}


def library_path() -> Path:
    digest = hashlib.sha256(_SOURCE.read_bytes() + platform.platform().encode()).hexdigest()[:20]
    root = Path(os.environ.get("ANMG_NATIVE_CACHE", str(Path.home() / ".cache" / "adaptive_mg")))
    ext = ".dll" if os.name == "nt" else (".dylib" if platform.system() == "Darwin" else ".so")
    return root / f"stencil9_{digest}{ext}"


def build_native(*, compiler: str | None = None, openmp: bool = True,
                 force: bool = False) -> dict[str, object]:
    """Build with a C++17 compiler. Retry serially when OpenMP is unavailable.

    Returns the exact command(s), output, elapsed build time, and real OpenMP
    capability; an unsuccessful build raises RuntimeError, never claims speed.
    """
    target = library_path()
    if target.exists() and not force:
        lib = load_native()
        return {"path": str(target), "cached": True, "openmp": bool(lib.anmg_stencil_openmp()), "seconds": 0.0}
    command = shlex.split(compiler or os.environ.get("CXX", "c++"))
    if not command or shutil.which(command[0]) is None:
        raise RuntimeError("C++ compiler not found; use --stencil-backend csr or install a C++17 compiler")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    base = command + ["-O3", "-std=c++17", "-fPIC", "-shared", "-ffp-contract=off", str(_SOURCE), "-o", str(temporary)]
    attempts = []
    start = perf_counter()
    for omp in ([True, False] if openmp else [False]):
        cmd = base + (["-fopenmp"] if omp else [])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        attempts.append({"command": cmd, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode == 0:
            # Validate the ABI before replacing a previously usable library.
            trial = ctypes.CDLL(str(temporary))
            if trial.anmg_stencil_abi() != 1:
                raise RuntimeError("native stencil ABI validation failed")
            os.replace(temporary, target)
            _CACHE.pop(str(target), None)
            report = {"path": str(target), "cached": False, "openmp": bool(trial.anmg_stencil_openmp()),
                      "seconds": perf_counter()-start, "attempts": attempts,
                      "source_sha256": hashlib.sha256(_SOURCE.read_bytes()).hexdigest()}
            target.with_suffix(target.suffix + ".json").write_text(json.dumps(report, indent=2))
            return report
    temporary.unlink(missing_ok=True)
    raise RuntimeError("native stencil build failed:\n" + json.dumps(attempts, indent=2))


def load_native() -> ctypes.CDLL:
    path = library_path()
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    if not path.exists():
        raise RuntimeError("native stencil is not built; run python scripts/build_native_stencil.py")
    lib = ctypes.CDLL(key)
    if lib.anmg_stencil_abi() != 1:
        raise RuntimeError("native stencil ABI mismatch; rebuild the library")
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.anmg_stencil9.argtypes = [ptr, ptr, ptr, ptr] + [ctypes.c_int]*5
    lib.anmg_stencil9.restype = ctypes.c_int
    _CACHE[key] = lib
    return lib


def native_available() -> bool:
    try:
        load_native()
        return True
    except (OSError, RuntimeError):
        return False


@dataclass
class NativeStencil:
    coefficients: np.ndarray
    threads: int = 1
    parallel_min: int = 32768

    def __post_init__(self) -> None:
        self.coefficients = np.ascontiguousarray(self.coefficients, dtype=np.float64)
        if self.coefficients.ndim != 4 or self.coefficients.shape[1] != 9:
            raise ValueError("native coefficients must have shape (q,9,nx,ny)")
        if min(self.coefficients.shape) < 1 or self.threads < 1 or self.parallel_min < 1:
            raise ValueError("invalid native stencil size/thread policy")
        if not np.isfinite(self.coefficients).all():
            raise ValueError("native stencil coefficients must be finite")
        if max(self.coefficients.shape) >= 2**31:
            raise ValueError("native stencil dimension exceeds the C ABI range")
        self._library = load_native()

    @classmethod
    def from_directions(cls, values: np.ndarray, *, diagonal: np.ndarray | None = None,
                        gains: np.ndarray | None = None, threads: int = 1,
                        parallel_min: int = 32768) -> "NativeStencil":
        values = np.array(values, dtype=np.float64, copy=True)
        if values.ndim != 4 or values.shape[1] != 9:
            raise ValueError("directions must have shape (q,9,nx,ny)")
        q, _, nx, ny = values.shape
        if diagonal is not None:
            diagonal = np.asarray(diagonal, dtype=np.float64)
            if diagonal.size != nx*ny or not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
                raise ValueError("normalizing diagonal must have N finite positive entries")
            d = diagonal.reshape(nx, ny)
            for k, (di, dj) in enumerate(OFFSETS_9):
                i0, i1 = max(0, -di), min(nx, nx-di)
                j0, j1 = max(0, -dj), min(ny, ny-dj)
                values[:, k, i0:i1, j0:j1] /= d[i0+di:i1+di, j0+dj:j1+dj]
        if gains is not None:
            gains = np.asarray(gains, dtype=np.float64).reshape(-1)
            if gains.size != q or not np.isfinite(gains).all():
                raise ValueError("gain shape/values do not match the basis count")
            values *= gains[:, None, None, None]
        return cls(values, threads=threads, parallel_min=parallel_min)

    def apply(self, residual: np.ndarray, *, gate: np.ndarray | None = None) -> np.ndarray:
        q, _, nx, ny = self.coefficients.shape
        r = np.ascontiguousarray(residual, dtype=np.float64).reshape(-1)
        if r.size != nx*ny:
            raise ValueError("residual size does not match native stencil")
        # Match CSR semantics: the *output-row* gate is applied before A*d.
        g = None if gate is None else np.ascontiguousarray(gate, dtype=np.float64).reshape(-1)
        if g is not None and g.size != r.size:
            raise ValueError("gate size does not match native stencil")
        out = np.empty((q, nx*ny), dtype=np.float64)
        ptr = ctypes.POINTER(ctypes.c_double)
        code = self._library.anmg_stencil9(
            self.coefficients.ctypes.data_as(ptr), r.ctypes.data_as(ptr),
            None if g is None else g.ctypes.data_as(ptr), out.ctypes.data_as(ptr),
            nx, ny, q, self.threads, self.parallel_min)
        if code != 0:
            raise RuntimeError(f"native stencil failed with status {code}")
        return out.T
