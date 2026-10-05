"""Tiny Windows memory helper (no psutil): current and peak working set in MB."""
import ctypes
from ctypes import wintypes


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
_k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL


def mem_mb():
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    ok = _k32.K32GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    if not ok:
        return float("nan"), float("nan")
    return pmc.WorkingSetSize / 2**20, pmc.PeakWorkingSetSize / 2**20


def report(tag):
    cur, peak = mem_mb()
    print(f"[mem] {tag}: current={cur:,.0f} MB  peak={peak:,.0f} MB")
