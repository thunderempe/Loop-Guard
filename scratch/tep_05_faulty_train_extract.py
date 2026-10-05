"""TEP Step 1 / script 05: ONE-TIME guarded load of TEP_Faulty_Training.RData.
- Refuses to run unless >= MIN_FREE_GB physical RAM is available (pyreadr cannot read partially; projected peak ~8 GB).
- Reports full-file structure (shape, dtypes, faults, runs/fault, samples/run) BEFORE filtering.
- Keeps only index columns + candidate-loop columns, runs 1..KEEP_RUNS per fault, as float32.
- Saves scratch/tep_cache_faulty_train_subset.pkl, deletes the full frame, reports peak memory.
"""
import ctypes, gc, sys, time
from ctypes import wintypes
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pyreadr
from tep_mem import report

MIN_FREE_GB = 8.5
KEEP_RUNS = 100
ROOT = Path(__file__).resolve().parent.parent


class MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


ms = MEMSTAT(); ms.dwLength = ctypes.sizeof(MEMSTAT)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
free_phys, free_commit = ms.ullAvailPhys / 2**30, ms.ullAvailPageFile / 2**30
print(f"pre-flight: free physical={free_phys:.2f} GB, free commit={free_commit:.2f} GB (need >= {MIN_FREE_GB} GB physical)")
if free_phys < MIN_FREE_GB and "--force" not in sys.argv:
    print("ABORT: not enough free memory. Nothing was loaded.")
    sys.exit(2)

KEEP_MEAS = [1, 4, 7, 9, 10, 12, 13, 15, 17, 19, 21, 22, 23, 26, 27, 30, 38]
KEEP_COLS = ["faultNumber", "simulationRun", "sample"] + [f"xmeas_{i}" for i in KEEP_MEAS] + [f"xmv_{i}" for i in range(1, 12)]

report("before load")
t0 = time.time()
res = pyreadr.read_r(str(ROOT / "data" / "tep" / "TEP_Faulty_Training.RData"))
key = list(res.keys())[0]
df = res[key]; del res; gc.collect()
print(f"load {time.time()-t0:.1f}s; object '{key}'; shape={df.shape}")
report("after load")
print("dtypes:", df.dtypes.value_counts().to_dict())
print("columns identical to fault-free layout:", list(df.columns[:3]), "...", list(df.columns[-3:]), f"n={df.shape[1]}")
print(f"total NaNs: {int(df.isna().sum().sum())}")
g = df.groupby("faultNumber")
print("faults:", sorted(df.faultNumber.unique().tolist()))
runs_per = g.simulationRun.nunique(); rows_per = g.size()
print(f"runs per fault: min={runs_per.min()} max={runs_per.max()};  rows per fault: min={rows_per.min()} max={rows_per.max()}")
spr = df.groupby(["faultNumber", "simulationRun"]).size()
print(f"samples per run: min={spr.min()} max={spr.max()};  sample range {df['sample'].min()}..{df['sample'].max()}")

sub = df.loc[df.simulationRun <= KEEP_RUNS, KEEP_COLS].copy()
del df; gc.collect()
for c in KEEP_COLS[3:]:
    sub[c] = sub[c].astype(np.float32)
sub.to_pickle(ROOT / "scratch" / "tep_cache_faulty_train_subset.pkl")
print(f"saved subset shape={sub.shape}, {sub.memory_usage(deep=True).sum()/2**20:.0f} MB")
del sub; gc.collect()
report("end (after freeing)")
