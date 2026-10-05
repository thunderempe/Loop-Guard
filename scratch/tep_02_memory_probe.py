"""TEP Step 1 / script 02: measure peak memory of loading the fault-free RData, to project Faulty_Training cost."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pyreadr
from tep_mem import report, mem_mb

ROOT = Path(__file__).resolve().parent.parent
report("start")
base = mem_mb()[0]
t0 = time.time()
res = pyreadr.read_r(str(ROOT / "data" / "tep" / "TEP_FaultFree_Training.RData"))
df = res[list(res.keys())[0]]
del res
print(f"load {time.time()-t0:.1f}s, df.memory_usage = {df.memory_usage(deep=True).sum()/2**20:,.0f} MB")
report("after load")
peak = mem_mb()[1]
print(f"load overhead peak-minus-baseline = {peak-base:,.0f} MB for {len(df):,} rows")
print(f"linear projection to 5,000,000 rows (x{5_000_000/len(df):.0f}): {(peak-base)*5_000_000/len(df)/1024:,.2f} GB")
