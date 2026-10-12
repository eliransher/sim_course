import os, sys, time, pickle as pkl
import numpy as np
os.chdir(r"C:\Users\Eshel\workspace\sim_course\lectures\notebooks")
sys.path[:0] = ["../../code", "../../code/butools2/Python"]
from butools.fitting import PHFromTrace
np.random.seed(11)
a, S, services = pkl.load(open("../../pkls/fitting_example.pkl", "rb"))
SP = r"C:\Users\Eshel\AppData\Local\Temp\claude\C--Users-Eshel-workspace-sim-course\1db889cb-ab5e-41fe-b9a0-2a3195bb46a9\scratchpad"
out = {}
for size in [5, 10, 15, 20]:
    t0 = time.time()
    res = PHFromTrace(services, size)
    dt = time.time() - t0
    out[size] = (np.array(res[0]).flatten(), np.array(res[1]), dt)
    print(size, round(dt, 1), flush=True)
    pkl.dump(out, open(os.path.join(SP, "nb11_em.pkl"), "wb"))
