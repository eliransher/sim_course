import numpy as np
rng = np.random.default_rng(10)
N, lam = 2_000_000, 0.7
def run(S):
    A = np.cumsum(rng.exponential(1/lam, N))
    W = np.empty(N); W[0] = 0.0
    # Lindley recursion
    ia = np.diff(A)
    w = 0.0
    for i in range(1, N):
        w = max(0.0, w + S[i-1] - ia[i-1]); W[i] = w
    D = A + W + S
    ev_t = np.concatenate([A, D]); ev_d = np.concatenate([np.ones(N), -np.ones(N)])
    o = np.argsort(ev_t, kind="stable"); ev_t, ev_d = ev_t[o], ev_d[o]
    n = np.cumsum(ev_d)[:-1]; dt = np.diff(ev_t)
    cut = int(0.05 * len(dt))  # drop warm-up
    n, dt = n[cut:], dt[cut:]
    pmf = np.bincount(n.astype(int), weights=dt) / dt.sum()
    return dict(sojourn=(W + S)[N//20:].mean(), wait=W[N//20:].mean(), L=(n*dt).sum()/dt.sum(), pmf=pmf)
# hyper: a=(0.05,0.95), rates 0.1 and 1.9
br = rng.random(N) < 0.05
Sh = np.where(br, rng.exponential(10, N), rng.exponential(1/1.9, N))
Se = rng.gamma(5, 0.2, N)
res = {"hyper": run(Sh), "erlang": run(Se)}
for k, r in res.items():
    print(k, "W=%.3f Wq=%.3f L=%.3f P0..5=%s maxn=%d" % (r["sojourn"], r["wait"], r["L"], np.round(r["pmf"][:6], 4), len(r["pmf"])-1))
np.save(r"C:\Users\Eshel\AppData\Local\Temp\claude\C--Users-Eshel-workspace-sim-course\1db889cb-ab5e-41fe-b9a0-2a3195bb46a9\scratchpad\nb10_pmf.npy",
        {"hyper": res["hyper"]["pmf"], "erlang": res["erlang"]["pmf"]}, allow_pickle=True)
# PK check
for name, m2 in [("hyper", 9.526+1), ("erlang", 1.2)]:
    wq = lam*m2/(2*(1-0.7)); print(name, "PK: Wq=%.3f W=%.3f L=%.3f" % (wq, wq+1, lam*(wq+1)))
