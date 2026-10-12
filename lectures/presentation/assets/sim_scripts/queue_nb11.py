import os, pickle as pkl
import numpy as np
SP = r"C:\Users\Eshel\AppData\Local\Temp\claude\C--Users-Eshel-workspace-sim-course\1db889cb-ab5e-41fe-b9a0-2a3195bb46a9\scratchpad"
ROOT = r"C:\Users\Eshel\workspace\sim_course"
rng = np.random.default_rng(1111)

def ph_sample(a, S, n):
    a = np.asarray(a, float).ravel(); S = np.asarray(S, float); k = len(a)
    rates = -np.diag(S)
    P = np.zeros((k, k + 1))                      # jump probs incl. absorption (col k)
    off = S.copy(); np.fill_diagonal(off, 0)
    P[:, :k] = off / rates[:, None]; P[:, k] = 1 - P[:, :k].sum(1)
    P = np.clip(P, 0, None); P /= P.sum(1, keepdims=True); C = np.cumsum(P, 1)
    a = np.clip(a, 0, None); a /= a.sum()
    state = rng.choice(k, size=n, p=a); t = np.zeros(n); alive = np.arange(n)
    while alive.size:
        st = state[alive]
        t[alive] += rng.exponential(1 / rates[st])
        u = rng.random(alive.size)
        nxt = (u[:, None] > C[st]).sum(1)
        nxt = np.minimum(nxt, k)
        state[alive] = nxt
        alive = alive[nxt < k]
    return t

def mg1(arr_gaps, S):
    N = len(S); W = np.empty(N); W[0] = 0.0; w = 0.0
    for i in range(1, N):
        w = w + S[i-1] - arr_gaps[i-1]
        if w < 0: w = 0.0
        W[i] = w
    A = np.concatenate([[0.0], np.cumsum(arr_gaps)])[:N]
    D = A + W + S
    ev_t = np.concatenate([A, D]); ev_d = np.concatenate([np.ones(N), -np.ones(N)])
    o = np.argsort(ev_t, kind="stable"); ev_t, ev_d = ev_t[o], ev_d[o]
    n = np.cumsum(ev_d)[:-1]; dt = np.diff(ev_t)
    c = len(dt) // 20; n, dt = n[c:], dt[c:]
    pmf = np.bincount(n.astype(int), weights=dt) / dt.sum()
    return dict(Wq=W[N//20:].mean(), L=(n * dt).sum() / dt.sum(), pmf=pmf)

N, lam = 2_000_000, 0.9
a, S, data = pkl.load(open(ROOT + r"\pkls\fitting_example.pkl", "rb"))
em = pkl.load(open(os.path.join(SP, "nb11_em.pkl"), "rb"))
gaps = rng.exponential(1 / lam, N)                 # common arrivals for all fits
U = None
svc = {"true": ph_sample(a, S, N), "exponential": rng.exponential(data.mean(), N)}
for k in sorted(em):
    svc[f"PH-{k}"] = ph_sample(em[k][0], em[k][1], N)
res = {}
for name, s in svc.items():
    r = mg1(gaps, s); r["mean"] = s.mean(); r["scv"] = s.var() / s.mean()**2
    res[name] = r
t = res["true"]["pmf"]
for name, r in res.items():
    p = r["pmf"]; m = max(len(p), len(t))
    r["SAE"] = np.abs(np.pad(p, (0, m - len(p))) - np.pad(t, (0, m - len(t)))).sum()
    print(f"{name:12s} mean={r['mean']:.3f} scv={r['scv']:.3f} Wq={r['Wq']:.2f} L={r['L']:.2f} SAE={r['SAE']:.4f}")
pkl.dump(res, open(os.path.join(SP, "nb11_queue.pkl"), "wb"))
