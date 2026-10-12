import sys, pickle as pkl, time, numpy as np
SP = r"C:\Users\Eshel\AppData\Local\Temp\claude\C--Users-Eshel-workspace-sim-course\1db889cb-ab5e-41fe-b9a0-2a3195bb46a9\scratchpad"
sys.path.insert(0, SP)
from transient_engine import simulate, piecewise
from scipy.linalg import expm
rng = np.random.default_rng(1314)
out = {}

def exact_mm1(lam_s, mu_s, hours=24, M=200):
    """P(N(t)=k) at integer hours for an M/M/1 with hourly piecewise rates, from empty."""
    lam, mu = piecewise(lam_s), piecewise(mu_s)
    p = np.zeros(M); p[0] = 1; res = [p.copy()]
    for h in range(hours):
        l, m = float(lam(np.array([h + 0.5]))[0]), float(mu(np.array([h + 0.5]))[0])
        Q = np.zeros((M, M))
        for i in range(M - 1): Q[i, i + 1] = l; Q[i + 1, i] = m
        np.fill_diagonal(Q, -Q.sum(1))
        p = p @ expm(Q); res.append(p.copy())
    return np.array(res)

t0 = time.time()
# ---- NB13 A: M/M/1 lam=mu=1, t=5
rec, _ = simulate(5000, 5.0, [1], [{0: 1}], {0: 1}, grid=[5.0], rng=rng)
n5 = rec[:, 0, 0]
ex5 = exact_mm1({0: 1}, {0: 1}, hours=5)[5]
out["A"] = dict(n5=n5, exact=ex5)
print("A: sim P(N>=11)=%.4f exact=%.4f  E sim %.3f exact %.3f max %d" % ((n5 >= 11).mean(), ex5[11:].sum(), n5.mean(), (np.arange(200) * ex5).sum(), n5.max()))

# ---- NB13 B/C: daily schedule
lam_s = {0: 1, 12: 0.5, 20: 0.9}
opts = {"original": {0: 1}, "00-02": {0: 2, 2: 1}, "04-06": {0: 1, 4: 2, 6: 1}, "10-12": {0: 1, 10: 2, 12: 1},
        "12-14": {0: 1, 12: 2, 14: 1}, "20-22": {0: 1, 20: 2, 22: 1}}
grid = np.arange(25.0)
out["B"] = {}
for name, mu_s in opts.items():
    rec, _ = simulate(10_000, 24.0, [1], [mu_s], lam_s, grid=grid, rng=rng)
    N = rec[:, :, 0]
    ex = exact_mm1(lam_s, mu_s)
    pm = np.stack([np.bincount(N[:, g], minlength=60)[:60] / N.shape[0] for g in range(25)])
    out["B"][name] = dict(EN=N.mean(0), pmf=pm, EN_exact=(ex * np.arange(200)).sum(1), pmf_exact=ex[:, :60])
    print("B %-8s avg E[N] sim %.3f exact %.3f  peak %.2f at h%d" % (name, N.mean(0).mean(), out["B"][name]["EN_exact"].mean(),
          N.mean(0).max(), N.mean(0).argmax()))
print("t=%.0fs" % (time.time() - t0))

# ---- NB14 D: tandem, Poisson 1/20, mu 1/10,1/15,1/60, horizon 240
mu3 = [{0: 1 / 10}, {0: 1 / 15}, {0: 1 / 60}]
g240 = np.arange(241.0)
out["D"] = {}
for caps in [(1, 2, 3), (2, 2, 2)]:
    rec, _ = simulate(20_000, 240.0, list(caps), mu3, {0: 1 / 20}, grid=g240, rng=rng)
    EN = rec.mean(0)                      # (241, 3)
    P6 = (rec >= 6).mean(0); P5 = (rec >= 5).mean(0)
    out["D"][caps] = dict(EN=EN, P6=P6, P5=P5)
    print("D", caps, "avg per station", np.round(EN.mean(0), 3), "mean", EN.mean().round(3), "maxP6", np.round(P6.max(0), 3), "maxP5", np.round(P5.max(0), 3))
print("t=%.0fs" % (time.time() - t0))

# ---- NB14 E: saturated throughput, mu 1/10,1/45,1/55, 480 min
muE = [{0: 1 / 10}, {0: 1 / 45}, {0: 1 / 55}]
out["E"] = {}
for caps in [(2, 3, 3), (3, 3, 2)]:
    _, done = simulate(20_000, 480.0, list(caps), muE, saturated=True, rng=rng)
    out["E"][caps] = done
    print("E", caps, "mean %.2f sd %.2f P(>30)=%.4f" % (done.mean(), done.std(), (done > 30).mean()))

# ---- NB14 F: machine types, mu 1/22,1/25,1/20, 480 min
muF = [{0: 1 / 22}, {0: 1 / 25}, {0: 1 / 20}]
out["F"] = {}
for caps in [(2, 2, 2), (2, 2, 1), (2, 2, 3), (1, 2, 1)]:
    _, done = simulate(20_000, 480.0, list(caps), muF, saturated=True, rng=rng)
    out["F"][caps] = done
    print("F", caps, "E[completions] %.2f sd %.2f" % (done.mean(), done.std()))
print("t=%.0fs" % (time.time() - t0))
pkl.dump(out, open(SP + r"\nb13_14.pkl", "wb"))
