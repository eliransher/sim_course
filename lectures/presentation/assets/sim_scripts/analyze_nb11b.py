"""Lesson 11b: statistics, queue results and figures for the MAP-fitting deck.

Reads pkls/map_fitting_example.pkl and assets/nb11b/fits.pkl (from fit_nb11b.py).
Correlations: rho_L(i, j) = corr(X_t^i, X_{t+L}^j) for lags L = 1, 2 and powers i, j = 1, 2.
Queue: single server, arrivals from each model, exponential service with rate 1.25 (load 0.8).
Exact results from butools MAPMAP1; checked by a 2,000,000-customer simulation.
"""
import os, sys, pickle as pkl
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

ROOT = r"C:\Users\Eshel\workspace\sim_course"
sys.path[:0] = [ROOT + r"\code", ROOT + r"\code\butools2\Python", os.path.dirname(__file__)]
import butools
butools.verbose = False
from butools.map import MarginalMomentsFromMAP, LagkJointMomentsFromMAP, MarginalDistributionFromMAP
from butools.queues import MAPMAP1
from fit_nb11b import sample_erchmm

OUT = ROOT + r"\lectures\presentation\assets\nb11b"
NAVY, TEAL, AMBER, RED, GREY, GREEN = "#1F3A5F", "#0D9488", "#F59E0B", "#C0392B", "#5D6D7E", "#1E8449"
plt.rcParams.update({"font.size": 13})
MU = 1.25
PAIRS = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)]  # (lag, i, j)
LABELS = [f"lag {L}\n({i},{j})" for L, i, j in PAIRS]


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ statistics
def corr_map(D0, D1):
    D0, D1 = np.matrix(D0), np.matrix(D1)
    m = [1.0] + list(MarginalMomentsFromMAP(D0, D1, 4))
    out = []
    for L, i, j in PAIRS:
        J = np.array(LagkJointMomentsFromMAP(D0, D1, 2, L))
        out.append((J[i, j] - m[i] * m[j]) / np.sqrt((m[2 * i] - m[i] ** 2) * (m[2 * j] - m[j] ** 2)))
    return np.array(out)


def corr_trace(x):
    return np.array([np.corrcoef(x[:-L] ** i, x[L:] ** j)[0, 1] for L, i, j in PAIRS])


def ph_as_map(a, A):
    a = np.asarray(a, float).ravel(); A = np.asarray(A, float)
    return A, np.outer(-A @ np.ones(len(a)), a)


def ph_pdf(a, A, x):
    a = np.asarray(a, float).ravel(); A = np.asarray(A, float); s = -A @ np.ones(len(a))
    return np.array([a @ expm(A * xx) @ s for xx in x])


def erchmm_params(D0, D1, orders):
    first = np.cumsum([0] + list(orders[:-1])); last = first + np.array(orders) - 1
    rates = -np.diag(D0)[first]
    P = np.array([[D1[last[i], first[j]] / rates[i] for j in range(len(orders))] for i in range(len(orders))])
    from fit_nb11b import erchmm_to_map
    R0, R1 = erchmm_to_map(orders, rates, P)
    assert np.allclose(R0, D0, atol=1e-8) and np.allclose(R1, D1, atol=1e-8), "not an ER-CHMM in branch order"
    return rates, P


# ------------------------------------------------------------------ simulation (check of the exact numbers)
def ph_sample(a, S, n, rng):
    a = np.asarray(a, float).ravel(); S = np.asarray(S, float); k = len(a)
    rates = -np.diag(S); off = S.copy(); np.fill_diagonal(off, 0)
    P = np.zeros((k, k + 1)); P[:, :k] = off / rates[:, None]; P[:, k] = 1 - P[:, :k].sum(1)
    P = np.clip(P, 0, None); P /= P.sum(1, keepdims=True); C = np.cumsum(P, 1)
    a = np.clip(a, 0, None); a /= a.sum()
    state = rng.choice(k, size=n, p=a); t = np.zeros(n); alive = np.arange(n)
    while alive.size:
        st = state[alive]; t[alive] += rng.exponential(1 / rates[st])
        nxt = np.minimum((rng.random(alive.size)[:, None] > C[st]).sum(1), k)
        state[alive] = nxt; alive = alive[nxt < k]
    return t


def queue_sim(gaps, svc):
    N = len(svc); W = np.empty(N); W[0] = 0.0; w = 0.0
    for i in range(1, N):
        w = w + svc[i - 1] - gaps[i - 1]
        if w < 0: w = 0.0
        W[i] = w
    A = np.concatenate([[0.0], np.cumsum(gaps)])[:N]; D = A + W + svc
    ev_t = np.concatenate([A, D]); ev_d = np.concatenate([np.ones(N), -np.ones(N)])
    o = np.argsort(ev_t, kind="stable"); ev_t, ev_d = ev_t[o], ev_d[o]
    n = np.cumsum(ev_d)[:-1]; dt = np.diff(ev_t); c = len(dt) // 20; n, dt = n[c:], dt[c:]
    return W[N // 20:].mean(), np.bincount(n.astype(int), weights=dt) / dt.sum()


if __name__ == "__main__":
    D0, D1, trace = pkl.load(open(ROOT + r"\pkls\map_fitting_example.pkl", "rb"))
    fits = pkl.load(open(os.path.join(OUT, "fits.pkl"), "rb"))
    models = {"True process": (D0, D1)}
    for k, f in fits.items():
        models[k] = ph_as_map(f["alpha"], f["A"]) if f["kind"] == "PH" else (f["D0"], f["D1"])

    S0, S1 = np.array([[-MU]]), np.array([[MU]])
    K = 60
    res = {}
    for name, (M0, M1) in models.items():
        M0, M1 = np.matrix(M0), np.matrix(M1)
        m1, m2 = MarginalMomentsFromMAP(M0, M1, 2)
        st = MAPMAP1(M0, M1, S0, S1, "stMoms", 1, "ncDistr", K)
        res[name] = dict(states=M0.shape[0], mean=m1, scv=(m2 - m1 ** 2) / m1 ** 2, corr=corr_map(M0, M1),
                         Wq=float(np.ravel(st[0])[0]) - 1 / MU, pmf=np.ravel(st[1]))
    res["Trace (data)"] = dict(states=None, mean=trace.mean(), scv=trace.var() / trace.mean() ** 2, corr=corr_trace(trace))
    t = res["True process"]["pmf"]
    for name, r in res.items():
        if "pmf" in r:
            r["SAE"] = np.abs(r["pmf"] - t).sum()
            r["err"] = r["Wq"] / res["True process"]["Wq"] - 1

    # simulation check, common service times
    rng = np.random.default_rng(1111); N = 2_000_000
    svc = rng.exponential(1 / MU, N)
    sims = {}
    sims["True process"] = sample_erchmm([3, 6], [6.0, 3.0], np.array([[0.9, 0.1], [0.2, 0.8]]), N, rng)[0]
    sims["PH-8"] = ph_sample(fits["PH-8"]["alpha"], fits["PH-8"]["A"], N, rng)
    for k, f in fits.items():
        if f["kind"] == "MAP":
            rates, P = erchmm_params(f["D0"], f["D1"], f["orders"])
            sims[k] = sample_erchmm(f["orders"], rates, P, N, rng)[0]
    for name, g in sims.items():
        Wq, pmf = queue_sim(g, svc)
        res[name]["Wq_sim"] = Wq

    for name, r in res.items():
        print(f"{name:13s} st={r['states']} mean={r['mean']:.3f} scv={r['scv']:.3f} "
              f"corr={np.round(r['corr'], 3)} Wq={r.get('Wq', np.nan):.3f} sim={r.get('Wq_sim', np.nan):.3f} "
              f"err={r.get('err', np.nan):+.1%} SAE={r.get('SAE', np.nan):.3f}")
    for k, f in fits.items():
        print(k, "fit time", round(f["time"], 1), "s  logli", round(float(f["logli"]), 4))
    pkl.dump(res, open(os.path.join(OUT, "results.pkl"), "wb"))

    # ------------------------------------------------------------------ figures
    # 1. the trace: a stretch of the sequence + consecutive pairs
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    n0 = 250
    axs[0].bar(np.arange(n0), trace[:n0], width=1.0, color=TEAL)
    axs[0].set_xlabel("interval number"); axs[0].set_ylabel("interval length")
    axs[0].set_title("First 250 intervals: busy and quiet stretches", fontsize=14); clean(axs[0])
    axs[1].scatter(trace[:-1][:6000], trace[1:][:6000], s=3, alpha=0.35, color=NAVY)
    axs[1].set_xlabel("$X_t$"); axs[1].set_ylabel("$X_{t+1}$"); axs[1].set_xlim(0, 6); axs[1].set_ylim(0, 6)
    axs[1].set_title("Consecutive intervals", fontsize=14); clean(axs[1])
    fig.tight_layout(); save(fig, "trace.png")

    # 2. marginal: data histogram + true marginal density
    x = np.linspace(0.005, 6, 400)
    a_t, A_t = MarginalDistributionFromMAP(np.matrix(D0), np.matrix(D1))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.hist(trace, bins=300, density=True, color="#9FB8D0", alpha=0.75, label="50,000 intervals")
    ax.plot(x, ph_pdf(a_t, A_t, x), color=NAVY, lw=2.6, label="true marginal density")
    ax.set_xlim(0, 6); ax.set_xlabel("interval length"); ax.set_ylabel("density"); ax.legend(frameon=False); clean(ax)
    save(fig, "marginal.png")

    # 3. marginal fits
    show = [("PH-8", "PH, 8 phases"), ("MAP-2", "MAP, 2 states"), ("MAP-4", "MAP, 4 states"), ("MAP-8", "MAP, 8 states")]
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, sharey=True)
    for ax, (k, lab) in zip(axs.ravel(), show):
        M0, M1 = models[k]; a_f, A_f = MarginalDistributionFromMAP(np.matrix(M0), np.matrix(M1))
        ax.hist(trace, bins=300, density=True, color="#9FB8D0", alpha=0.7)
        ax.plot(x, ph_pdf(a_t, A_t, x), "--", color=NAVY, lw=1.6, label="true")
        ax.plot(x, ph_pdf(a_f, A_f, x), color=AMBER if k == "PH-8" else TEAL, lw=2.6, label="fit")
        ax.set_title(f"{lab}   (SCV {res[k]['scv']:.2f})", fontsize=13); ax.set_xlim(0, 5); clean(ax)
    axs[0, 0].legend(frameon=False)
    for ax in axs[1]: ax.set_xlabel("interval length")
    fig.suptitle(f"Marginal fits  (true SCV {res['True process']['scv']:.2f})", fontsize=15)
    fig.tight_layout(); save(fig, "marginal_fits.png")

    # 4. correlations
    keys = [("True process", NAVY, "true process"), ("Trace (data)", GREY, "data (50,000)"), ("PH-8", AMBER, "PH, 8 phases"),
            ("MAP-2", "#99F6E4", "MAP, 2 states"), ("MAP-4", "#2DD4BF", "MAP, 4 states"), ("MAP-8", "#0F766E", "MAP, 8 states")]
    fig, ax = plt.subplots(figsize=(13, 4.8)); w = 0.13; xs = np.arange(len(PAIRS))
    for q, (k, c, lab) in enumerate(keys):
        ax.bar(xs + (q - 2.5) * w, res[k]["corr"], w, color=c, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels(LABELS); ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(r"corr$(X_t^{\,i},\ X_{t+L}^{\,j})$"); ax.legend(frameon=False, ncol=3, fontsize=11); clean(ax)
    ax.set_title("Autocorrelations: lags 1-2, powers 1-2", fontsize=14)
    ax.text(7.45, 0.33, "PH, 8 phases and MAP, 2 states:\nall autocorrelations are 0", ha="right",
            fontsize=12, color="#B45309", fontweight="bold")
    save(fig, "corr_fits.png")

    # 5. queue-length distribution
    fig, ax = plt.subplots(figsize=(11, 4.6)); n = np.arange(26)
    ax.bar(n, res["True process"]["pmf"][:26], color="#9FB8D0", label="true process")
    for k, c, lab, mk in [("PH-8", AMBER, "PH, 8 phases (no dependence)", "o"), ("MAP-2", "#2DD4BF", "MAP, 2 states", "s"),
                          ("MAP-8", "#0F766E", "MAP, 8 states", "^")]:
        ax.plot(n, res[k]["pmf"][:26], marker=mk, color=c, lw=2, ms=5, label=lab)
    ax.set_xlabel("number of customers in the system"); ax.set_ylabel("long-run fraction of time")
    ax.legend(frameon=False); clean(ax); save(fig, "queue_pmf.png")
    print("figures saved")
