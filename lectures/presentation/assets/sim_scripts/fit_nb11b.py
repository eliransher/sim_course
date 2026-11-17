"""Lesson 11b (MAP fitting): build the true process, sample the trace, run the fits.

True process: ER-CHMM with two regimes ("busy" / "quiet").
  busy : Erlang(3, rate 6), mean 0.5      quiet: Erlang(6, rate 3), mean 2
  after each interval the next regime is drawn from P = [[0.9, 0.1], [0.2, 0.8]]
  stationary regime probabilities (2/3, 1/3)  ->  mean interval 1.
Fits: PHFromTrace (8 phases, ignores dependence) and MAPFromTrace with Erlang
branch orders [1,1], [2,2], [3,3], [4,4] (2, 4, 6, 8 states).
Outputs: pkls/map_fitting_example.pkl  and  assets/nb11b/fits.pkl
"""
import os, sys, time, pickle as pkl
import numpy as np

ROOT = r"C:\Users\Eshel\workspace\sim_course"
sys.path[:0] = [ROOT + r"\code", ROOT + r"\code\butools2\Python"]
from butools.fitting import MAPFromTrace, PHFromTrace

OUT = ROOT + r"\lectures\presentation\assets\nb11b"
os.makedirs(OUT, exist_ok=True)


def erchmm_to_map(orders, rates, P):
    """D0, D1 of an ER-CHMM: branch i is Erlang(orders[i], rates[i]); after an interval
    of branch i the next interval uses branch j with probability P[i, j]."""
    n = sum(orders); first = np.cumsum([0] + list(orders[:-1])); last = first + np.array(orders) - 1
    D0 = np.zeros((n, n)); D1 = np.zeros((n, n))
    for i, (k, r) in enumerate(zip(orders, rates)):
        for s in range(first[i], last[i] + 1):
            D0[s, s] = -r
            if s < last[i]:
                D0[s, s + 1] = r
        for j in range(len(orders)):
            D1[last[i], first[j]] = r * P[i, j]
    return D0, D1


def sample_erchmm(orders, rates, P, n, rng, b0=0):
    C = np.cumsum(P, 1); u = rng.random(n); b = np.empty(n, int); b[0] = b0
    for i in range(1, n):
        b[i] = np.searchsorted(C[b[i - 1]], u[i], side="right")
    b = np.minimum(b, len(orders) - 1)
    return rng.gamma(np.asarray(orders, float)[b], 1 / np.asarray(rates, float)[b]), b


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T_ORD, T_RATE, T_P = [3, 6], [6.0, 3.0], np.array([[0.9, 0.1], [0.2, 0.8]])
    D0, D1 = erchmm_to_map(T_ORD, T_RATE, T_P)
    pi = np.array([2 / 3, 1 / 3])
    b0 = rng.choice(2, p=pi)
    trace, _ = sample_erchmm(T_ORD, T_RATE, T_P, 50_000, rng, b0)
    pkl.dump((D0, D1, trace), open(ROOT + r"\pkls\map_fitting_example.pkl", "wb"))
    print("trace mean", trace.mean(), "scv", trace.var() / trace.mean() ** 2, flush=True)

    fits = {}
    t0 = time.time(); a, A, ll = PHFromTrace(trace, 8)
    fits["PH-8"] = dict(kind="PH", alpha=np.array(a).ravel(), A=np.array(A), logli=ll, time=time.time() - t0)
    print("PH-8", round(fits["PH-8"]["time"], 1), "s", ll, flush=True)
    for orders in ([1, 1], [2, 2], [3, 3], [4, 4]):
        t0 = time.time(); F0, F1, ll = MAPFromTrace(trace, orders)
        key = f"MAP-{sum(orders)}"
        fits[key] = dict(kind="MAP", orders=orders, D0=np.array(F0), D1=np.array(F1), logli=ll, time=time.time() - t0)
        print(key, orders, round(fits[key]["time"], 1), "s", ll, flush=True)
        pkl.dump(fits, open(os.path.join(OUT, "fits.pkl"), "wb"))
    print("done")
