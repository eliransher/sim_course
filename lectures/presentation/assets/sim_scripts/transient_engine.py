"""Vectorized transient simulation of a tandem line of M/M/c stations (Poisson or saturated input),
with piecewise-constant rates, via uniformization/thinning across R replications."""
import numpy as np

def piecewise(schedule):
    """schedule: dict {start_time: rate} -> function t(array) -> rate(array)."""
    ks = np.array(sorted(schedule)); vs = np.array([schedule[k] for k in ks])
    return lambda t: vs[np.searchsorted(ks, t, side="right") - 1]

def simulate(R, horizon, caps, mu_sched, lam_sched=None, saturated=False, grid=None, rng=None):
    """caps: servers per station. mu_sched: list of dict schedules (per-server rate).
    lam_sched: dict schedule of arrival rate (ignored if saturated: station 1 always busy).
    Returns rec (R, len(grid), K) numbers in system at grid times, and completions (R,)."""
    rng = rng or np.random.default_rng()
    K = len(caps); caps = np.array(caps)
    mus = [piecewise(m) for m in mu_sched]
    lam = piecewise(lam_sched) if lam_sched is not None else None
    lam_max = 0.0 if saturated else max(lam_sched.values())
    Lam = lam_max + sum(c * max(m.values()) for c, m in zip(caps, mu_sched))
    n = np.zeros((R, K), dtype=np.int64)
    T = np.zeros(R); done = np.zeros(R, dtype=np.int64)
    grid = np.array([] if grid is None else grid, dtype=float)
    rec = np.zeros((R, len(grid), K), dtype=np.int64)
    alive = np.ones(R, bool)
    while alive.any():
        idx = np.flatnonzero(alive)
        Tn = T[idx] + rng.exponential(1 / Lam, idx.size)
        for g, tg in enumerate(grid):                        # state on [T, Tn) holds at grid times in it
            m = (T[idx] < tg) & (Tn >= tg)
            if m.any(): rec[idx[m], g] = n[idx[m]]
        fin = Tn >= horizon
        alive[idx[fin]] = False
        idx, Tn = idx[~fin], Tn[~fin]
        T[idx] = Tn
        if idx.size == 0: break
        # event rates at Tn
        rates = []
        if not saturated:
            rates.append(lam(Tn))
        for k in range(K):
            busy = caps[k] if (saturated and k == 0) else np.minimum(n[idx, k], caps[k])
            rates.append(busy * mus[k](Tn))
        rates = np.vstack(rates).T / Lam                     # (r, E)
        u = rng.random(idx.size)
        ev = (u[:, None] > np.cumsum(rates, 1)).sum(1)        # E = dummy
        off = 0
        if not saturated:
            a = idx[ev == 0]; n[a, 0] += 1; off = 1
        for k in range(K):
            s = idx[ev == off + k]
            if s.size == 0: continue
            if not (saturated and k == 0): n[s, k] -= 1
            if k + 1 < K: n[s, k + 1] += 1
            else: done[s] += 1
    for g, tg in enumerate(grid):                             # grid times after last event
        m = (T < tg) | (tg == 0)
        rec[m, g] = n[m] if tg > 0 else 0
    return rec, done
