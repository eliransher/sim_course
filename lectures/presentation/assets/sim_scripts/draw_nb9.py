import os, sys
os.chdir(r"C:\Users\Eshel\workspace\sim_course\lectures\notebooks")
sys.path.append("../../code")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import utils
OUT = r"C:\Users\Eshel\workspace\sim_course\lectures\presentation\assets\nb9"
NAVY, BLUE, GREEN, ORANGE, GREY = "#1F3A5F", "#2471A3", "#1E8449", "#D35400", "#5D6D7E"
plt.rcParams.update({"font.size": 14})

def node(ax, x, y, label, w=1.3, h=0.8, fc="white", ec=NAVY, fs=18, bold=True):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fc, ec=ec, lw=2.6, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=ec, zorder=4)
def arrow(ax, p, q, color=NAVY, label=None, lxy=None, lcolor=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, lw=2.4, color=color, zorder=2))
    if label:
        ax.text(*lxy, label, fontsize=16, color=lcolor or color, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight"); plt.close(fig)

# 1. split network
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.set_aspect("equal"); ax.axis("off")
node(ax, 0, 0, "Arrivals", w=1.8, ec=BLUE, fs=16)
node(ax, 3.6, 1.1, "S1"); node(ax, 3.6, -1.1, "S2")
node(ax, 7.2, 0, "Exit", w=1.4, ec=GREEN, fs=16)
arrow(ax, (0.9, 0.2), (2.95, 1.0), label="p", lxy=(1.8, 0.95))
arrow(ax, (0.9, -0.2), (2.95, -1.0), label="1 − p", lxy=(1.8, -0.95))
arrow(ax, (4.25, 1.0), (6.5, 0.2), GREEN); arrow(ax, (4.25, -1.0), (6.5, -0.2), GREEN)
ax.text(3.6, 1.85, "mean service 0.75", ha="center", fontsize=13, color=GREY)
ax.text(3.6, -1.95, "mean service 1.15", ha="center", fontsize=13, color=GREY)
ax.text(0, -0.75, "mean inter-arrival 0.95", ha="center", fontsize=13, color=GREY)
ax.set_xlim(-1.3, 8.1); ax.set_ylim(-2.3, 2.2); save(fig, "net_split_clean.png")

# 2. server mix
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.set_aspect("equal"); ax.axis("off")
node(ax, 0, 0, "Arrivals", w=1.8, ec=BLUE, fs=16)
node(ax, 2.9, 0, "FIFO queue", w=2.5, fs=16)
node(ax, 6.2, 1.1, r"slow × $c_s$", w=2.0, ec=ORANGE, fs=16)
node(ax, 6.2, -1.1, r"fast × $c_f$", w=2.0, ec=BLUE, fs=16)
node(ax, 9.3, 0, "Exit", w=1.4, ec=GREEN, fs=16)
arrow(ax, (0.9, 0), (1.65, 0))
arrow(ax, (4.15, 0.2), (5.2, 0.95)); arrow(ax, (4.15, -0.2), (5.2, -0.95))
arrow(ax, (7.2, 0.95), (8.6, 0.2), GREEN); arrow(ax, (7.2, -0.95), (8.6, -0.2), GREEN)
ax.text(6.2, 1.85, "cost 2 · mean service 1.45", ha="center", fontsize=13, color=GREY)
ax.text(6.2, -1.95, "cost 6 · mean service 0.55", ha="center", fontsize=13, color=GREY)
ax.text(0, -0.75, "mean inter-arrival 0.34", ha="center", fontsize=13, color=GREY)
ax.set_xlim(-1.3, 10.2); ax.set_ylim(-2.3, 2.2); save(fig, "net_mix_clean.png")

# 3. production line
fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_aspect("equal"); ax.axis("off")
node(ax, -0.3, 0, "Raw material\n(infinite)", w=2.9, h=1.0, ec=GREY, fs=14)
node(ax, 3.0, 0, "S1"); node(ax, 6.0, 1.1, "S2"); node(ax, 6.0, -1.1, "S3"); node(ax, 9.0, 0, "S4")
node(ax, 11.5, 0, "Exit", w=1.4, ec=GREEN, fs=16)
arrow(ax, (1.15, 0), (2.35, 0))
arrow(ax, (3.65, 0.2), (5.35, 1.0), label="0.55", lxy=(4.4, 0.95))
arrow(ax, (3.65, -0.2), (5.35, -1.0), label="0.45", lxy=(4.4, -0.95))
arrow(ax, (6.65, 1.0), (8.35, 0.2)); arrow(ax, (6.65, -1.0), (8.35, -0.2))
arrow(ax, (9.65, 0), (10.8, 0), GREEN)
for (x, y, m, dy) in [(3.0, 0, "0.62", -0.75), (6.0, 1.1, "0.95", 0.75), (6.0, -1.1, "0.85", -0.75), (9.0, 0, "0.90", -0.75)]:
    ax.text(x, y + dy, f"mean {m}", ha="center", va="center", fontsize=13, color=GREY)
ax.set_xlim(-2.0, 12.4); ax.set_ylim(-2.2, 2.2); save(fig, "net_tp_clean.png")

# 4. split results, log y
split = utils.evaluate_split_proportion_optimization(num_customers=2_500, seed=2026)
r = split["results"]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(r["split_to_S1"], r["mean_waiting_time"], "o-", color=BLUE, lw=2.2, label="mean waiting time")
ax.plot(r["split_to_S1"], r["p90_waiting_time"], "s-", color=ORANGE, lw=2.2, label="90th percentile")
ax.axvline(0.6, color="#C0392B", ls="--", lw=1.8, label="best split p = 0.6")
ax.set_yscale("log"); ax.set_xlabel("p = proportion routed to S1"); ax.set_ylabel("waiting time (log scale)")
ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
save(fig, "split_res_log.png")

# 5. server mix results, log x
mix = utils.evaluate_server_mix_optimization(num_customers=2_500, seed=3030)
m = mix["results"].sort_values("objective", ascending=False)
labels = [f"{a} slow + {b} fast" for a, b in zip(m["slow_servers"], m["fast_servers"])]
fig, ax = plt.subplots(figsize=(10, 5.2))
colors = [GREEN if i == len(m) - 1 else "#A9CCE3" for i in range(len(m))]
ax.barh(labels, m["objective"], color=colors)
for y, v in enumerate(m["objective"]):
    ax.text(v * 1.08, y, f"{v:,.1f}", va="center", fontsize=12, color=NAVY)
ax.set_xscale("log"); ax.set_xlim(8, 20000); ax.set_xlabel("objective J = staffing cost + 3 × mean wait (log scale)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(alpha=0.25, axis="x", which="both")
save(fig, "mix_res_log.png")
print("ok")

# 6. throughput results, tiers coloured
tp = utils.evaluate_throughput_capacity_optimization(seed=4040, production_horizon=15_000, total_servers=7)
t = tp["results"].head(12)
labels = [f"({a},{b},{c},{e})" for a, b, c, e in zip(t.S1_servers, t.S2_servers, t.S3_servers, t.S4_servers)]
cols = [GREEN if v > 2 else (BLUE if v > 1.5 else ORANGE) for v in t.throughput]
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(labels, t.throughput, color=cols)
for x, v in enumerate(t.throughput):
    ax.text(x, v + 0.04, f"{v:.2f}", ha="center", fontsize=12, color=NAVY)
ax.set_ylabel("throughput (completed / time unit)"); ax.set_xlabel("allocation (S1, S2, S3, S4)")
ax.set_ylim(0, 2.6); plt.xticks(rotation=35)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(alpha=0.25, axis="y")
save(fig, "tp_res_clean.png")
print("ok tp")
