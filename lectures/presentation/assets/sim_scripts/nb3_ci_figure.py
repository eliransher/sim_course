# -*- coding: utf-8 -*-
import os, sys
HERE = r"C:\Users\Eshel\workspace\sim_course\lectures\presentation"
sys.path.insert(0, HERE)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deckbuilder import Deck, make_formula
A = os.path.join(HERE, "assets", "nb3")

# CI lengths printed by notebook cell 25 (10 replications, 95%)
ns = np.array([2500, 50000, 500000]); L = np.array([0.6050, 0.2043, 0.0455])
fig, ax = plt.subplots(figsize=(9, 4.6))
xx = np.geomspace(1800, 7e5, 100)
ax.plot(xx, L[0] * np.sqrt(ns[0] / xx), ls="--", lw=2, color="#9AA8B6", label=r"reference $\propto 1/\sqrt{n}$")
ax.plot(ns, L, "-o", lw=2, ms=10, color="#2E86AB", mec="white", mew=2, label="CI length (95%, 10 runs)", zorder=3)
for n, l in zip(ns, L):
    ax.annotate(f"{l:.3f}", (n, l), textcoords="offset points", xytext=(12, 8), fontsize=14, color="#1A2433", weight="bold")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(ns); ax.set_xticklabels(["2,500", "50,000", "500,000"], fontsize=13)
ax.set_yticks([0.05, 0.1, 0.2, 0.5]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5"], fontsize=12)
ax.minorticks_off()
ax.set_xlabel("simulation length n (arrivals)", fontsize=13, color="#5A6B7B")
ax.set_ylabel("CI length of mean waiting time", fontsize=13, color="#5A6B7B")
ax.grid(True, color="#E3E9EF", lw=1); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color("#B8C4CF")
ax.legend(frameon=False, fontsize=12, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(A, "ci_length.png"), dpi=200); plt.close(fig)

f = make_formula(A, "ci_formula", [
    r"\bar{x} \;\pm\; t_{0.975,\,R-1}\,\frac{s}{\sqrt{R}}",
    r"\mathrm{CI\ length} = 2\,t_{0.975,\,R-1}\,\frac{s}{\sqrt{R}}"], fontsize=30)

d = Deck(footer_label="שיעור 3 · KPIs ואומדים")
d.figure_text(
    "רווח סמך לתוחלת זמן ההמתנה",
    f,
    [
        ("החזרות", "לכל אורך סימולציה יש 10 חזרות בלתי תלויות, וכל חזרה נותנת ממוצע המתנה אחד."),
        ("רמת ביטחון", "95% , עם ערך קריטי מהתפלגות סטודנט עם 9 דרגות חופש."),
        ("מה מודדים", "את אורך רווח הסמך: ככל שהוא קצר יותר, האומד מדויק יותר."),
    ],
    kicker="Confidence interval · R = 10 replications",
    img_h=2.0,
)
d.figure(
    "אורך רווח הסמך כפונקציה של אורך הסימולציה",
    os.path.join(A, "ci_length.png"),
    kicker="CI length vs. run length",
    note="פי 200 יותר הגעות מצמצמים את רווח הסמך פי 13 בערך, קרוב לכלל השורש: פי 14.",
    max_h=4.1,
)
d.save(os.path.join(os.environ["SP"], "nb3_ci_slides.pptx"))
