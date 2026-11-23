# -*- coding: utf-8 -*-
"""Insert one slide into deck 10 after the 'Notation' slide (17): a worked
numerical example of the PH density at a single point and of the mean,
for the size-2 hyper-exponential alpha = (0.8, 0.2), S = diag(-4, -0.25).
Figure: assets/sim_scripts/draw_nb10_ph_density_example.py (ph_density_example.png).
One-shot: rerunning inserts the slide again.
Usage: python build_nb10_ph_density.py <temp_dir> [output_path]  (default: overwrite the deck)
"""
import os, sys
from pptx import Presentation
from deckbuilder import Deck
from deck10_insert import restyle, transplant, renumber_footers

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets", "nb10")
DECK = os.path.join(HERE, "notebook_10_PH_SCV.pptx")
INSERT_AFTER = 17
TMP = os.path.join(sys.argv[1], "nb10_ph_density_tmp.pptx")

d = Deck(footer_label="שיעור 10 · שונות ותורים")
d.figure_text(
    "דוגמה מספרית: הצפיפות בנקודה אחת, והתוחלת",
    os.path.join(A, "ph_density_example.png"),
    [
        ("הנתונים", "ההיפר־מעריכית מדוגמה 2: שני שלבים בקצבים 4 ו־0.25. וקטור הסיום הוא מינוס סכום כל שורה במטריצה."),
        ("הצפיפות בנקודה 1", "המטריצה אלכסונית, ולכן האקספוננט שלה מתקבל איבר־איבר על האלכסון. התוצאה: עירוב של שתי צפיפויות מעריכיות."),
        ("התוחלת", "ההופכית של מטריצה אלכסונית הופכת כל איבר. מקבלים 0.8 כפול 0.25, ועוד 0.2 כפול 4 — בדיוק 1."),
    ],
    kicker="Worked example  ·  hyper-exponential, α = (0.8, 0.2), x = 1",
    img_h=2.75,
)
d.save(TMP)

src = Presentation(TMP)
restyle(src)
dst = Presentation(DECK)
assert "Notation" in "".join(sh.text_frame.text for sh in dst.slides[INSERT_AFTER - 1].shapes if sh.has_text_frame)
transplant(src, dst, INSERT_AFTER)
renumber_footers(dst, "שיעור 10")
OUT = sys.argv[2] if len(sys.argv) > 2 else DECK
dst.save(OUT)
print("deck 10 now has", len(dst.slides), "slides")
