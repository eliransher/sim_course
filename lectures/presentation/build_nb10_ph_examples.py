# -*- coding: utf-8 -*-
"""Insert the 'PH examples with mean 1' slides into deck 10 after slide 10.

Deck 10 is hand-restyled (violet), so it is NOT rebuilt from build_nb10.py:
the new slides are built with deckbuilder in a temp deck, given the deck-10
palette, and transplanted into the existing file. Figures come from
assets/sim_scripts/draw_nb10_ph_examples.py.
Usage: python build_nb10_ph_examples.py <temp_dir> [output_path]  (default: overwrite the deck)
"""
import copy, io, os, re, sys
from pptx import Presentation
from pptx.oxml.ns import qn
from deckbuilder import Deck

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets", "nb10")
DECK = os.path.join(HERE, "notebook_10_PH_SCV.pptx")
INSERT_AFTER = 10
TMP = os.path.join(sys.argv[1], "nb10_ph_examples_tmp.pptx")
NS = '{http://schemas.openxmlformats.org/drawingml/2006/main}'

# ------------------------------------------------------------------ new slides
d = Deck(footer_label="שיעור 10 · שונות ותורים")

d.content(
    "דוגמאות: שלוש התפלגויות פאזה עם תוחלת 1",
    [
        ("שלוש דוגמאות", "ארלנג, היפר־מעריכית, והיפר־ארלנג עם שני שיאים."),
        ("לכל דוגמה", "וקטור ההתחלה, מטריצת הקצבים, והמבנה של השלבים."),
        ("ההשוואה", "הצפיפות, השונות, מקדם השונות בריבוע והצידוד."),
        ("למה תוחלת 1", "כשהתוחלת זהה, כל ההבדלים נובעים מהצורה בלבד — ומקדם השונות שווה לשונות."),
    ],
    kicker="PH(α, T) examples  ·  E[X] = 1",
)

d.figure_text(
    "דוגמה 1: ארלנג עם ארבעה שלבים",
    os.path.join(A, "ph_ex_erlang.png"),
    [
        ("המבנה", "ארבעה שלבים בטור, כל אחד בקצב 4. תמיד מתחילים בשלב הראשון."),
        ("המטריצה", "על האלכסון מינוס 4 — היציאה מכל שלב. מעליו 4 — המעבר לשלב הבא."),
        ("התוחלת", "ארבעה שלבים, כל אחד באורך ממוצע של רבע — ביחד 1."),
    ],
    kicker="Erlang  ·  4 phases, rate 4  ·  E[X] = 4 × 1/4 = 1",
    img_h=2.2,
)

d.figure_text(
    "דוגמה 2: היפר־מעריכית",
    os.path.join(A, "ph_ex_hyper.png"),
    [
        ("המבנה", "שני שלבים במקביל. מגרילים ענף אחד, ושוהים רק בשלב שלו."),
        ("וקטור ההתחלה", "ענף מהיר בהסתברות 0.8, וענף איטי בהסתברות 0.2."),
        ("המטריצה", "אלכסונית — אין מעברים בין שלבים. מכל שלב יוצאים ישר לסיום."),
    ],
    kicker="Hyper-exponential  ·  E[X] = 0.8 × 1/4 + 0.2 × 4 = 1",
    img_h=2.2,
)

d.figure_text(
    "דוגמה 3: היפר־ארלנג עם שני שיאים",
    os.path.join(A, "ph_ex_hyerl.png"),
    [
        ("המבנה", "שני ענפים במקביל, ובכל אחד כמה שלבים בטור — עירוב של שתי התפלגויות ארלנג."),
        ("המטריצה", "בנויה מבלוקים: בלוק ארלנג לכל ענף, ואפסים ביניהם — אין מעבר בין הענפים."),
        ("שני שיאים", "ענף קצר סביב 0.3, וענף ארוך ומרוכז סביב 1.5. הרבה שלבים — פחות שונות בענף."),
    ],
    kicker="Hyper-Erlang  ·  E[X] = ½ × 0.4 + ½ × 1.6 = 1",
    img_h=2.55,
)

d.figure_text(
    "מהו צידוד?",
    os.path.join(A, "skewness_shapes.png"),
    [
        ("ההגדרה", "המומנט השלישי של הסטייה מהתוחלת, מחולק בסטיית התקן בחזקת שלוש."),
        ("מה הוא מודד", "את חוסר הסימטריה: לאיזה צד נמתח הזנב, וכמה."),
        ("בזמני שירות", "הצידוד כמעט תמיד חיובי: רוב הזמנים קצרים, ומעטים ארוכים מאוד."),
    ],
    kicker="Skewness  ·  third standardized moment",
    img_h=2.55,
)

d.figure(
    "הצפיפויות של שלוש הדוגמאות",
    os.path.join(A, "ph_ex_pdfs.png"),
    kicker="Densities  ·  same mean, different shapes",
    note="אותה תוחלת — וצורות שונות לגמרי: ארלנג מרוכזת, היפר־מעריכית עם זנב ארוך, והיפר־ארלנג עם שני שיאים.",
    max_h=4.25,
)

d.table(
    "שונות, מקדם השונות וצידוד",
    ["Distribution", "Phases", "E[X]", "V[X]", "SCV", "Skewness"],
    [["Exponential", "1", "1", "1", "1", "2"],
     ["Erlang", "4", "1", "0.25", "0.25", "1.00"],
     ["Hyper-exp.", "2", "1", "5.50", "5.50", "4.60"],
     ["Hyper-Erlang", "16", "1", "0.49", "0.49", "0.56"]],
    kicker="SCV = V[X] / E[X]²   ·   skewness = E[(X − E[X])³] / σ³",
    note="ההיפר־מעריכית בולטת: שונות פי 5.5 מהמעריכית וצידוד גבוה — זנב ימני ארוך. להיפר־ארלנג שני שיאים, אבל צידוד נמוך.",
    col_w=1.95, row_h=0.62,
)
d.save(TMP)

# ------------------------------------------------------------------ deck-10 palette (violet, square boxes)
FILL = {'2E86AB': '7C3AED', '1F3A5F': '3B0764', 'E07A2F': '2DD4BF', 'F1F5F9': 'F5F3FF', 'E7EEF4': 'EDE9FE'}
TEXT = {'2E86AB': '6D28D9', 'E07A2F': '5EEAD4', '9FC0D8': 'DDD6FE', 'C7D3DE': 'E9E3FF', 'C7D8E6': 'E9E3FF'}
LINE = {'D3DDE6': 'C4B5FD'}


def restyle(prs):
    for s in prs.slides:
        for sh in s.shapes:
            for clr in sh._element.iter(NS + 'srgbClr'):
                t = re.sub(r'\{.*?\}', '', clr.getparent().getparent().tag)
                m = TEXT if t == 'rPr' else (LINE if t == 'ln' else FILL)
                v = clr.get('val').upper()
                if v in m:
                    clr.set('val', m[v])
            if sh.shape_type == 1 and sh.auto_shape_type == 5:
                g = sh._element.spPr.find(NS + 'prstGeom'); g.set('prst', 'rect')
                av = g.find(NS + 'avLst')
                if av is not None:
                    for c in list(av):
                        av.remove(c)


src = Presentation(TMP)
restyle(src)

# ------------------------------------------------------------------ transplant into deck 10
dst = Presentation(DECK)
layout = [l for l in dst.slide_layouts if l.name == src.slides[0].slide_layout.name][0]
new_ids = []
for ss in src.slides:
    ns = dst.slides.add_slide(layout)
    for ph in list(ns.placeholders):
        ph._element.getparent().remove(ph._element)
    tree = ns.shapes._spTree
    for el in ss.shapes._spTree.iterchildren():
        if el.tag.split('}')[1] in ('nvGrpSpPr', 'grpSpPr'):
            continue
        el2 = copy.deepcopy(el)
        for blip in el2.iter(NS + 'blip'):
            img = ss.part.related_part(blip.get(qn('r:embed')))
            _, rid = ns.part.get_or_add_image_part(io.BytesIO(img.blob))
            blip.set(qn('r:embed'), rid)
        tree.append(el2)
    new_ids.append(dst.slides._sldIdLst[-1])
lst = dst.slides._sldIdLst
for k, sid in enumerate(new_ids):
    lst.remove(sid)
    lst.insert(INSERT_AFTER + k, sid)

# footers: content slide k shows k-1
for k, s in enumerate(dst.slides, 1):
    for sh in s.shapes:
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if r.text.startswith('שיעור 10') and re.search(r'·  \d+$', r.text):
                        r.text = re.sub(r'\d+$', str(k - 1), r.text)
OUT = sys.argv[2] if len(sys.argv) > 2 else DECK
dst.save(OUT)
print("deck 10 now has", len(dst.slides), "slides")
