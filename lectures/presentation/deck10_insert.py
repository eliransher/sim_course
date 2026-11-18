# -*- coding: utf-8 -*-
"""Helpers for inserting deckbuilder slides into the hand-restyled deck 10
(violet palette, square boxes) without rebuilding it."""
import copy, io, re
from pptx.oxml.ns import qn

NS = '{http://schemas.openxmlformats.org/drawingml/2006/main}'

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




def transplant(src, dst, insert_after):
    """Copy every slide of `src` into `dst`, placed after slide number `insert_after`."""
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
        lst.insert(insert_after + k, sid)


def renumber_footers(dst, prefix):
    """Footer of slide k shows k-1 (the title slide has no footer)."""
    for k, s in enumerate(dst.slides, 1):
        for sh in s.shapes:
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        if r.text.startswith(prefix) and re.search(r'·  \d+$', r.text):
                            r.text = re.sub(r'\d+$', str(k - 1), r.text)
