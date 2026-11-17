# -*- coding: utf-8 -*-
"""Teal style used by deck 11 and its sister deck 11b (MAP fitting).

Applied on top of the default deckbuilder palette: deep-teal accents, amber
highlights, borderless content boxes and a thicker rule under each title.
Usage as a script: python restyle_teal.py <deck.pptx> [<deck.pptx> ...]
"""
import re, sys
from pptx import Presentation
from pptx.util import Pt

NS = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
FILL = {'2E86AB': '0D9488', '1F3A5F': '134E4A', 'E07A2F': 'F59E0B', 'F1F5F9': 'F0FDFA', 'E7EEF4': 'CCFBF1'}
TEXT = {'2E86AB': '0F766E', 'E07A2F': 'FBBF24', '9FC0D8': '99F6E4', 'C7D3DE': 'CCFBF1', 'C7D8E6': 'CCFBF1'}
LINE = {'D3DDE6': '99F6E4'}
RULE_H = Pt(4)


def restyle(prs):
    for s in prs.slides:
        for sh in s.shapes:
            el = sh._element
            for clr in el.iter(NS + 'srgbClr'):
                t = re.sub(r'\{.*?\}', '', clr.getparent().getparent().tag)
                m = TEXT if t == 'rPr' else (LINE if t == 'ln' else FILL)
                v = clr.get('val').upper()
                if v in m:
                    clr.set('val', m[v])
            if sh.shape_type != 1:
                continue
            # borderless rounded content boxes
            if sh.auto_shape_type == 5:
                sh.line.fill.background()
            # thicker rule under the title (the thin full-width accent bar at y = 1.72in)
            if sh.auto_shape_type == 1 and sh.height < Pt(3) and sh.width > Pt(600) and abs(sh.top - 1572768) < 20000:
                sh.top = sh.top - (RULE_H - sh.height) // 2
                sh.height = RULE_H


if __name__ == "__main__":
    for path in sys.argv[1:]:
        p = Presentation(path)
        restyle(p)
        p.save(path)
        print("restyled", path)
