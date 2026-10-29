#!/usr/bin/env python
"""Add slide transitions and entrance animations to the deck written by build_deck.js.

    python animate.py AML-GraphMiner_soutenance.pptx

Every object whose name starts with ``a<k>`` appears when the slide opens, group after group
(k = 1, 2, ...): fade by default, a wipe from the left for names ``a<k>w_...``. Each slide gets a
fade transition. pptxgenjs cannot write animations, so they are added to the slide XML here.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

STEP_MS = 350  # delay between two groups
DUR_MS = 500  # length of one effect
NAME = re.compile(r'<p:cNvPr id="(\d+)" name="a(\d+)(w?)_[^"]*"')


def kind_at(xml: str, pos: int) -> str:
    """Type of the shape that owns the cNvPr at ``pos`` (sp, pic or graphicFrame)."""
    starts = {k: xml.rfind(f"<p:{k}>", 0, pos) for k in ("sp", "pic", "graphicFrame", "cxnSp")}
    return max(starts, key=lambda k: starts[k])


def effect(ctn: int, spid: str, first: bool, wipe: bool) -> tuple[str, int]:
    preset, sub, filt = ("22", "8", "wipe(left)") if wipe else ("10", "0", "fade")
    node = "afterEffect" if first else "withEffect"
    xml = (
        f'<p:par><p:cTn id="{ctn}" presetID="{preset}" presetClass="entr" presetSubtype="{sub}"'
        f' fill="hold" grpId="0" nodeType="{node}"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
        f'<p:childTnLst><p:set><p:cBhvr><p:cTn id="{ctn + 1}" dur="1" fill="hold"><p:stCondLst>'
        f'<p:cond delay="0"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>'
        f"<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr>"
        f'<p:to><p:strVal val="visible"/></p:to></p:set><p:animEffect transition="in"'
        f' filter="{filt}"><p:cBhvr><p:cTn id="{ctn + 2}" dur="{DUR_MS}"/><p:tgtEl>'
        f'<p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:animEffect></p:childTnLst></p:cTn></p:par>'
    )
    return xml, ctn + 3


def timing(xml: str) -> str:
    groups: dict[int, list[tuple[str, bool, str]]] = {}
    for m in NAME.finditer(xml):
        groups.setdefault(int(m.group(2)), []).append(
            (m.group(1), m.group(3) == "w", kind_at(xml, m.start()))
        )
    if not groups:
        return ""
    ctn, pars, builds = 4, [], []
    for i, k in enumerate(sorted(groups)):
        group_id = ctn
        ctn += 1
        effects = []
        for j, (spid, wipe, kind) in enumerate(groups[k]):
            e, ctn = effect(ctn, spid, j == 0, wipe)
            effects.append(e)
            if kind == "sp":
                builds.append(f'<p:bldP spid="{spid}" grpId="0" animBg="1"/>')
            elif kind == "graphicFrame":
                builds.append(f'<p:bldGraphic spid="{spid}" grpId="0"><p:bldAsOne/></p:bldGraphic>')
        pars.append(
            f'<p:par><p:cTn id="{group_id}" fill="hold"><p:stCondLst><p:cond delay="{i * STEP_MS}"/>'
            f"</p:stCondLst><p:childTnLst>{''.join(effects)}</p:childTnLst></p:cTn></p:par>"
        )
    return (
        '<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">'
        '<p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite"'
        ' nodeType="mainSeq"><p:childTnLst><p:par><p:cTn id="3" fill="hold"><p:stCondLst>'
        '<p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond>'
        f"</p:stCondLst><p:childTnLst>{''.join(pars)}</p:childTnLst></p:cTn></p:par></p:childTnLst>"
        '</p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>'
        '</p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl>'
        "</p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst>"
        f"<p:bldLst>{''.join(builds)}</p:bldLst></p:timing>"
        if builds
        else '<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">'
        '<p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite"'
        ' nodeType="mainSeq"><p:childTnLst><p:par><p:cTn id="3" fill="hold"><p:stCondLst>'
        '<p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond>'
        f"</p:stCondLst><p:childTnLst>{''.join(pars)}</p:childTnLst></p:cTn></p:par></p:childTnLst>"
        '</p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>'
        '</p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl>'
        "</p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>"
    )


def animate(path: Path) -> int:
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    n = 0
    for slide in sorted((tmp / "ppt" / "slides").glob("slide*.xml")):
        xml = slide.read_text(encoding="utf-8")
        if "<p:timing>" in xml:
            continue
        extra = '<p:transition spd="med"><p:fade/></p:transition>' + timing(xml)
        anchor = "</p:clrMapOvr>"
        if anchor in xml:
            xml = xml.replace(anchor, anchor + extra, 1)
        else:
            xml = xml.replace("</p:cSld>", "</p:cSld>" + extra, 1)
        slide.write_text(xml, encoding="utf-8")
        n += 1
    out = path.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        # [Content_Types].xml first, as Office writes it
        files = sorted(p for p in tmp.rglob("*") if p.is_file())
        files.sort(key=lambda p: p.name != "[Content_Types].xml")
        for f in files:
            z.write(f, f.relative_to(tmp).as_posix())
    shutil.move(out, path)
    shutil.rmtree(tmp)
    return n


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "AML-GraphMiner_soutenance.pptx")
    print(f"animated {animate(target)} slides in {target}")
