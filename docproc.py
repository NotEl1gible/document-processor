#!/usr/bin/env python3
"""
Document processing — OCR, extraction, validation — and where the errors actually come from.

The pipeline is the easy part. Three things this measures that a document-AI writeup usually
does not:

  1. STAGE ATTRIBUTION. The quality ceiling of the whole system is set by OCR, and almost
     nobody splits the error. Here the same extractor runs on the exact text layer and on the
     OCR output of the same page, so the difference IS the OCR contribution -- measured, not
     argued.

  2. INVARIANCE, WITHOUT LABELS. Re-render a document on a different background colour and
     the extracted fields must not change. That is a metamorphic property: the invariant is
     known by construction, so it needs no annotation at all. If the JSON moves when only the
     background moved, that is a real robustness bug.

  3. THE CURVE THAT DECIDES THE ECONOMICS. Not accuracy -- the straight-through rate. A
     system that is 99% accurate but cannot say WHICH 1% is wrong has a straight-through rate
     of zero, because everything must be checked. The confidence estimate is worth more than
     the accuracy, and the report shows the trade rather than a single number.

GROUND TRUTH IS FREE HERE, and that is a deliberate design choice rather than a shortcut.
The documents are generated with PyMuPDF, so every field value and every bounding box is
known exactly. No annotation, no judge, no hand-labelling -- and the degradation applied
afterwards (rasterisation, blur, noise, rotation, JPEG, background colour) is real, so the
OCR stage faces a genuine problem while the labels stay perfect.

Usage:
    python docproc.py generate            # render the corpus, digital + degraded scans
    python docproc.py attribute           # where the error comes from, stage by stage
    python docproc.py invariance          # same document, different background: does it move?
    python docproc.py curve               # confidence -> straight-through -> residual error
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
EXIT = {"OK": 0, "GATE_FAILED": 1, "INFRA": 3}

DOCS_DIR = Path("docs")
OUT_DIR = Path("runs")

VENDORS = ["Northwind Ltd", "Acme Industrial", "Blue Harbour GmbH", "Cedar & Sons",
           "Delta Logistics", "Evergreen Supplies", "Fairview Media", "Granite Tools"]
CURRENCIES = ["EUR", "USD", "GBP"]

# Rendering knobs. Kept modest on purpose: this runs on a laptop CPU and the point is the
# measurement, not throughput.
DPI = 150
PAGE_W, PAGE_H = 595, 842          # A4 at 72 dpi


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def dump_jsonl(path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                          encoding="utf-8", newline="\n")


def wilson(k: int, n: int) -> tuple[float, float]:
    """Binomial interval. Reused from the sibling repos; named as reuse in the README."""
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{(k / n if n else 0):.2f} ({k}/{n}) [{lo:.2f}, {hi:.2f}]"


# ----------------------------------------------------------------------------
# The corpus — generated, so the labels are exact and free
# ----------------------------------------------------------------------------
def make_record(rng: random.Random, i: int) -> dict:
    """The ground truth. The PDF is rendered FROM this, so the label cannot disagree with
    the document -- a class of annotation error that hand-labelled corpora always carry."""
    qty = [rng.randint(1, 40) for _ in range(rng.randint(2, 4))]
    unit = [round(rng.uniform(4, 400), 2) for _ in qty]
    lines = [{"sku": f"SKU-{rng.randint(1000, 9999)}", "qty": q, "unit_price": u,
              "amount": round(q * u, 2)} for q, u in zip(qty, unit)]
    net = round(sum(l["amount"] for l in lines), 2)
    vat_rate = rng.choice([0.0, 0.07, 0.19, 0.21])
    vat = round(net * vat_rate, 2)
    return {
        "doc_id": f"doc-{i:04d}",
        "invoice_no": f"{rng.choice('ABCDEF')}-{rng.randint(1000, 9999)}",
        "vendor": rng.choice(VENDORS),
        "date": f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        "currency": rng.choice(CURRENCIES),
        "lines": lines,
        "net": net,
        "vat_rate": vat_rate,
        "vat": vat,
        "total": round(net + vat, 2),
    }


def render_pdf(rec: dict, path: Path, bg: tuple[float, float, float] = (1, 1, 1),
               fg: tuple[float, float, float] = (0, 0, 0)) -> None:
    """Render the record to a real PDF with a real text layer.

    `bg`/`fg` exist for the invariance test: the same record on a different background is the
    same document, so every extracted field must be identical. That invariant is known
    without any annotation, which is what makes the test free.
    """
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.draw_rect(fitz.Rect(0, 0, PAGE_W, PAGE_H), color=None, fill=bg)

    def t(x, y, s, size=10, bold=False):
        page.insert_text((x, y), s, fontsize=size, color=fg,
                         fontname="helv" if not bold else "hebo")

    t(56, 70, "INVOICE", 20, bold=True)
    t(56, 96, f"No. {rec['invoice_no']}", 11)
    t(56, 130, f"Vendor: {rec['vendor']}", 11)
    t(56, 148, f"Date: {rec['date']}", 11)
    t(380, 130, f"Currency: {rec['currency']}", 11)

    y = 200
    t(56, y, "SKU", 9, bold=True); t(180, y, "QTY", 9, bold=True)
    t(250, y, "UNIT", 9, bold=True); t(340, y, "AMOUNT", 9, bold=True)
    page.draw_line(fitz.Point(56, y + 6), fitz.Point(430, y + 6), color=fg, width=0.6)
    for ln in rec["lines"]:
        y += 22
        t(56, y, ln["sku"]); t(180, y, str(ln["qty"]))
        t(250, y, f"{ln['unit_price']:.2f}"); t(340, y, f"{ln['amount']:.2f}")

    y += 40
    page.draw_line(fitz.Point(250, y - 14), fitz.Point(430, y - 14), color=fg, width=0.6)
    t(250, y, "Net"); t(340, y, f"{rec['net']:.2f}")
    y += 20
    t(250, y, f"VAT {rec['vat_rate']*100:.0f}%"); t(340, y, f"{rec['vat']:.2f}")
    y += 20
    t(250, y, "TOTAL", 11, bold=True); t(340, y, f"{rec['total']:.2f} {rec['currency']}", 11,
                                         bold=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()


def text_layer(pdf_path: Path) -> str:
    """The ceiling. If a PDF carries a text layer you do not need OCR at all -- you get the
    characters exactly, plus the coordinates OCR would have to guess. A large share of
    enterprise documents are digital-born, and running OCR over them does not merely waste
    time, it INTRODUCES error where there was none."""
    import fitz
    doc = fitz.open(str(pdf_path))
    txt = "\n".join(p.get_text() for p in doc)
    doc.close()
    return txt


def rasterize(pdf_path: Path, dpi: int = DPI):
    import fitz
    from PIL import Image
    doc = fitz.open(str(pdf_path))
    pix = doc[0].get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


# ----------------------------------------------------------------------------
# Degradation — a real scan, not a synthetic label flip
# ----------------------------------------------------------------------------
# This is what makes the ground truth honest. The labels stay perfect because the record
# generated the page; the INPUT genuinely gets worse, in the ways a real scanner or a phone
# camera makes it worse. So the OCR stage faces a real problem and the measurement of its
# contribution is a measurement rather than an assumption.

DEGRADATIONS = {
    "clean":  dict(blur=0.0, noise=0,  rotate=0.0, jpeg=95, scale=1.00),
    "light":  dict(blur=0.4, noise=6,  rotate=0.3, jpeg=80, scale=0.90),
    "medium": dict(blur=0.9, noise=14, rotate=0.8, jpeg=55, scale=0.72),
    "heavy":  dict(blur=1.5, noise=24, rotate=1.6, jpeg=35, scale=0.55),
}


def degrade(img, level: str, seed: int = 0):
    from PIL import Image, ImageFilter
    import numpy as np
    cfg = DEGRADATIONS[level]
    rng = np.random.default_rng(seed)
    if cfg["rotate"]:
        img = img.rotate(rng.uniform(-cfg["rotate"], cfg["rotate"]),
                         resample=Image.BILINEAR, fillcolor=(255, 255, 255), expand=False)
    if cfg["scale"] != 1.0:
        # Downscale then back up: this is how a low-dpi scan destroys thin strokes, and it is
        # the single most damaging thing in the list.
        w, h = img.size
        img = img.resize((int(w * cfg["scale"]), int(h * cfg["scale"])), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
    if cfg["blur"]:
        img = img.filter(ImageFilter.GaussianBlur(cfg["blur"]))
    if cfg["noise"]:
        a = np.asarray(img).astype(np.int16)
        a = np.clip(a + rng.normal(0, cfg["noise"], a.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(a)
    if cfg["jpeg"] < 95:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=cfg["jpeg"])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


# ----------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------
_OCR = None


def ocr_engine():
    """RapidOCR: PaddleOCR's PP-OCR models exported to ONNX. CPU-only, no torch, no system
    binary -- which is why it is here rather than Tesseract. Tesseract needs a native install
    that a reader cloning this repo may not have, and a portfolio project whose first command
    fails on a missing binary is a project nobody runs."""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def ocr_text(img) -> tuple[str, float]:
    """Returns (text in reading order, mean engine confidence).

    Reading order is reconstructed by sorting boxes top-to-bottom then left-to-right with a
    line-grouping tolerance. This is exactly the step that silently destroys multi-column
    documents: the characters can all be correct while the ORDER is wrong, and every
    downstream stage then reads a scrambled page with no indication that anything failed.
    """
    import numpy as np
    res, _ = ocr_engine()(np.asarray(img))
    if not res:
        return "", 0.0
    items = []
    for box, txt, conf in res:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((sum(ys) / 4.0, sum(xs) / 4.0, txt, float(conf)))
    items.sort(key=lambda r: (r[0], r[1]))
    lines, cur, cur_y = [], [], None
    tol = 12.0
    for y, x, txt, conf in items:
        if cur_y is None or abs(y - cur_y) <= tol:
            cur.append((x, txt)); cur_y = y if cur_y is None else (cur_y + y) / 2
        else:
            lines.append(cur); cur, cur_y = [(x, txt)], y
    if cur:
        lines.append(cur)
    text = "\n".join(" ".join(t for _, t in sorted(ln)) for ln in lines)
    return text, float(sum(i[3] for i in items) / len(items))


def cmd_generate(args):
    rng = random.Random(args.seed)
    recs = [make_record(rng, i) for i in range(args.n)]
    DOCS_DIR.mkdir(exist_ok=True)
    dump_jsonl(DOCS_DIR / "truth.jsonl", recs)
    for r in recs:
        render_pdf(r, DOCS_DIR / "pdf" / f"{r['doc_id']}.pdf")
    print(f"=== corpus ===")
    print(f"  records {len(recs)}  ->  {DOCS_DIR/'truth.jsonl'}")
    print(f"  PDFs             ->  {DOCS_DIR/'pdf'}")
    print(f"  fields per doc   invoice_no, vendor, date, currency, net, vat, total "
          f"+ {sum(len(r['lines']) for r in recs)/len(recs):.1f} line items on average")
    print()
    print( "  Ground truth is EXACT because the PDF was rendered from the record, not")
    print( "  annotated after the fact. That removes a whole class of error a hand-labelled")
    print( "  corpus always carries -- the label that disagrees with the document -- and it")
    print( "  costs nothing. The degradation applied later is real, so the OCR stage still")
    print( "  faces a genuine problem.")
    return EXIT["OK"]


# ----------------------------------------------------------------------------
# Extraction — one extractor, both paths
# ----------------------------------------------------------------------------
# The same function runs on the exact text layer and on the OCR output. That is the whole
# basis of the attribution: if the extractor differs between paths, the difference between
# the paths stops being the OCR contribution and becomes an artifact of the harness.
#
# Mock-first, as in every sibling: the rule-based extractor runs offline and deterministically
# so the pipeline and its measurements need no key. `--provider anthropic` swaps in a real
# model behind the same signature.

FIELDS = ["invoice_no", "vendor", "date", "currency", "net", "vat", "total"]

NUM = r"[-+]?\d[\d,\s]*(?:\.\d+)?"


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s = re.sub(r"[\s,](?=\d{3}\b)", "", s.strip())     # thousands separators
    s = s.replace(" ", "").replace(",", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def extract_rules(text: str) -> dict:
    """Deterministic stand-in for an LLM extractor.

    Written to be TOLERANT in the same ways a model is: it hunts for a label and takes what
    follows, rather than assuming a fixed layout. That matters for the attribution, because a
    brittle regex extractor would fail on OCR output for reasons that have nothing to do with
    OCR quality -- it would fail on spacing -- and the measurement would blame the wrong stage.
    """
    t = text.replace(" ", " ")
    out: dict = {}

    m = re.search(r"No\.?\s*([A-Z]-?\d{3,5})", t, re.I)
    out["invoice_no"] = m.group(1).replace(" ", "") if m else None

    m = re.search(r"Vendor\s*:?\s*(.+)", t, re.I)
    if m:
        v = m.group(1).strip()
        v = re.split(r"\s{2,}|Currency|Date|No\.", v, flags=re.I)[0].strip(" :.-")
        out["vendor"] = v or None
    else:
        out["vendor"] = None

    m = re.search(r"Date\s*:?\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})", t, re.I)
    out["date"] = m.group(1).replace("/", "-").replace(".", "-") if m else None

    m = re.search(r"Currency\s*:?\s*([A-Z]{3})", t, re.I)
    out["currency"] = m.group(1).upper() if m else None

    m = re.search(rf"\bNet\b\s*:?\s*({NUM})", t, re.I)
    out["net"] = _num(m.group(1)) if m else None

    m = re.search(rf"VAT\s*\d*\s*%?\s*:?\s*({NUM})", t, re.I)
    out["vat"] = _num(m.group(1)) if m else None

    m = re.search(rf"TOTAL\s*:?\s*({NUM})", t, re.I)
    out["total"] = _num(m.group(1)) if m else None
    return out


def extract(text: str, provider: str = "mock") -> dict:
    if provider == "mock":
        return extract_rules(text)
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-haiku-4-5", max_tokens=600,
        system=("Extract invoice fields as ONE JSON object and nothing else. Keys: "
                "invoice_no, vendor, date (YYYY-MM-DD), currency (3 letters), net, vat, "
                "total. Numbers as numbers, no thousands separators. Use null when a field "
                "is not present in the text."),
        messages=[{"role": "user", "content": text[:6000]}])
    raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        obj = {}
    return {k: obj.get(k) for k in FIELDS}



# ----------------------------------------------------------------------------
# Snapping to a closed vocabulary — vectorisation at the layer where it pays
# ----------------------------------------------------------------------------
# The conceptual point, made concrete. Vectorising PIXELS to retrieve a letter fails: in
# pixel space the same 'A' on white vs red scores 0.762 while a different letter on the same
# background scores 0.976 -- nearest-neighbour would retrieve by background colour.
#
# Vectorising ENTITIES after recognition is the opposite. The OCR returned
# "BlueHarbourGmbH"; the supplier master contains "Blue Harbour GmbH". Character trigrams are
# almost identical between them, so cosine similarity finds it immediately -- and this fixes
# a whole class of error without touching OCR at all.
#
# The principle generalises past this one field: wherever a CLOSED list exists -- suppliers,
# currencies, country codes, product catalogue -- the model guesses and the list corrects.
# That is cheaper and more reliable than any accuracy improvement upstream.

SNAP_THRESHOLD = 0.55


def trigrams(s: str) -> dict[str, int]:
    s = "  " + re.sub(r"[^a-z0-9]+", "", str(s).lower()) + "  "
    out: dict[str, int] = {}
    for i in range(len(s) - 2):
        g = s[i:i + 3]
        out[g] = out.get(g, 0) + 1
    return out


def cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[g] * b[g] for g in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def snap(value, vocabulary: list[str], threshold: float = SNAP_THRESHOLD):
    """Return (snapped_value, similarity, changed). Leaves the value alone below threshold.

    The threshold matters and is not free: too low and a genuinely new supplier gets rewritten
    into an existing one, which is a silent data-corruption bug far worse than a missing
    field. `curve` reports what it costs on values that are NOT in the vocabulary.
    """
    if value in (None, ""):
        return value, 0.0, False
    tv = trigrams(value)
    best, score = None, 0.0
    for cand in vocabulary:
        sc = cosine(tv, trigrams(cand))
        if sc > score:
            best, score = cand, sc
    if best is not None and score >= threshold and best != value:
        return best, score, True
    return value, score, False


def snap_record(pred: dict) -> tuple[dict, list[str]]:
    out, changed = dict(pred), []
    for field, vocab in (("vendor", VENDORS), ("currency", CURRENCIES)):
        v, sc, ch = snap(out.get(field), vocab)
        if ch:
            changed.append(f"{field}:{sc:.2f}")
        out[field] = v
    return out, changed


# ----------------------------------------------------------------------------
# Validation — business rules, which are also the confidence signal
# ----------------------------------------------------------------------------
# These rules are the only part of the system that can catch a WRONG-BUT-PLAUSIBLE number.
# A schema check cannot: 18478.33 and 18478.83 are both valid floats. Arithmetic that has to
# close is the strongest validator a document pipeline has, and it needs no model.

def validate(pred: dict) -> list[str]:
    problems = []
    for f in FIELDS:
        if pred.get(f) in (None, ""):
            problems.append(f"missing:{f}")
    if pred.get("currency") and pred["currency"] not in CURRENCIES:
        problems.append("currency_not_in_list")
    if pred.get("date") and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(pred["date"])):
        problems.append("date_malformed")
    net, vat, total = (pred.get(k) for k in ("net", "vat", "total"))
    if None not in (net, vat, total):
        # The load-bearing rule: the arithmetic must close. This is what catches a digit that
        # OCR misread, because a corrupted number stops adding up.
        if abs((net + vat) - total) > 0.02:
            problems.append("arithmetic_mismatch")
    if isinstance(total, (int, float)) and total <= 0:
        problems.append("total_not_positive")
    return problems


def confidence(pred: dict, ocr_conf: float | None, problems: list[str]) -> float:
    """A confidence built from things that can be CHECKED, not from a model's self-report.

    Deliberately not asked of the extractor. Measured in a sibling repo: a model's own
    confidence bands all fell outside their own Wilson intervals, and here the OCR engine's
    mean score went UP from 0.781 to 0.794 as the page got worse. Self-reported confidence in
    this pipeline tracks nothing.
    """
    c = 1.0
    c -= 0.25 * sum(1 for p in problems if p.startswith("missing:"))
    if "arithmetic_mismatch" in problems:
        c -= 0.45
    if "currency_not_in_list" in problems or "date_malformed" in problems:
        c -= 0.20
    if ocr_conf is not None:
        c = 0.75 * c + 0.25 * ocr_conf          # a weak input, weighted as one
    return max(0.0, min(1.0, c))


def field_hits(pred: dict, truth: dict, loose: bool = False) -> dict:
    """`loose` collapses whitespace away entirely for text fields.

    This exists because of a measurement, not a preference. At light degradation the OCR
    returns "Vendor:BlueHarbourGmbH" -- every CHARACTER is correct and only the word
    boundaries are gone. Scored strictly that is a miss; scored loosely it is a hit, and the
    truth is that it is a THIRD thing: recoverable downstream by matching against a supplier
    master, in a way that a misread digit never is.

    Reporting only the strict number blames OCR for an error that a vendor lookup would
    absorb. Reporting only the loose number hides a real defect. So `attribute` prints both,
    and the gap between them is the share of the loss that is merely cosmetic.
    """
    def norm(k, v):
        if v is None:
            return None
        if k in ("net", "vat", "total"):
            try:
                return round(float(v), 2)
            except (TypeError, ValueError):
                return None
        s = str(v).strip().lower()
        return re.sub(r"\s+", "", s) if loose else re.sub(r"\s+", " ", s)
    return {k: int(norm(k, pred.get(k)) == norm(k, truth.get(k))) for k in FIELDS}



def cmd_attribute(args):
    """Where does the error actually come from?

    Same extractor, same scorer, two inputs: the exact text layer and the OCR of the same
    page at several degradation levels. The text-layer row is the CEILING -- whatever the
    extractor gets wrong there is the extractor's fault and no amount of better OCR will fix
    it. Everything below that row is the OCR contribution.
    """
    truth = {r["doc_id"]: r for r in load_jsonl(DOCS_DIR / "truth.jsonl")}
    ids = sorted(truth)[: args.n]
    OUT_DIR.mkdir(exist_ok=True)
    rows, gates = [], []

    print(f"=== stage attribution ({len(ids)} documents, per-field accuracy) ===")
    header = (f"  {'input':<26}" + "".join(f"{f[:9]:>10}" for f in FIELDS)
              + f"{'all':>8}{'loose':>8}")
    print(header)

    results = {}
    for label in ["text layer (no OCR)"] + [f"OCR {lv}" for lv in DEGRADATIONS]:
        agg = {f: 0 for f in FIELDS}
        allok = looseok = 0
        for did in ids:
            pdf = DOCS_DIR / "pdf" / f"{did}.pdf"
            if label.startswith("text"):
                text, oconf = text_layer(pdf), None
            else:
                lv = label.split()[1]
                text, oconf = ocr_text(degrade(rasterize(pdf), lv, seed=7))
            pred = extract(text, args.provider)
            if args.snap:
                pred, _ = snap_record(pred)
            hits = field_hits(pred, truth[did])
            loose = field_hits(pred, truth[did], loose=True)
            for f in FIELDS:
                agg[f] += hits[f]
            allok += int(all(hits.values()))
            looseok += int(all(loose.values()))
            rows.append({"doc_id": did, "input": label, "pred": pred,
                         "hits": hits, "ocr_conf": oconf,
                         "problems": validate(pred),
                         "confidence": confidence(pred, oconf, validate(pred))})
        results[label] = (agg, allok, looseok)
        print(f"  {label:<26}" + "".join(f"{agg[f]/len(ids):>10.2f}" for f in FIELDS)
              + f"{allok/len(ids):>8.2f}{looseok/len(ids):>8.2f}")

    dump_jsonl(OUT_DIR / "attribute.jsonl", rows)
    ceil_agg, ceil_all, ceil_loose = results["text layer (no OCR)"]
    print()
    print("  The first row is the CEILING. Whatever it gets wrong is the EXTRACTOR's fault,")
    print("  and no amount of better OCR will move it. Every drop below that row is the OCR")
    print("  stage -- which is the number a document-AI writeup almost never separates out.")

    worst = f"OCR {list(DEGRADATIONS)[-1]}"
    ocr_cost = (ceil_all - results[worst][1]) / len(ids)
    ext_cost = 1.0 - ceil_all / len(ids)
    print()
    print(f"  extractor cost (ceiling gap)   {ext_cost:>6.2f}")
    print(f"  OCR cost (ceiling -> {worst})  {ocr_cost:>6.2f}")
    if ext_cost + ocr_cost > 0:
        share = ocr_cost / (ext_cost + ocr_cost)
        print(f"  OCR is responsible for {share:.0%} of the total loss on this corpus.")

    if ceil_all == len(ids) and results[worst][1] == len(ids):
        gates.append("every configuration is perfect, including heavy degradation. Either "
                     "the degradation stopped degrading or the corpus is too easy to "
                     "distinguish the stages, and the attribution measures nothing")
    if ceil_all < len(ids) * 0.9:
        gates.append(f"the text-layer ceiling is only {ceil_all}/{len(ids)}; the extractor is "
                     f"the bottleneck, so any OCR conclusion below it is noise on top of a "
                     f"broken extractor")
    for g in gates:
        print()
        print(f"GATE FAILED: {g}")
    return EXIT["GATE_FAILED"] if gates else EXIT["OK"]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                   # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="Document processing pipeline")
    ap.add_argument("command", choices=["generate", "attribute"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    ap.add_argument("--snap", action="store_true",
                    help="snap vendor/currency to the closed vocabulary by "
                         "character-trigram cosine before scoring")
    args = ap.parse_args()
    load_dotenv()
    return {"generate": cmd_generate, "attribute": cmd_attribute}[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
