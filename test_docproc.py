"""Offline and hermetic. No OCR model is loaded by any test here, and that is deliberate.

The parts of a document pipeline that fail SILENTLY are the scoring, the validation and the
snapping — not the recogniser. So the suite runs in seconds on CPU with no ONNX download, and
still guards every claim the README makes. The two OCR-shaped tests use recorded text rather
than a live engine, so they are fast and they cannot flake on a model version.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import docproc as D


# ---------------------------------------------------------------- ground truth
def test_the_pdf_is_rendered_from_the_record(tmp_path):
    """Ground truth cannot disagree with the document, because the document is generated FROM
    the truth. This is the property that makes the whole corpus free of annotation error, so
    it is asserted rather than assumed."""
    import random
    rec = D.make_record(random.Random(0), 0)
    pdf = tmp_path / "x.pdf"
    D.render_pdf(rec, pdf)
    text = D.text_layer(pdf)
    for field in ("invoice_no", "vendor", "date", "currency"):
        assert str(rec[field]) in text, f"{field} is not present in the rendered page"
    assert f"{rec['total']:.2f}" in text


def test_arithmetic_of_the_generated_record_closes():
    """If the generator ever emits a record where net + vat != total, the single most
    load-bearing validation rule becomes a false-alarm generator."""
    import random
    rng = random.Random(1)
    for i in range(50):
        r = D.make_record(rng, i)
        assert abs((r["net"] + r["vat"]) - r["total"]) < 0.02
        assert abs(sum(l["amount"] for l in r["lines"]) - r["net"]) < 0.02


# ---------------------------------------------------------------- extraction
CLEAN = """INVOICE
No. E-3281
Vendor: Delta Logistics
Date: 2026-03-25
Currency: EUR
SKU QTY UNIT AMOUNT
SKU-1234 10 25.00 250.00
Net 250.00
VAT 19% 47.50
TOTAL 297.50 EUR"""

# What RapidOCR actually returned at light degradation: the characters are all correct and
# the SPACES are gone, and "Currency" has been merged onto the vendor line because the two
# sit at the same height in different columns.
OCR_LIGHT = """INVOICE
No.E-3281
Vendor:DeltaLogistics Currency:EUR
Date:2026-03-25
SKU QTY UNIT AMOUNT
SKU-1234 10 25.00 250.00
Net 250.00
VAT19% 47.50
TOTAL 297.50EUR"""


def test_extractor_handles_the_clean_text_layer():
    p = D.extract(CLEAN)
    assert p["invoice_no"] == "E-3281"
    assert p["vendor"] == "Delta Logistics"
    assert p["date"] == "2026-03-25"
    assert p["currency"] == "EUR"
    assert p["net"] == 250.00 and p["vat"] == 47.50 and p["total"] == 297.50


def test_extractor_is_not_the_bottleneck_on_ocr_output():
    """The attribution claim depends on this. A brittle extractor would fail on OCR output
    for reasons that have nothing to do with recognition quality -- it would fail on spacing
    -- and the measurement would then blame the wrong stage."""
    p = D.extract(OCR_LIGHT)
    assert p["invoice_no"] == "E-3281"
    assert p["currency"] == "EUR"
    assert p["net"] == 250.00 and p["total"] == 297.50
    # the vendor arrives with its spaces missing, which is the whole point of the next test
    assert p["vendor"] is not None and "Delta" in p["vendor"].replace(" ", "")


def test_the_vendor_failure_is_spacing_not_characters():
    """Measured, and it is why the report prints a strict AND a loose column: up to medium
    degradation OCR loses word boundaries, not characters. Scoring one number from two
    different failure modes is wrong in both directions."""
    p = D.extract(OCR_LIGHT)
    truth = {"vendor": "Delta Logistics"}
    assert D.field_hits(p, truth)["vendor"] == 0
    assert D.field_hits(p, truth, loose=True)["vendor"] == 1


# ---------------------------------------------------------------- snapping
def test_snap_repairs_lost_spaces_exactly():
    v, sc, changed = D.snap("BlueHarbourGmbH", D.VENDORS)
    assert changed and v == "Blue Harbour GmbH"
    assert sc > 0.99, "trigrams should be blind to the missing spaces"


def test_snap_repairs_a_real_character_error():
    v, sc, changed = D.snap("Delta Logistlcs", D.VENDORS)
    assert changed and v == "Delta Logistics"


def test_snap_leaves_a_genuinely_new_value_alone():
    """The dangerous direction. Snapping an unknown supplier onto an existing one is silent
    data corruption -- far worse than a missing field, because it produces a payment to the
    wrong company that every downstream check will accept."""
    v, sc, changed = D.snap("Zenith Robotics", D.VENDORS)
    assert not changed and v == "Zenith Robotics"
    assert sc < D.SNAP_THRESHOLD


def test_snap_is_a_no_op_on_an_exact_value():
    for name in D.VENDORS:
        v, _, changed = D.snap(name, D.VENDORS)
        assert v == name and not changed


def test_trigram_cosine_ranks_by_content_not_by_length():
    a = D.trigrams("Blue Harbour GmbH")
    assert D.cosine(a, D.trigrams("BlueHarbourGmbH")) > 0.99
    assert D.cosine(a, D.trigrams("Granite Tools")) < 0.3


# ---------------------------------------------------------------- validation
def _pred(**kw):
    base = {"invoice_no": "A-1000", "vendor": "Acme Industrial", "date": "2026-01-05",
            "currency": "EUR", "net": 100.0, "vat": 19.0, "total": 119.0}
    return base | kw


def test_clean_prediction_has_no_problems():
    assert D.validate(_pred()) == []


def test_arithmetic_rule_catches_a_misread_digit():
    """The load-bearing validator. A schema check cannot see this: 119.0 and 190.0 are both
    valid floats and both fit the type. Only the requirement that the sum closes catches a
    digit that OCR got wrong, and it needs neither a model nor a label."""
    assert "arithmetic_mismatch" in D.validate(_pred(total=190.0))
    assert "arithmetic_mismatch" not in D.validate(_pred())


def test_missing_fields_and_bad_enums_are_reported():
    assert "missing:vendor" in D.validate(_pred(vendor=None))
    assert "currency_not_in_list" in D.validate(_pred(currency="XYZ"))
    assert "date_malformed" in D.validate(_pred(date="05/01/2026"))


def test_confidence_falls_when_the_arithmetic_breaks():
    good = D.confidence(_pred(), None, D.validate(_pred()))
    bad_pred = _pred(total=190.0)
    bad = D.confidence(bad_pred, None, D.validate(bad_pred))
    assert good > bad + 0.3


def test_confidence_does_not_come_from_self_report():
    """Deliberate, and measured: the OCR engine's own mean score went UP from 0.781 to 0.794
    as the page got worse. A curve built on self-reported confidence would be smooth,
    plausible and meaningless, so the engine score is only a lightly weighted input."""
    p = _pred(total=190.0)
    problems = D.validate(p)
    optimistic = D.confidence(p, 0.99, problems)
    pessimistic = D.confidence(p, 0.10, problems)
    assert optimistic - pessimistic < 0.30, "the engine score must not dominate the checks"
    assert optimistic < 0.75, "a broken sum must stay low however confident the engine is"


# ---------------------------------------------------------------- degradation
def test_degradation_actually_degrades():
    """If the levels stop differing, the attribution table becomes five copies of one row and
    the whole measurement is vacuous."""
    import numpy as np
    from PIL import Image
    img = Image.new("RGB", (400, 120), (255, 255, 255))
    base = np.asarray(D.degrade(img, "clean", seed=1)).astype(float)
    prev = 0.0
    for level in ("light", "medium", "heavy"):
        d = np.asarray(D.degrade(img, level, seed=1)).astype(float)
        diff = float(np.abs(d - base).mean())
        assert diff > prev, f"{level} is not more degraded than the level before it"
        prev = diff


def test_background_colour_reaches_the_raster(tmp_path):
    """The invariance test is only meaningful if the colour survives to the pixels. If
    something flattened it earlier, the test would pass by measuring nothing."""
    import numpy as np
    import random
    rec = D.make_record(random.Random(2), 0)
    white, pink = tmp_path / "w.pdf", tmp_path / "p.pdf"
    D.render_pdf(rec, white, bg=(1, 1, 1))
    D.render_pdf(rec, pink, bg=(0.98, 0.88, 0.90))
    a = np.asarray(D.rasterize(white)).astype(float)
    b = np.asarray(D.rasterize(pink)).astype(float)
    assert np.abs(a - b).mean() > 3.0, "the backgrounds are indistinguishable in pixels"


# ---------------------------------------------------------------- the pixel-space claim
def test_pixel_space_ranks_background_above_letter():
    """The measurement the README leads its retrieval argument with, asserted so it cannot
    quietly stop being true: in raw pixels, the same letter on a different background is LESS
    similar than a different letter on the same background. That is why you vectorise
    entities after recognition rather than patches before it."""
    import numpy as np
    from PIL import Image, ImageDraw

    def patch(bg, ch):
        im = Image.new("RGB", (16, 16), bg)
        ImageDraw.Draw(im).text((3, 1), ch, fill=(0, 0, 0))
        return np.asarray(im, dtype=np.float32).ravel() / 255.0

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    same_letter_other_bg = cos(patch((255, 255, 255), "A"), patch((220, 40, 40), "A"))
    other_letter_same_bg = cos(patch((255, 255, 255), "A"), patch((255, 255, 255), "B"))
    assert same_letter_other_bg < other_letter_same_bg


# ---------------------------------------------------------------- artifacts
def test_committed_attribution_is_self_consistent():
    p = Path("runs/attribute.jsonl")
    if not p.exists():
        pytest.skip("no committed run yet")
    rows = D.load_jsonl(p)
    ceiling = [r for r in rows if r["input"] == "text layer (no OCR)"]
    assert ceiling, "the ceiling row is missing, so nothing can be attributed"
    ok = sum(1 for r in ceiling if all(r["hits"].values()))
    assert ok == len(ceiling), (
        f"the text-layer ceiling is {ok}/{len(ceiling)}; the extractor has become the "
        f"bottleneck and every OCR conclusion below it is noise on top of that")
    heavy = [r for r in rows if r["input"] == "OCR heavy"]
    if heavy:
        assert any(not all(r["hits"].values()) for r in heavy), (
            "heavy degradation now scores perfectly, which means the degradation stopped "
            "degrading and the attribution measures nothing")
