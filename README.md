# document-processor

[![CI](https://github.com/NotEl1gible/document-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/NotEl1gible/document-processor/actions/workflows/ci.yml)

PDF or scan in, validated structured data out, low-confidence results routed to a human.
The pipeline is the easy part. This measures three things a document-AI writeup usually
does not.

---

## 1. Where the error actually comes from

The same extractor and the same scorer run on the exact PDF text layer and on the OCR of the
same page. The difference **is** the OCR contribution — measured, not argued.

```
$ python docproc.py attribute

  input                 invoice_no  vendor  date  currency   net   vat  total   all  loose
  text layer (no OCR)         1.00    1.00  1.00      1.00  1.00  1.00   1.00  1.00   1.00
  OCR clean                   1.00    1.00  1.00      1.00  1.00  1.00   1.00  1.00   1.00
  OCR light                   1.00    0.67  1.00      1.00  1.00  1.00   1.00  0.67   1.00
  OCR medium                  1.00    0.58  1.00      1.00  1.00  1.00   1.00  0.58   1.00
  OCR heavy                   0.17    0.00  0.08      0.58  0.00  0.17   0.75  0.00   0.00
```

The first row is the **ceiling**: it is perfect, so the extractor is not the bottleneck and
100% of the loss on this corpus is OCR. Two things fall out that an aggregate accuracy would
have hidden — **numbers survive far longer than proper nouns**, and `total` survives longest
of all because it is set in bold at a larger size.

**The `strict` vs `loose` split is the finding**, and it came from reading predictions rather
than aggregates. Up to medium degradation OCR loses **spaces, not characters**:

```
truth: Blue Harbour GmbH      OCR: BlueHarbourGmbH
```

Strictly that is a miss (0.58). With whitespace collapsed it is a hit (1.00). Reporting one
number from two different failure modes is wrong in **both** directions at once: it overstates
the problem below heavy degradation and adds nothing above it.

---

## 2. The fix is not better OCR

It is vectorisation at the layer where it pays. Character trigrams and cosine against the
supplier master:

```
BlueHarbourGmbH   cos 1.000  ->  Blue Harbour GmbH     (lost spaces)
Delta Logistlcs   cos 0.812  ->  Delta Logistics       (a real character error)
Zenith Robotics   cos 0.250      left alone            (genuinely not in the list)
```

| | without snapping | with snapping |
|---|---|---|
| OCR light | 0.67 | **1.00** |
| OCR medium | 0.58 | **1.00** |
| OCR heavy | 0.00 | 0.00 |

**Below a degradation threshold you do not need better OCR at all — you need a lookup table.
Above it the lookup cannot help and you need a better scan or a human.** That is a sharper
conclusion than any accuracy number, and it generalises: wherever a closed list exists —
suppliers, currencies, country codes, catalogue — the model guesses and the list corrects.

Note what snapping did *not* do: heavy degradation stayed at 0.00, so it introduced no false
snaps. That direction matters more than the gain. Snapping an unknown supplier onto an
existing one is silent data corruption — a payment to the wrong company that every downstream
check accepts — which is why the threshold is explicit and `test_snap_leaves_a_genuinely_new_value_alone`
guards it.

### Why not just retrieve the image patch?

Because retrieval needs a distance, and in pixel space the distance is wrong. Measured:

```
same letter 'A', white background vs red     0.762
DIFFERENT letters 'A' vs 'B', same background 0.976
```

Nearest-neighbour on raw pixels retrieves by **background colour**, not by letter. You cannot
escape the embedding — retrieval and embedding are the same problem, and the embedding is
what defines "near". So vectorise **entities after recognition**, not patches before it.
`test_pixel_space_ranks_background_above_letter` asserts that measurement so it cannot quietly
stop being true.

---

## 3. Invariance — an instrument that needs no labels

The same document on five paper colours. Every extracted field must be identical, because
changing the paper cannot change the amount on an invoice. The invariant follows from what a
document **is**, so no annotation is involved anywhere.

```
$ python docproc.py invariance

  extraction identical across all backgrounds: 0.83 (5/6) [0.44, 0.97]

  output MOVED when only the paper colour moved:
    doc-0004  on cream   vendor: 'BlueHarbourGmbH' -> 'Blue Harbour GmbH'
    doc-0004  on grey    vendor: 'BlueHarbourGmbH' -> 'Blue HarbourGmbH'
```

Three different answers for one document on three paper colours. That also shows the missing
spaces are **not a deterministic property of the engine** but instability at the binarisation
threshold — which makes the vocabulary snap a necessity rather than a cosmetic, and with it
invariance goes to **1.00 (6/6)**.

This is the only instrument here that would still work on a corpus whose ground truth nobody
has.

---

## 4. The curve that decides the economics

Accuracy is the wrong headline. **A system that is 99% accurate but cannot say which 1% is
wrong has a straight-through rate of zero, because everything must be checked.**

```
$ python docproc.py curve

   threshold        auto-processed        escaped errors   review
        0.00    1.00 (48/48)              0.25 (12/48)       0.00
        0.30    0.81 (39/48)               0.08 (3/39)       0.19
        0.50    0.75 (36/48)               0.00 (0/36)       0.25
        0.90    0.75 (36/48)               0.00 (0/36)       0.25
        0.95     0.17 (8/48)                0.00 (0/8)       0.83
```

**At threshold 0.50 nothing wrong escapes and 75% still go through untouched.** Two features
of this curve are worth more than the headline:

- **0.50 to 0.90 is a plateau.** Raising the threshold in that range buys nothing — same 36
  documents, same zero errors. Confidence here is not a smooth quantity; it is close to a
  discrete flag for *did the arithmetic close*. Tuning it finely is wasted effort, and a
  smooth-looking curve would have been a lie.
- **0.95 is a cliff.** Anyone who set the threshold high "to be safe" would quadruple the
  manual load for **zero** reduction in escaped errors. That mistake is invisible without the
  curve.

Two costs are tracked separately because different people pay them: **review load** in salary
(and a reviewer is *slower* per document than a typist — they must read the page *and* check
the machine), and **escaped errors** downstream in a wrong payment, found much later if at all.

### The confidence is built from checks, not self-report

Deliberately. Measured here: the OCR engine's own mean score went **up** from 0.781 to 0.794
as the page got worse. A curve built on it would be smooth, plausible and meaningless.

`net + vat = total` is the load-bearing rule. A schema check cannot catch a misread digit —
`18478.33` and `18478.83` are both valid floats — but arithmetic that has to close catches it
with no model and no label.

---

## Ground truth is free, and that is the design

The documents are **rendered from** the records with PyMuPDF, so every field value is known
exactly. No annotation, no judge, no hand-labelling — and it removes a class of error every
hand-labelled corpus carries: the label that disagrees with the document.

The degradation applied afterwards is **real** — rasterisation, downscale-and-back, blur,
sensor noise, rotation, JPEG — so the OCR stage faces a genuine problem while the labels stay
perfect. `test_degradation_actually_degrades` fails if the ladder ever flattens.

What this costs, stated plainly: the corpus is one invoice layout. It says nothing about
handwriting, stamps, multi-page tables or non-Latin scripts, and the *shape* of the curve
would change on a harder corpus even though the *method* would not.

---

## Stack notes

**RapidOCR** (PP-OCR exported to ONNX) rather than Tesseract: CPU-only, no torch, no system
binary — so `pip install -r requirements.txt` is enough for a reader to actually run this.
Tesseract needs a native install, and a portfolio project whose first command fails on a
missing binary is a project nobody runs.

**The first step of the pipeline is not OCR.** If the PDF carries a text layer you get the
characters exactly, plus the coordinates OCR would have to guess. A large share of enterprise
documents are digital-born, and running OCR over them does not merely waste time — it
*introduces* error where there was none.

```
INVOICE      bbox=(72,83)-(138,105)
#A-4471      bbox=(147,83)-(207,105)
```

---

## Layout

```
docproc.py        corpus generation, degradation, OCR, extraction, validation,
                  snapping, and the four instruments
docs/truth.jsonl  the ground truth the PDFs were rendered from
runs/             the committed artifacts the numbers above come from
test_docproc.py   19 hermetic tests; none loads an OCR model
```

```bash
pip install -r requirements.txt
python -m pytest -q
python docproc.py generate --n 40
python docproc.py attribute --n 12 --snap
python docproc.py invariance --n 6
python docproc.py curve
```

Runs fully offline. `--provider anthropic` swaps a real model in behind the same extractor
signature; every measurement above is from the deterministic path.

## Deliberately not built

- **A vision-model branch** (image straight to JSON). It is the interesting comparison and it
  is named rather than faked: a decoder that is a language model produces *fluent* errors,
  where CTC-style recognition produces *garbled* ones — and garbled errors are the ones a
  validator can see. That claim needs a measurement this repo does not yet have.
- **Celery / a queue.** The measurements are batch; a task queue would be scaffolding around
  a result, not a result.
- **A review UI.** The routing decision and its cost are measured in `curve`; the screen that
  displays them adds no evidence.

## License

MIT
