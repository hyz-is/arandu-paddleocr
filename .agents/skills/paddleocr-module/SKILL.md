---
name: paddleocr-module
description: Change this Arandu package's OCR pipeline. Use when the request is to "add a model", "support another language", "change the detector", "tune the thresholds", "fix a box", "the text is wrong", "decode differently", "add a charset", "read vertical text", "add a route", "parse the CPF", "return whether the document is valid", or when a change touches pipeline.go, db.go, geometry.go, crop.go, ctc.go, models.go, config.go, module.go, manifests/, dictionaries/ or cmd/paddleocr-models. Covers what may and may not change, where each constant comes from, and the tests that hold each property.
license: MIT
---

# Changing the pipeline

## Refused outright

- **A decision or document semantics.** No "valid", no field extraction, no
  CPF or MRZ check digit. The package returns text, boxes, confidences and
  `Models`. A charset is the most it is told about a document.
- **A model without a manifest,** or a manifest without a pinned revision, a
  digest and its licences.
- **A download in the library or in a test.** `cmd/paddleocr-models` is the
  only network code, and the tests run it in `-check` mode only.
- **Python,** anywhere, including "just to compare against PaddleOCR".
- **A route.** Text is read through the application's controllers.

## Where the constants come from

Each model directory ships an `inference.yml`; it wins over memory. The
detector: `DetResizeForTest resize_long 960` (applied as a cap on the longer
side, multiples of 32), `NormalizeImage` ImageNet mean/std on BGR, DB `thresh
0.3`, `box_thresh 0.6`, `unclip_ratio 1.5`, `max_candidates 1000`. The
recognizer: `RecResizeImg [3, 48, 320]`, `(x/255 - 0.5)/0.5` on BGR, CTC with a
space class. The classifiers: ImageNet on RGB, 160×80 for lines, short side
256 and centre crop 224 for pages. A change to any of them names the yml line
in the commit message.

## Decoding

Class 0 is the blank; class i is dictionary line i-1; the last class is the
space. Decode by index: the latin dictionary lists some characters twice, and
both classes are real. The recognizers already output probabilities;
`DecodeCTC` applies a softmax only to a step that is not a distribution.

## Adding a model

1. Pin the Hugging Face revision; take the digest from the LFS object id and
   check it against the file you have.
2. Write `manifests/<id>.json`: id, version (12-character revision), file
   (`<repo>/inference.onnx`), sha256, `source_url` at
   `/resolve/<40-hex revision>/`, licences, `status`, inputs and outputs as
   the graph declares them, precision.
3. A recognizer also needs its dictionary under `dictionaries/` and an entry in
   `dictionaries` in models.go with its digest and line count.
4. Extend `tests/Unit/models_test.go`, and a feature test that reads a page
   with it.

## Golden values

Unit tests compute their expected values by hand, in a comment beside the
assertion, never by running the code under test. The feature tests draw their
pages with the Go fonts and invented text.

## Concurrency

`Recognize` holds the module's read lock for the whole call; `Close` the write
lock. Sessions are shared. Any change here runs `go test -race ./...` with the
library and the models present, and reads the skip lines.
