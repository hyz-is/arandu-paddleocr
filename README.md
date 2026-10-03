<p align="center">
  <img src=".github/logo.svg" alt="Arandu" width="180">
</p>

# Arandu PaddleOCR

It reads the text on an image with the PaddleOCR PP-OCRv5 models, and decides
nothing about what the text means.

The models run in ONNX Runtime through
[`arandu-onnx`](https://github.com/hyz-is/arandu-onnx). Everything around them —
resizing, DB post-processing, geometry, perspective crop, CTC decoding — is Go.
A call returns each line of text with its box, its confidence, the detector's
score, and the identity of every model that contributed, for the application to
record as evidence.

## What it is not

- **Not a decision.** It never says a document is valid, a field matches or a
  check passed. It returns text and confidences; thresholds and verdicts belong
  to the application's decision layer.
- **Not a document parser.** It does not know what a CPF, a date or a
  machine-readable zone is. An optional charset is the most it is told.
- **Not a model distribution.** The model weights are not in this repository.
  A separate command downloads them from their pinned upstream revisions and
  verifies their digests; see [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).
- **Not a network client.** The library reads files from the directory it is
  given and never opens a connection.

## Requirements

| | |
| --- | --- |
| Go | 1.26 or later |
| cgo | `CGO_ENABLED=1` and a C compiler (the ONNX Runtime binding is cgo) |
| ONNX Runtime | the shared library, **1.26 or later**; development and CI use **1.30.0**. Installation per operating system is in the [`arandu-onnx` README](https://github.com/hyz-is/arandu-onnx#installing-onnx-runtime) (`brew install onnxruntime` on macOS; the release archive on Linux) |
| Arandu | framework 0.50 or later, and `github.com/hyz-is/arandu-onnx` |
| Models | about 117 MB for all six files (13 MB for the default mobile detector and Latin recognizer); fetched once with `cmd/paddleocr-models` |

## Install

```bash
go get github.com/hyz-is/arandu-paddleocr@v0.1.0
go run github.com/hyz-is/arandu-paddleocr/cmd/paddleocr-models@v0.1.0 -dir /srv/models/ocr
```

## Wire it

An Arandu application registers a module explicitly. There is no service
provider, no container and no discovery, so these are the lines to paste into
`bootstrap/app.go` and there are no others.

The imports, with the other module imports:

```go
import (
	onnx "github.com/hyz-is/arandu-onnx"
	paddleocr "github.com/hyz-is/arandu-paddleocr"
)
```

The construction, in `Build`, before `k.Register`, after the ONNX module. Every
value comes from the application's own typed configuration; this package reads
no environment variable:

```go
	ocrModule, err := paddleocr.New(paddleocr.Config{
		ModelsDir:              cfg.OCR.ModelsDir,
		Detector:               paddleocr.DetectorMobile,
		UseDocOrientation:      cfg.OCR.DocOrientation,
		UseTextlineOrientation: cfg.OCR.TextlineOrientation,
	}, onnxModule.Runtime())
	if err != nil {
		return App{}, err
	}
```

And the registration, inside the `k.Register(...)` call already there, after
`onnxModule`:

```go
		onnxModule,
		ocrModule,
```

`Boot` initializes the runtime if `arandu-onnx` has not yet, and opens the
configured models, verifying each; a missing or altered file stops the boot,
naming the model. `Health` reports whether the models are open and the runtime
under them is usable. `Close` closes the sessions. Registering it adds no
route, no table and no background loop.

## Configuration

| Field | Default | What it does |
| --- | --- | --- |
| `ModelsDir` | required | The directory the model files are in, laid out as `cmd/paddleocr-models` writes it |
| `Detector` | `mobile` | `mobile` or `server` (larger, about five times slower) |
| `Recognizer` | `latin` | `latin` (accented Latin, 838 classes) or `en` (438 classes) |
| `UseDocOrientation` | `false` | Classify and turn the page by 0/90/180/270 before detection |
| `UseTextlineOrientation` | `false` | Turn lines read upside down before recognition |
| `LimitSideLen` | `960` | Longest side the detector sees, 32 to 4000 |
| `Thresh` | `0.3` | Probability a pixel needs to count as text |
| `BoxThresh` | `0.6` | Mean probability a box needs to be kept |
| `UnclipRatio` | `1.5` | How far a box is grown back, up to 10 |
| `MaxCandidates` | `1000` | Text regions examined per page |
| `RecBatchSize` | `6` | Lines per recognizer run, up to 64 |
| `RecMaxWidth` | `3200` | Widest line image the recognizer is given, 320 to 8000 |
| `TextlineThreshold` | `0.9` | Probability of 180 degrees a line needs to be turned |

A numeric field left at zero takes its default. `New` validates the
configuration and returns an error rather than starting half-wired.

`Thresh`, `BoxThresh` and `TextlineThreshold` are the models' own
post-processing constants, not decisions: they choose which pixels and boxes
are text, and nothing about what the text means.

## Read a page

```go
img, err := onnx.DecodeImage(upload) // JPEG or PNG, turned upright by EXIF
page, err := ocrModule.Recognize(ctx, img, paddleocr.Options{})
for _, line := range page.Lines {
	// line.Text, line.Confidence, line.DetectionScore, line.Box, line.Order, line.Flipped
}
evidence := page.Models // one onnx.ModelInfo per model: ID, Version, SHA256, Runtime, Provider, Status
```

A machine-readable zone, read through its alphabet:

```go
page, err := ocrModule.Recognize(ctx, zone, paddleocr.Options{
	Charset: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
})
```

`Page` carries the size of the image as given, the `Rotation` applied to stand
it upright, the `Lines` in reading order, the `Models` that contributed and the
`Timings` of each stage. Every `Line.Box` is in the coordinates of the image as
given; its corners are the text's top-left, top-right, bottom-right and
bottom-left, so on a turned page the first corner is not the image's top-left.

`Recognize` is safe for concurrent use. Calls share the ONNX Runtime sessions,
whose runs are thread-safe; the module's reader-writer lock lets them proceed
together and makes `Close` wait for them. Before `Boot` it returns
`ErrNotBooted`, after `Close` `ErrClosed`.

## The pipeline, stage by stage

1. **Page orientation** (optional): PP-LCNet_x1_0_doc_ori classifies the page
   as turned by 0, 90, 180 or 270 degrees (short side 256, centre crop 224,
   ImageNet normalization in RGB order), and the page is turned upright.
2. **Detection**: the page is scaled so its longer side is at most 960 pixels,
   in multiples of 32, normalized in BGR order, and PP-OCRv5 det returns a
   probability map. The `resize_long: 960` of the model's `inference.yml` is a
   cap on the longer side, never an enlargement. DB post-processing follows:
   binarize at 0.3, 8-connected regions, minimum-area rectangle by rotating
   calipers over the convex hull, mean probability inside the box (kept from
   0.6), unclip by 1.5 × area / perimeter, rectangle again, scale back to the
   page, clip, and reading order with a 10-pixel row tolerance.
3. **Crop**: each box is straightened with a perspective warp (a homography
   from its four corners, bilinear sampling, edge repeated). A crop at least
   1.5 times taller than wide is turned 90 degrees.
4. **Line orientation** (optional): PP-LCNet_x0_25_textline_ori, on the line
   resized to 160×80 with ImageNet normalization in RGB order, turns a line
   whose probability of 180 degrees reaches 0.9.
5. **Recognition**: each line is resized to height 48, normalized to −1..1 in
   BGR order, batched with lines of similar aspect ratio and right-padded with
   zeros, and the output is decoded by CTC greedy decoding: argmax per step,
   repeats collapsed, blank dropped, classes mapped **by index** onto the
   dictionary (it lists some characters twice). Confidence is the mean
   probability of the kept steps. A charset narrows the classes that compete.

The constants are those of each model's own `inference.yml` and of PaddleOCR's
published DB and CTC post-processing; the unit tests pin them with values
worked out by hand.

The pure-Go stages are exported for reuse and testing: `DBPostProcess`,
`BoxScore`, `ConvexHull`, `MinAreaRect`, `MiniBox`, `Unclip`,
`OrderClockwise`, `ReadingOrder`, `PerspectiveTransform`, `WarpQuad`,
`CropSize`, `Rotate`, `DecodeCTC`, `NewVocabulary`, `RecognizerVocabulary`,
`DetectorInputSize` and `RecognizerInputWidth`. `Manifests` and
`LookupManifest` return the embedded model manifests.

## The models

The model files are not in this repository and are not distributed with it.
Each has a manifest, embedded in the package under `manifests/`, recording its
pinned source revision, SHA-256, licences and contract; `Runtime.Open` verifies
every file against it before ONNX Runtime reads a byte.

| id | file | sha256 | size | default |
| --- | --- | --- | --- | --- |
| `pp-ocrv5_mobile_det` | `PP-OCRv5_mobile_det_onnx/inference.onnx` | `a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d` | 4.8 MB | yes |
| `pp-ocrv5_server_det` | `PP-OCRv5_server_det_onnx/inference.onnx` | `10803475a591f7dc623e24670fb5752ec94d39a1f8cf069aac1b6f0ce19cfc85` | 88 MB | `Detector: server` |
| `latin_pp-ocrv5_mobile_rec` | `latin_PP-OCRv5_mobile_rec_onnx/inference.onnx` | `7888113072263cb471b93f66dd5e2ad70548dc526fa1ace760d0d973dd121498` | 8.0 MB | yes |
| `en_pp-ocrv5_mobile_rec` | `en_PP-OCRv5_mobile_rec_onnx/inference.onnx` | `b5f833dfc5d0eb71da397b4efa06ebeee9b431b690a47d6af40d77d8eabc557f` | 7.8 MB | `Recognizer: en` |
| `pp-lcnet_x0_25_textline_ori` | `PP-LCNet_x0_25_textline_ori_onnx/inference.onnx` | `94a6a0a0425f2b5f08b5df72086f2d72abe40f1d22f6d12d2cd83674f11f2ff3` | 1.0 MB | when enabled |
| `pp-lcnet_x1_0_doc_ori` | `PP-LCNet_x1_0_doc_ori_onnx/inference.onnx` | `af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92` | 6.8 MB | when enabled |

| | all six models |
| --- | --- |
| Source | the official `PaddlePaddle/<name>_onnx` repositories on Hugging Face, each URL pinned to a commit (in the manifest and in [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md)) |
| Code licence | Apache-2.0 (PaddleOCR) |
| Model licence | Apache-2.0 |
| Commercial use | yes, under Apache-2.0 |
| Training data | not enumerated by the upstream repositories; see the PaddleOCR PP-OCRv5 documentation. Not evaluated here on real identity documents |
| Status | `approved`, fp32 |

No model here is `dev-only`, so this module has no switch to refuse one; a
`dev-only` manifest would load and say so in every `ModelInfo.Status`.

The recognizers' dictionaries — the `character_dict` of each model's
`inference.yml`, one character per line — are embedded under `dictionaries/`
and checked against their digest when a vocabulary is built. They are
Apache-2.0 material from PaddlePaddle; see [NOTICE](NOTICE).

### Download and verify

```bash
go run github.com/hyz-is/arandu-paddleocr/cmd/paddleocr-models@v0.1.0 -dir /srv/models/ocr
go run github.com/hyz-is/arandu-paddleocr/cmd/paddleocr-models@v0.1.0 -dir /srv/models/ocr -check
go run github.com/hyz-is/arandu-paddleocr/cmd/paddleocr-models@v0.1.0 -dir /srv/models/ocr \
  -only pp-ocrv5_mobile_det,latin_pp-ocrv5_mobile_rec
```

The command downloads each file over HTTPS from the revision its manifest pins,
follows only HTTPS redirects, caps each download at 512 MB, hashes while it
writes, and renames the file into place only when the digest matches. A file
already present with the right digest is left alone. `-check` verifies what is
there and connects to nothing; it exits non-zero when a file is missing or
altered. **The command is the only code in this repository that opens a
network connection**, and an application never links it, which is why
`arandu.mod.toml` declares `network = false`.

## Accuracy and speed

Measured by the feature tests on synthetic pages drawn in Go Regular (the Go
fonts, BSD licence) with invented text — "REPÚBLICA FEDERATIVA DO BRASIL",
"NOME: MARIA DA SILVA TESTE", "CPF 123.456.789-09" and an MRZ-like
"IDBRA123456789<<<<<<<<<<<<<<<" — at 32, 40 and 48 pixels and tilted by 0 and
±2 degrees: every line found on every page, 32 of 36 lines read exactly, a
character error rate of 0.65 %. The accented line is read exactly every time;
all four misses are the MRZ-like line's run of fifteen `<`, read one or two
characters short — a long run of one character is where CTC greedy decoding is
weakest. Pages turned by 90, 180 and 270 degrees are stood upright by the page
classifier, and upside-down lines are turned by the line classifier.

Speed, on the 851×472 four-line page, CPU provider, ONNX Runtime 1.30.0, on an
Apple M4 Max (`go test -run '^$' -bench . ./tests/Feature/`):

| configuration | per page |
| --- | --- |
| mobile detector | about 95–120 ms |
| mobile detector, both orientation stages | about 95–140 ms |
| server detector | about 580 ms |

These figures are indicative. They are on clean synthetic text, on one machine,
and are not a measurement on photographs of documents.

## Limitations, and what needs evaluation

- Accuracy is measured on synthetic text only. Before relying on it, measure
  character error rate on a consented, representative set of the documents you
  will read — camera captures, glare, blur, low resolution, worn prints.
- Lines are rectangles: curved text is boxed by its minimum-area rectangle.
- Vertical text is turned 90 degrees and read as a horizontal line; nothing
  here reads top-to-bottom scripts.
- The crop interpolates bilinearly where PaddleOCR uses bicubic.
- With line orientation on and page orientation off, an upside-down page is
  read line by line correctly but in reverse reading order.
- Long runs of one character, such as the filler `<` of a machine-readable
  zone, may be read a character or two short. A caller that knows the zone's
  fixed length pads or checks it; this package does not.

## Security and privacy

Images of identity documents are personal data, and in many jurisdictions
(LGPD, GDPR) the data on them is sensitive. This package keeps nothing: it does
not store, log or transmit the image or the text it reads. Storing the raw
image, encrypting it, minimising what is kept and auditing access are the
application's responsibility.

`DecodeImage` bounds the pixel count before decoding, and every model file is
verified against its digest before the native runtime parses it. Report
vulnerabilities privately; see [SECURITY.md](SECURITY.md).

## Why there is no table, policy or route

It wraps a technology and owns no record, so there is nothing for a policy to
decide about; a route would read any image for anybody who sent one. Text is
read through the application's controllers and Grants. `arandu.mod.toml`
declares `filesystem = true` and nothing else, and `tests/Unit/audit_test.go`
checks that declaration against what the code calls.

## Tests

```bash
export GOWORK=off CGO_ENABLED=1
test -z "$(gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go'))"
go build ./...
go vet ./...
go test -race -count=1 ./...
```

The pipeline tests read the models from the directory `PADDLEOCR_MODELS_DIR`
names, or from `../../models/ocr_official` when it is unset, and skip, saying
where they looked, when the files are not there; they skip too when ONNX
Runtime is absent. No test downloads anything. To run them:

```bash
go run ./cmd/paddleocr-models -dir /tmp/ocr-models
PADDLEOCR_MODELS_DIR=/tmp/ocr-models go test -race -count=1 ./...
```

CI installs ONNX Runtime 1.30.0, fetches and verifies the models with the same
command, and runs the suite with nothing skipped. Every test page is drawn in
Go with invented text; no photograph or real document is read.

## Versioning and releases

The module follows [Semantic Versioning](https://semver.org/). While the major
version is `v0`, a minor release may break the API, and every break is named in
[UPGRADE.md](UPGRADE.md); CI compares the exported API with the last tag. A
changed default (`LimitSideLen`, `BoxThresh`, the detector) or a new model
revision changes every application that never set it, and is always a
changelog line.

A release is a tag `vX.Y.Z` on `main` whose notes are the `## [X.Y.Z] - date`
entry already committed in [CHANGELOG.md](CHANGELOG.md). The release workflow
refuses a `go.mod` that still replaces a module with a local path, re-runs the
gates on the exact tag archive with the models fetched, audits known
vulnerabilities, and publishes those notes. A published version is immutable.

## Licence

The code is MIT; see [LICENSE.md](LICENSE.md). Copyright HYZIS - Serviços
Digitais Ltda.

The dictionaries under `dictionaries/` are PaddlePaddle's, under the Apache
License 2.0; see [NOTICE](NOTICE). The model weights are not part of this
repository; each is under its own licence, listed in
[THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).
