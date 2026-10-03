<p align="center">
  <img src=".github/logo.svg" alt="Arandu" width="180">
</p>

# Arandu PaddleOCR

An Arandu package that reads the text on an image with the PaddleOCR PP-OCRv5
models, run through ONNX Runtime by
[`arandu-onnx`](https://github.com/hyz-is/arandu-onnx). It finds the lines,
straightens them, reads them, and returns each line with its box, its
confidence and the models that produced it. It decides nothing: what the text
means — a name, a CPF, a machine-readable zone — is the application's
business.

The models run in ONNX Runtime. Everything around them is Go:

1. **Page orientation** (optional): PP-LCNet_x1_0_doc_ori classifies the page
   as turned by 0, 90, 180 or 270 degrees (short side 256, centre crop 224,
   ImageNet normalization in RGB order), and the page is turned upright.
2. **Detection**: the page is scaled so its longer side is at most 960 pixels,
   in multiples of 32, normalized in BGR order, and PP-OCRv5 det returns a
   probability map. (The `resize_long: 960` of its `inference.yml` is applied as
   PaddleX applies it to the PP-OCRv5 detectors: a cap on the longer side,
   never an enlargement.) DB post-processing follows: binarize at 0.3, 8-connected
   regions, minimum-area rectangle by rotating calipers over the convex hull,
   mean probability inside the box (kept from 0.6), unclip by 1.5 × area /
   perimeter, rectangle again, scale back to the page, clip, and PaddleOCR's
   reading order with a 10-pixel row tolerance.
3. **Crop**: each box is straightened with a perspective warp (a homography
   from its four corners, bilinear sampling, edge repeated). A crop at least
   1.5 times taller than wide is turned 90 degrees.
4. **Line orientation** (optional): PP-LCNet_x0_25_textline_ori, on the line
   resized to 160×80 with ImageNet normalization in RGB order, turns a line
   whose probability of 180 degrees reaches 0.9.
5. **Recognition**: each line is resized to height 48, normalized to −1..1 in
   BGR order, batched with lines of similar aspect ratio and right-padded with
   zeros, and the recognizer's output is decoded by CTC greedy decoding:
   argmax per step, repeats collapsed, blank dropped, classes mapped **by
   index** onto the dictionary (it lists some characters twice). Confidence is
   the mean probability of the kept steps. An optional charset narrows the
   classes that compete, for a machine-readable zone say.

The constants are those of each model's own `inference.yml` and of PaddleOCR's
DB and CTC post-processing; the unit tests pin them with values worked out by
hand.

## Build requirement

Those of `arandu-onnx`: `CGO_ENABLED=1`, a C compiler, and the ONNX Runtime
shared library, 1.26 or later, on the machine that runs the application.
Development and the tests use 1.30.0.

## Install

```bash
go get github.com/hyz-is/arandu-paddleocr
```

Until `arandu-onnx` is tagged, `go.mod` resolves it from the sibling checkout:

```
replace github.com/hyz-is/arandu-onnx => ../arandu-onnx
```

That line is removed, and `arandu-onnx` required at its tag, once that module
is published. The release workflow refuses to publish this one while it is
there.

## The models

The model files are not in this repository. Each has a manifest, embedded in
the package under `manifests/`, recording its source revision, SHA-256,
licences and contract; `Runtime.Open` verifies every file against it before
ONNX Runtime reads a byte.

| id | model | revision | sha256 | size |
| --- | --- | --- | --- | --- |
| `pp-ocrv5_mobile_det` | PP-OCRv5_mobile_det | `e6f4fa85f00e` | `a431985659dc…` | 4.8 MB |
| `pp-ocrv5_server_det` | PP-OCRv5_server_det | `dcf248c4dcc0` | `10803475a591…` | 88 MB |
| `latin_pp-ocrv5_mobile_rec` | latin_PP-OCRv5_mobile_rec, 838 classes | `89d3a50e2c27` | `788811307226…` | 8.0 MB |
| `en_pp-ocrv5_mobile_rec` | en_PP-OCRv5_mobile_rec, 438 classes | `3fafbc3b5dcf` | `b5f833dfc5d0…` | 7.8 MB |
| `pp-lcnet_x0_25_textline_ori` | PP-LCNet_x0_25_textline_ori | `aea1f18d9733` | `94a6a0a0425f…` | 1.0 MB |
| `pp-lcnet_x1_0_doc_ori` | PP-LCNet_x1_0_doc_ori | `7330ab703912` | `af9a0a4f317f…` | 6.8 MB |

All come from the official `PaddlePaddle/<name>_onnx` repositories on Hugging
Face, under Apache-2.0, `approved`, fp32. The full digests are in the
manifests. The recognizers' dictionaries — the `character_dict` of each
model's `inference.yml`, one character per line — are embedded under
`dictionaries/` and checked against their digest when the module is built.

A models directory holds each file under its repository's name, as the
manifests' `file` fields say (`PP-OCRv5_mobile_det_onnx/inference.onnx`, …).
`cmd/paddleocr-models` fills one:

```bash
go run github.com/hyz-is/arandu-paddleocr/cmd/paddleocr-models -dir /srv/models/ocr
go run github.com/hyz-is/arandu-paddleocr/cmd/paddleocr-models -dir /srv/models/ocr -check
```

It downloads each file over HTTPS from the revision its manifest pins, follows
only HTTPS redirects, hashes while it writes, and keeps the file only when the
digest matches. `-only` takes a comma-separated list of ids; `-check` verifies
what is there and connects to nothing. **The command is the only code here
that opens a network connection.** The library an application links reads
files from the directory it is given, which is why `arandu.mod.toml` declares
`network = false`.

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

## Read a page

```go
img, err := onnx.DecodeImage(upload) // JPEG or PNG, turned upright by EXIF
page, err := ocrModule.Recognize(ctx, img, paddleocr.Options{})
for _, line := range page.Lines {
	// line.Text, line.Confidence, line.DetectionScore, line.Box, line.Order
}
evidence := page.Models // ID, Version, SHA256, Runtime, Precision, Provider, Status
```

A machine-readable zone, read through its alphabet:

```go
page, err := ocrModule.Recognize(ctx, zone, paddleocr.Options{
	Charset: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
})
```

`Page` carries the size of the image as given, the `Rotation` applied to stand
it upright, the `Lines` in reading order, the `Models` that contributed (each
`onnx.ModelInfo`, with digest and status) and the `Timings` of each stage.
Every `Line.Box` is in the coordinates of the image as given; its corners are
the text's top-left, top-right, bottom-right and bottom-left, so on a turned
page the first corner is not the image's top-left.

The pure-Go stages are exported for reuse and testing: `DBPostProcess`,
`BoxScore`, `ConvexHull`, `MinAreaRect`, `MiniBox`, `Unclip`,
`OrderClockwise`, `ReadingOrder`, `PerspectiveTransform`, `WarpQuad`,
`CropSize`, `Rotate`, `DecodeCTC`, `Vocabulary`, `DetectorInputSize` and
`RecognizerInputWidth`.

## Configuration

| Field | Default | What it does |
| --- | --- | --- |
| `ModelsDir` | required | The directory the model files are in |
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

A numeric field left at zero takes its default.

## Accuracy and speed

Measured by the feature tests on synthetic pages drawn in Go Regular (the Go
fonts, BSD licence) with invented text — "REPÚBLICA FEDERATIVA DO BRASIL",
"NOME: MARIA DA SILVA TESTE", "CPF 123.456.789-09" and an MRZ-like
"IDBRA123456789<<<<<<<<<<<<<<<" — at 32, 40 and 48 pixels and tilted by 0 and
±2 degrees: every line found on every page, 32 of 36 lines read exactly, a
character error rate of 0.65 %. The accented line is read exactly every time;
all four misses are the MRZ-like line's run of fifteen `<`, read one or two
characters short — a long run of one character is where CTC greedy decoding
is weakest. Pages turned by 90, 180 and 270 degrees
are stood upright by the page classifier, and upside-down lines are turned by
the line classifier.

On an Apple M4 Max, CPU provider, ONNX Runtime 1.30.0, the 851×472 four-line
page takes about 120 ms with the mobile detector, 138 ms with both orientation
stages on, and 580 ms with the server detector
(`go test -run '^$' -bench . ./tests/Feature/`).

These are figures on clean synthetic text, not on photographs of documents.

## Concurrency

`Recognize` is safe for concurrent use. Calls share the ONNX Runtime sessions,
whose runs are thread-safe; the module's reader-writer lock lets them proceed
together and makes `Close` wait for them.

## Why there is no table, policy or route

This repository has the shape of the Arandu package skeleton without its demo
entity, like `arandu-onnx`. It wraps a technology and owns no record, so there
is nothing for a policy to decide about; a route would read any image for
anybody who sent one. Text is read through the application's controllers and
Grants.

## Limitations

- Lines are rectangles: curved text is boxed by its minimum-area rectangle.
- Vertical text is turned 90 degrees and read as a horizontal line; nothing
  here reads top-to-bottom scripts.
- The crop interpolates bilinearly where PaddleOCR uses bicubic.
- With line orientation on and page orientation off, an upside-down page is
  read line by line correctly but in reverse reading order.
- Accuracy is measured on synthetic text only.
- Long runs of one character, such as the filler `<` of a machine-readable
  zone, may be read a character or two short. A caller that knows the zone's
  fixed length pads or checks it; this package does not.

## Gates

```sh
export GOWORK=off CGO_ENABLED=1
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

The pipeline tests read the models from `KUAA_MODELS_DIR`, or from the KUAA
workspace's `models/ocr_official` two levels above this checkout, and skip,
saying where they looked, when the files are not there; they skip too when
ONNX Runtime is absent. A skip is a reason to fetch the models, not a green
result for the pipeline. No test downloads anything.

## Licence

MIT. See [LICENSE.md](LICENSE.md).

The dictionaries under `dictionaries/` are the character lists of the
PaddleOCR recognizers, from PaddlePaddle's model repositories, under the
Apache License 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>). The model
files, which this repository does not contain, are under the same licence.
