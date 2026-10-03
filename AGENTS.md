# Working on Arandu PaddleOCR

This is an Arandu package that reads the text on an image with the PaddleOCR
PP-OCRv5 models, run through ONNX Runtime by `arandu-onnx`. It returns lines
of text with their boxes, confidences and the `ModelInfo` of every model that
contributed.
It is a Go module somebody `go get`s and registers by hand in their own
`bootstrap/app.go`, which is the whole difference from working in an
application. There is no service provider, no container and no discovery — if a
line of wiring is not written in the installer's repository, it does not happen.

Read `.agents/skills/` before writing code. Each skill is a procedure, and the
one you need is named by the situation you are in.

## The rules

These come from the engines' design, not from taste. They were written for
KUAA, the identity-verification application these engines were built for, and
they hold for every installer. A change that breaks one is not
merged.

1. **No decisions, and no document semantics.** This package returns text,
   boxes, confidences and the models behind them. It never says a document is
   valid, a field matches or a check passed, and it does not know what a CPF or
   an MRZ is: a charset is the most it is told. Those belong to the
   application.
2. **No Python.** Not as a runtime, not in a gate, not in a test, not to
   generate a fixture. Models arrive as already-exported ONNX files; test pages
   are drawn in Go with the Go fonts.
3. **Model provenance is mandatory.** Every model is opened through its
   embedded manifest, which `Runtime.Open` verifies. A new model is a manifest
   with a pinned revision, a digest and its licences, reviewed before its
   status is `approved`.
4. **The library does not touch the network.** `cmd/paddleocr-models` is the
   one program that downloads, and an application never links it. The library
   reads model files from `Config.ModelsDir`.
5. **Configuration is typed.** `Config` comes from the application. This
   package reads no environment variable; only the tests read
   `PADDLEOCR_MODELS_DIR`.
6. **Fixtures are synthetic.** No real document, face, name or number in a
   test, an example or a log line.

## The gates

Nothing is finished until all four exit zero.

```sh
export GOWORK=off CGO_ENABLED=1
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

`CGO_ENABLED=1` is required: arandu-onnx's binding is cgo. The pipeline tests
need the ONNX Runtime shared library and the model files, from
`PADDLEOCR_MODELS_DIR` or `../../models/ocr_official`; where either
is absent they skip saying why, and the skip is a reason to fix the machine,
not a green result for the pipeline.

`GOWORK=off` matters because this checkout may sit beside a Go workspace that
does not list it.

While `go.mod` replaces `github.com/hyz-is/arandu-onnx` with `../arandu-onnx`,
build from a checkout that has it beside this one. Once arandu-onnx is required
at a tag the replace is gone, the checkout builds on its own, and the release
workflow, which refuses a local replace, can publish this module.

`aru doctor` reads applications, not libraries. That is why
`tests/Unit/audit_test.go` compares `arandu.mod.toml` with what the code calls.

## What this repository holds

```
module.go     registration: Name, Routes (none), Boot, Health, Close
config.go     what the application passes in, and the defaults
models.go     the embedded manifests and dictionaries
pipeline.go   Recognize: orientation, detection, crop, recognition
db.go         DB post-processing of the detector's probability map
geometry.go   hull, minimum-area rectangle, unclip, reading order
crop.go       homography, perspective crop, rotation
ctc.go        vocabulary, charset mask, CTC greedy decoding
manifests/    one manifest per model
dictionaries/ the recognizers' character lists
cmd/paddleocr-models/  downloads and verifies the model files
NOTICE        the licence of the bundled PaddlePaddle dictionaries
THIRD_PARTY_MODELS.md  the upstream, revision, digest and licence of every model
```

Tests live under `tests/Unit` (geometry, DB, crop, CTC, manifests, config, the
audit) and `tests/Feature` (the module, the command's check mode, and every
test that runs a model).

## What does not exist here, on purpose

| A change reaches for | What is here instead |
| --- | --- |
| a verdict, a field parser, a CPF check | text and confidences. The application interprets them |
| a model file in the repository | its manifest; the file is fetched by `cmd/paddleocr-models` |
| a download in the library or a test | `cmd/paddleocr-models`, run once where the application is deployed |
| loading a model by path alone | `Runtime.Open(manifest, ModelsDir)` |
| a character looked up in the dictionary by value | decoding by class index: the dictionary has duplicates |
| a route that reads an image | nothing. Text is read through the application's controllers |
| OpenCV, a Python reference, a fixture made outside Go | the Go implementations and pages drawn with the Go fonts |

## Writing code

Everything in the source is in English: identifiers, doc comments, internal
comments, error messages, log messages, and the names and messages of tests.

Every exported symbol carries a doc comment, and the comment documents the
symbol and nothing else. A date, an issue number, a version in progress or the
name of another repository does not belong there.

Tests go under `tests/`, in a capitalized category directory declaring a
lowercase external package: `tests/Unit` holds `package unit_test`,
`tests/Feature` holds `package feature_test`. Both import the package by its
module path, which is what makes them see exactly what a caller sees.
