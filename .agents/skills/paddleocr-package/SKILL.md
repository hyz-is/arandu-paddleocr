---
name: paddleocr-package
description: Install, wire and use the Arandu PaddleOCR package (Go, Arandu) in an application. Use when the request is to "install arandu-paddleocr", "read text from an image", "OCR a document", "read the MRZ", "go get github.com/hyz-is/arandu-paddleocr", "wire it into bootstrap/app.go", "download the OCR models", "the boot fails opening pp-ocrv5_mobile_det", "model digest mismatch", "the text comes back empty", or when a project's go.mod already requires github.com/hyz-is/arandu-paddleocr. Covers the build requirement, the models directory, the wiring and where it goes, the Config fields, the result, and what each error means.
license: MIT
---

# Using Arandu PaddleOCR

A module that reads the text on an image with the PP-OCRv5 models on
`arandu-onnx`'s runtime. It returns lines, boxes, confidences and the models
that produced them, and decides nothing. It is registered by hand — there is no
service provider, no container and no discovery.

## Build requirement

`CGO_ENABLED=1`, a C compiler, and the ONNX Runtime shared library (1.26 or
later) on the machine that runs the application, as `arandu-onnx` requires.

## Install and fetch the models

```bash
go get github.com/hyz-is/arandu-paddleocr@v0.1.0
go run github.com/hyz-is/arandu-paddleocr/cmd/paddleocr-models@v0.1.0 -dir /srv/models/ocr
```

The command downloads every model over HTTPS from its pinned revision and keeps
a file only when its SHA-256 matches; `-check` verifies without connecting,
`-only` takes ids. Run it where the application is deployed or when its image is
built. The library itself never downloads.

## The wiring, and where it goes

All of it is in `bootstrap/app.go`, after the `arandu-onnx` module.

```go
import (
	onnx "github.com/hyz-is/arandu-onnx"
	paddleocr "github.com/hyz-is/arandu-paddleocr"
)
```

In `Build`, before `k.Register`, with every value from the application's typed
configuration (the package reads no environment variable):

```go
	ocrModule, err := paddleocr.New(paddleocr.Config{
		ModelsDir:              cfg.OCR.ModelsDir,
		UseDocOrientation:      cfg.OCR.DocOrientation,
		UseTextlineOrientation: cfg.OCR.TextlineOrientation,
	}, onnxModule.Runtime())
	if err != nil {
		return App{}, err
	}
```

Inside the `k.Register(...)` call, after `onnxModule`:

```go
		ocrModule,
```

## Reading

```go
img, err := onnx.DecodeImage(r)
page, err := ocrModule.Recognize(ctx, img, paddleocr.Options{})
// page.Lines[i].Text, .Confidence, .DetectionScore, .Box, .Order, .Flipped
// page.Rotation, page.Models (record them as evidence), page.Timings
```

`Options{Charset: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"}` restricts the
recognizer to an alphabet. Boxes are in the coordinates of the image as given.

## Symptoms to recognise

- **Boot fails opening `<id>` with "no such file".** The models directory is
  not filled, or `ModelsDir` points elsewhere. Run the command with `-check`.
- **`model digest mismatch`.** The file is not the reviewed one. Fetch it again
  with the command; never edit a digest.
- **`ONNX Runtime library unavailable`.** See `arandu-onnx`.
- **`the charset has …, which the recognizer's dictionary does not`.** A
  character the configured recognizer cannot output; the `en` recognizer has
  no accents.
- **Few or no lines on a photograph.** The detector sees at most
  `LimitSideLen` pixels on the longer side; small text on a large photo may
  need a crop or a larger limit.
