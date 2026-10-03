---
name: paddleocr-release
description: Get this Arandu package green, declared and published. Use when the request is to "run the gates", "check it builds", "the tests skip", "the tests fail", "go build says the directory prefix does not contain modules listed in go.work", "update arandu.mod.toml", "add a dependency", "bump the framework", "remove the replace", "tag a release", "publish it", "write the changelog", or before opening a pull request. Covers the four gates, the models the tests need, what the manifest declares and who checks it, the replace of arandu-onnx, and the immutability of a published module version.
license: MIT
---

# Getting it green and out

## The four gates

```sh
export GOWORK=off CGO_ENABLED=1
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

`GOWORK=off` because this checkout may sit beside a workspace that does not
list it. `CGO_ENABLED=1` because arandu-onnx's binding is cgo. The pipeline
tests need ONNX Runtime and the models (from `PADDLEOCR_MODELS_DIR`, or
`../../models/ocr_official`); they skip, saying why, without either. A
run where they skipped has not tested the pipeline: read the skip lines before
calling it green.

`aru doctor` is not a gate: it reads applications, and this is a library.

## What the manifest declares, and who reads it

```toml
name = "hyz-is/paddleocr"
framework = ">= 0.50"
profiles = ["conventional", "performance"]

[permissions]
network = false
filesystem = true
exec = false
migrations = false
```

`tests/Unit/audit_test.go` compares it with what the code calls. `filesystem`
is the model files, read through arandu-onnx's `Runtime.Open`, which the audit
counts as a read. `network = false` holds because `cmd/paddleocr-models`, the
download command, is a `main` package no application links;
`TestTheDownloadCommandIsWhereTheNetworkIs` holds that it is the only place a
network call is written.

The `framework` line is a floor, and `go.mod` says it too. Move both together.

## The replace

`go.mod` replaces `github.com/hyz-is/arandu-onnx` with `../arandu-onnx` while
that module is unpublished. Before the first tag here: tag arandu-onnx, require
it at that version, delete the replace and its comment, and run the gates from
a checkout without the sibling. The release workflow refuses a local replace.

## Dependencies

The framework, arandu-onnx, and `golang.org/x/image` for the tests' fonts. A new
`require` needs an argument in the pull request and, if it reaches the network,
the filesystem or another process, a manifest change in the same commit.

## Publishing

A published Go module version is **immutable**. Before tagging: the gates pass
with the pipeline tests run rather than skipped, `arandu.mod.toml` matches the
code, `CHANGELOG.md` has the entry under the version, and `UPGRADE.md` names
every incompatible change. A changed default (`LimitSideLen`, `BoxThresh`, the
detector) changes every application that never set it: it is a changelog line.
A manifest's digest or revision changing is a model changing, and is one too.
