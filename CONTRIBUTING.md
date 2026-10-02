# Contributing

Thank you for considering a contribution.

## Before you open a pull request

Run the four gates. They are what CI runs, and a pull request that fails one of
them fails for everybody who pulls after it.

```bash
export GOWORK=off CGO_ENABLED=1
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

The two filters on `gofmt` are not preference. `gofmt` is the only tool in the
chain that ignores build tags, so it reads files the compiler deliberately
excludes; `testdata/` holds fixtures that are invalid on purpose.

The pipeline tests need the ONNX Runtime shared library and the model files,
and skip, saying why, where either is absent. Fetch the models once with
`go run ./cmd/paddleocr-models -dir <dir>` and point `KUAA_MODELS_DIR` at the
directory. A pull request that touches the pipeline is run with both present.

## What a change has to keep true

Five properties are the reason this package is what it is. A change that breaks
one of them will not be merged, whatever else it improves.

1. **Text, never decisions.** No verdict, no field validation, no document
   semantics. The application interprets the text and records the `Models`
   beside it.
2. **No model without its manifest.** Every model is opened through its
   embedded manifest, pinned to a revision and a digest. No path around
   `Runtime.Open` is added.
3. **The library does not touch the network.** Downloading lives in
   `cmd/paddleocr-models` and nowhere else, and no test downloads.
4. **No Python.** Not at runtime, not in a gate, not to generate a fixture.
5. **`arandu.mod.toml` matches the code.** Adding an outbound call, a file
   write or a process means declaring it there in the same commit.

## Style

Everything in the source is in English: identifiers, doc comments, internal
comments, error messages, log messages, and the names and messages of tests.
`pkg.go.dev` publishes the doc comments, and its readers are users of this
package.

Every exported symbol carries a doc comment saying what it does. The comment
documents the symbol and nothing else: no dates, no issue numbers, no
references to other repositories. Why a signature is what it is belongs there
when it is a fact about the code; anything else belongs in the pull request.

## Commits

One commit per logical change. Formatting is not mixed with behaviour, because
a diff that does both is a diff nobody reviews.

## Reporting a vulnerability

Do not open an issue. See [SECURITY.md](SECURITY.md).

## Licence

By contributing you agree that your contribution is licensed under the MIT
licence, the same as the rest of github.com/kuaa-io/arandu-paddleocr.
