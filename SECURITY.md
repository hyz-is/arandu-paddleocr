# Security policy

## Supported versions

The latest minor release of Arandu PaddleOCR receives security fixes. Older
minor releases do not: a published Go module version is immutable, so a fix is
a new release rather than a moved tag.

## Reporting a vulnerability

Report it privately, through GitHub's advisory form:

<https://github.com/kuaa-io/arandu-paddleocr/security/advisories/new>

Do not open a public issue and do not describe the problem in a pull request.
A report that arrives in public is a report every reader of this repository can
act on before there is a release to upgrade to.

Include what you need to reproduce it: the version, the configuration, and an
image that triggers it -- a synthetic one, never a real document.

Expect an acknowledgement within a few days. If the report is confirmed, the
fix, the release and the advisory are published together, and you are credited
unless you ask not to be.

## What is in scope

Anything that lets a model run that its manifest did not admit, or lets an
image cost more than it should. In particular:

- a way to open a model whose file does not match its manifest's SHA-256;
- `cmd/paddleocr-models` keeping a file whose digest does not match, following
  a redirect off HTTPS, or writing outside the directory it was given;
- an image that crashes the process, reads out of bounds, or costs memory or
  time out of proportion to its size;
- a race that lets a call run on a module that was closed.

## What is not

- A vulnerability in ONNX Runtime or in the PaddleOCR models themselves.
  Report it to those projects.
- What an application decides from the text. This package returns text.
