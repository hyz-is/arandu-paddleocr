# Changelog

Everything worth knowing about a release of Arandu PaddleOCR is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

### Added

- `Module`: `foundation.Module` with no route, plus `Bootable` (opens the
  configured models, verified against their manifests), `Health` and
  `Closable`. `New(Config, *onnx.Runtime)`.
- `Module.Recognize`: page orientation (optional), PP-OCRv5 detection with DB
  post-processing, perspective crop, line orientation (optional) and
  recognition with CTC greedy decoding, returning `Page{Width, Height,
  Rotation, Lines, Models, Timings}`. `Options.Charset` narrows the classes the
  recognizer chooses among.
- `Config`: `ModelsDir`, `Detector` (`mobile`, `server`), `Recognizer`
  (`latin`, `en`), `UseDocOrientation`, `UseTextlineOrientation`,
  `LimitSideLen`, `Thresh`, `BoxThresh`, `UnclipRatio`, `MaxCandidates`,
  `RecBatchSize`, `RecMaxWidth`, `TextlineThreshold`, with `Validate` and
  `WithDefaults`.
- Manifests for PP-OCRv5_mobile_det, PP-OCRv5_server_det,
  latin_PP-OCRv5_mobile_rec, en_PP-OCRv5_mobile_rec,
  PP-LCNet_x0_25_textline_ori and PP-LCNet_x1_0_doc_ori, pinned to their
  Hugging Face revisions, and the recognizers' dictionaries. `Manifests`,
  `LookupManifest`, `RecognizerVocabulary`.
- Pure-Go stages: `DBPostProcess`, `BoxScore`, `ConvexHull`, `MinAreaRect`,
  `MiniBox`, `Unclip`, `UnclipDistance`, `OrderClockwise`, `ReadingOrder`,
  `PerspectiveTransform`, `WarpQuad`, `CropSize`, `Rotate`, `Vocabulary`,
  `DecodeCTC`, `DetectorInputSize`, `RecognizerInputWidth`.
- `cmd/paddleocr-models`: downloads the model files over HTTPS and verifies
  their digests; `-check` verifies without connecting.
