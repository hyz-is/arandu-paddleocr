# Skills

Procedures an assistant follows when working on this package.

They live in `.agents/skills/<name>/SKILL.md`, which is the path the coding
assistants read from. It is one directory rather than a file per vendor, so a
skill written once is read by whatever this package is being written with.

Each file opens with frontmatter carrying a `name` and a `description`. The
`name` has to equal the directory name exactly, or the skill is not loaded. The
description is what a tool reads to decide whether the skill is relevant, so it
names the situation you are in rather than the topic it covers.

| skill | when it fires |
| --- | --- |
| `paddleocr-module` | changing the pipeline, the post-processing, the decoding, a manifest or the download command |
| `paddleocr-release` | the gates, the manifest, a dependency, a version, a tag |
| `paddleocr-vault-notes` | writing the note, the gap or the journal entry, when this checkout sits inside the KUAA vault |
| `paddleocr-package` | installing and wiring this package **into an application** |

The last one has a different audience from the other three, and that is on
purpose: it travels with the package so that an assistant working in somebody
else's project has the build requirement, the models, the wiring and the result
shape in front of it instead of guessing.

## Why these exist

The common failure modes are a field parser or a verdict creeping into the
engine, a model loaded without its manifest, a download added to the library or
to a test, a character decoded by looking it up in the dictionary rather than
by its class index, and a Python reference script to "check" a value. None
belongs here.
