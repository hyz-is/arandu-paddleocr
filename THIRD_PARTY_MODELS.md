# Third-party models

The model weights this package runs are **not** in this repository and are not
distributed with it. `cmd/paddleocr-models` downloads each file from the
upstream revision its manifest pins and keeps it only when its SHA-256 matches.
Whoever downloads and deploys them accepts their licence; the table below is
what was read before each was marked `approved`, and is not legal advice.

| id | upstream repository | pinned revision | sha256 | code licence | model licence | commercial use | redistribution | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pp-ocrv5_mobile_det` | [PaddlePaddle/PP-OCRv5_mobile_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx) | `e6f4fa85f00e168c862bc462aebca69eef9b3d3d` | `a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d` | Apache-2.0 | Apache-2.0 | yes | allowed, with licence and notice | approved |
| `pp-ocrv5_server_det` | [PaddlePaddle/PP-OCRv5_server_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det_onnx) | `dcf248c4dcc064c6d030db241255f9e8fbc6733a` | `10803475a591f7dc623e24670fb5752ec94d39a1f8cf069aac1b6f0ce19cfc85` | Apache-2.0 | Apache-2.0 | yes | allowed, with licence and notice | approved |
| `latin_pp-ocrv5_mobile_rec` | [PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx) | `89d3a50e2c27e2e7cceeab0e944c25c807d5db4f` | `7888113072263cb471b93f66dd5e2ad70548dc526fa1ace760d0d973dd121498` | Apache-2.0 | Apache-2.0 | yes | allowed, with licence and notice | approved |
| `en_pp-ocrv5_mobile_rec` | [PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx) | `3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b` | `b5f833dfc5d0eb71da397b4efa06ebeee9b431b690a47d6af40d77d8eabc557f` | Apache-2.0 | Apache-2.0 | yes | allowed, with licence and notice | approved |
| `pp-lcnet_x0_25_textline_ori` | [PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx](https://huggingface.co/PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx) | `aea1f18d97338aaf84b463f90192f2820524a00a` | `94a6a0a0425f2b5f08b5df72086f2d72abe40f1d22f6d12d2cd83674f11f2ff3` | Apache-2.0 | Apache-2.0 | yes | allowed, with licence and notice | approved |
| `pp-lcnet_x1_0_doc_ori` | [PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx) | `7330ab7039123e46af2dc03154b9969aa412c61d` | `af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92` | Apache-2.0 | Apache-2.0 | yes | allowed, with licence and notice | approved |

Each file is `inference.onnx` at the root of its repository, downloaded from
`https://huggingface.co/<repository>/resolve/<revision>/inference.onnx` and
stored as `<repository name>/inference.onnx` under the models directory.

## Training data

The upstream repositories do not enumerate the training data of these models;
the PaddleOCR documentation describes PP-OCRv5. This package has measured them
only on synthetic pages drawn in Go (see the README). Their accuracy on real
identity documents, and any difference across document types, print quality or
capture devices, has not been evaluated here and needs a consented,
representative evaluation set before it is relied on.

## Changing a model

A new model or a new revision is a new manifest under `manifests/`, with the
revision pinned in `source_url`, the digest of the exact file, and the licence
fields filled from what the upstream repository says. It starts as `dev-only`
until its licence has been reviewed, and moving it to `approved` is a
changelog line.
