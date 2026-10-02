package unit_test

import (
	"net/url"
	"regexp"
	"strings"
	"testing"

	onnx "github.com/kuaa-io/arandu-onnx"
	paddleocr "github.com/kuaa-io/arandu-paddleocr"
)

func TestEveryShippedManifestRecordsItsProvenance(t *testing.T) {
	t.Parallel()

	manifests, err := paddleocr.Manifests()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		paddleocr.ModelEnglishRecognizer,
		paddleocr.ModelLatinRecognizer,
		paddleocr.ModelTextlineOrientation,
		paddleocr.ModelDocOrientation,
		paddleocr.ModelMobileDetector,
		paddleocr.ModelServerDetector,
	}
	if len(manifests) != len(want) {
		t.Fatalf("%d manifests, want %d", len(manifests), len(want))
	}
	revision := regexp.MustCompile(`^https://huggingface\.co/PaddlePaddle/([A-Za-z0-9_.-]+)/resolve/([0-9a-f]{40})/inference\.onnx$`)
	files := map[string]bool{}
	for i, m := range manifests {
		if m.ID != want[i] {
			t.Errorf("manifest %d is %s, want %s", i, m.ID, want[i])
		}
		if m.Status != onnx.StatusApproved || m.CommercialUse != onnx.CommercialUseYes || m.Precision != onnx.PrecisionFP32 {
			t.Errorf("%s: status %s, commercial use %s, precision %s; want approved, yes, fp32", m.ID, m.Status, m.CommercialUse, m.Precision)
		}
		if m.License != "Apache-2.0" || m.ModelLicense != "Apache-2.0" {
			t.Errorf("%s: licences %q and %q, want Apache-2.0", m.ID, m.License, m.ModelLicense)
		}
		// The source is pinned to a revision, and the version is that
		// revision's short form; the file sits under the repository's name.
		match := revision.FindStringSubmatch(m.SourceURL)
		if match == nil {
			t.Errorf("%s: source_url %s is not a pinned Hugging Face revision", m.ID, m.SourceURL)
			continue
		}
		if !strings.HasPrefix(match[2], m.Version) || len(m.Version) != 12 {
			t.Errorf("%s: version %s is not the 12-character prefix of revision %s", m.ID, m.Version, match[2])
		}
		if m.File != match[1]+"/inference.onnx" {
			t.Errorf("%s: file %s, want %s/inference.onnx", m.ID, m.File, match[1])
		}
		if !strings.EqualFold(strings.ReplaceAll(match[1], "_onnx", ""), m.ID) {
			t.Errorf("%s: comes from %s, whose name is not its id", m.ID, match[1])
		}
		if files[m.File] {
			t.Errorf("%s: file %s is shared with another manifest", m.ID, m.File)
		}
		files[m.File] = true
		if u, err := url.Parse(m.SourceURL); err != nil || u.Scheme != "https" {
			t.Errorf("%s: source_url is not https", m.ID)
		}
		if len(m.Inputs) != 1 || m.Inputs[0].Name != "x" || len(m.Outputs) != 1 || m.Outputs[0].Name != "fetch_name_0" {
			t.Errorf("%s: contract %+v -> %+v, want x -> fetch_name_0", m.ID, m.Inputs, m.Outputs)
		}
	}
}

func TestTheManifestsRecordTheReviewedDigests(t *testing.T) {
	t.Parallel()

	// The digests the model review recorded, which the files' Hugging Face
	// LFS object ids also are.
	for id, digest := range map[string]string{
		paddleocr.ModelMobileDetector:      "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d",
		paddleocr.ModelServerDetector:      "10803475a591f7dc623e24670fb5752ec94d39a1f8cf069aac1b6f0ce19cfc85",
		paddleocr.ModelLatinRecognizer:     "7888113072263cb471b93f66dd5e2ad70548dc526fa1ace760d0d973dd121498",
		paddleocr.ModelEnglishRecognizer:   "b5f833dfc5d0eb71da397b4efa06ebeee9b431b690a47d6af40d77d8eabc557f",
		paddleocr.ModelTextlineOrientation: "94a6a0a0425f2b5f08b5df72086f2d72abe40f1d22f6d12d2cd83674f11f2ff3",
		paddleocr.ModelDocOrientation:      "af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92",
	} {
		m, err := paddleocr.LookupManifest(id)
		if err != nil {
			t.Fatal(err)
		}
		if m.SHA256 != digest {
			t.Errorf("%s: sha256 %s, want %s", id, m.SHA256, digest)
		}
	}
	if _, err := paddleocr.LookupManifest("no-such-model"); err == nil {
		t.Error("an unknown id was found")
	}
}

func TestEachRecognizerOutputsAClassPerVocabularyEntry(t *testing.T) {
	t.Parallel()

	for _, id := range []string{paddleocr.ModelLatinRecognizer, paddleocr.ModelEnglishRecognizer} {
		m, err := paddleocr.LookupManifest(id)
		if err != nil {
			t.Fatal(err)
		}
		v, err := paddleocr.RecognizerVocabulary(id)
		if err != nil {
			t.Fatal(err)
		}
		if classes := m.Outputs[0].Shape[2]; int64(v.Classes()) != classes {
			t.Errorf("%s outputs %d classes and its vocabulary has %d", id, classes, v.Classes())
		}
	}
}
