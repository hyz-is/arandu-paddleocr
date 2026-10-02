package paddleocr

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"fmt"
	"io/fs"
	"path"
	"sort"
	"strings"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// The identifiers of the models this package ships manifests for.
const (
	// ModelMobileDetector is PP-OCRv5_mobile_det, the default text detector.
	ModelMobileDetector = "pp-ocrv5_mobile_det"
	// ModelServerDetector is PP-OCRv5_server_det, larger and slower than the
	// mobile detector.
	ModelServerDetector = "pp-ocrv5_server_det"
	// ModelLatinRecognizer is latin_PP-OCRv5_mobile_rec, the default
	// recognizer: Latin script with accents, 838 classes.
	ModelLatinRecognizer = "latin_pp-ocrv5_mobile_rec"
	// ModelEnglishRecognizer is en_PP-OCRv5_mobile_rec: unaccented Latin, 438
	// classes.
	ModelEnglishRecognizer = "en_pp-ocrv5_mobile_rec"
	// ModelTextlineOrientation is PP-LCNet_x0_25_textline_ori, which tells a
	// text line at 0 degrees from one at 180.
	ModelTextlineOrientation = "pp-lcnet_x0_25_textline_ori"
	// ModelDocOrientation is PP-LCNet_x1_0_doc_ori, which tells a page at 0,
	// 90, 180 or 270 degrees.
	ModelDocOrientation = "pp-lcnet_x1_0_doc_ori"
)

//go:embed manifests/*.json
var manifestFiles embed.FS

//go:embed dictionaries/*.txt
var dictionaryFiles embed.FS

// Manifests returns the manifest of every model this package can run, sorted
// by identifier. Each one records where its file was obtained, its SHA-256
// and its licences, and Runtime.Open verifies the file against it.
func Manifests() ([]onnx.Manifest, error) {
	entries, err := fs.ReadDir(manifestFiles, "manifests")
	if err != nil {
		return nil, fmt.Errorf("paddleocr: listing the embedded manifests: %w", err)
	}
	out := make([]onnx.Manifest, 0, len(entries))
	for _, entry := range entries {
		data, err := manifestFiles.ReadFile(path.Join("manifests", entry.Name()))
		if err != nil {
			return nil, fmt.Errorf("paddleocr: reading the embedded manifest %s: %w", entry.Name(), err)
		}
		m, err := onnx.LoadManifest(data)
		if err != nil {
			return nil, fmt.Errorf("paddleocr: embedded manifest %s: %w", entry.Name(), err)
		}
		if want := strings.TrimSuffix(entry.Name(), ".json"); m.ID != want {
			return nil, fmt.Errorf("paddleocr: embedded manifest %s carries the id %q", entry.Name(), m.ID)
		}
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

// LookupManifest returns the manifest of one model by its identifier.
func LookupManifest(id string) (onnx.Manifest, error) {
	all, err := Manifests()
	if err != nil {
		return onnx.Manifest{}, err
	}
	for _, m := range all {
		if m.ID == id {
			return m, nil
		}
	}
	return onnx.Manifest{}, fmt.Errorf("paddleocr: no model is called %q", id)
}

// dictionary is a recognizer's character list, embedded in this package with
// the digest of the text it was extracted from.
type dictionary struct {
	file   string
	sha256 string
	lines  int
}

// dictionaries are the character lists of the recognizers, as the
// character_dict of each model's inference.yml lists them, one per line.
var dictionaries = map[string]dictionary{
	ModelLatinRecognizer: {
		file:   "dictionaries/latin_ppocrv5_dict.txt",
		sha256: "ccbcc45730b3fbbd9050c5bc74db6a99067141ef1035e3d14889a84a6b9b1aff",
		lines:  836,
	},
	ModelEnglishRecognizer: {
		file:   "dictionaries/en_ppocrv5_dict.txt",
		sha256: "e025a66d31f327ba0c232e03f407ae8d105e1e709e7ccb3f408aa778c24e70d6",
		lines:  436,
	},
}

// RecognizerVocabulary returns the vocabulary of a recognizer model: its
// embedded dictionary, checked against the digest it was recorded with, and
// the space class every PP-OCRv5 recognizer is trained with.
func RecognizerVocabulary(id string) (Vocabulary, error) {
	d, ok := dictionaries[id]
	if !ok {
		return Vocabulary{}, fmt.Errorf("paddleocr: %q is not a recognizer this package has a dictionary for", id)
	}
	data, err := dictionaryFiles.ReadFile(d.file)
	if err != nil {
		return Vocabulary{}, fmt.Errorf("paddleocr: reading the dictionary of %s: %w", id, err)
	}
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != d.sha256 {
		return Vocabulary{}, fmt.Errorf("paddleocr: the dictionary of %s has sha256 %s and %s was recorded", id, got, d.sha256)
	}
	lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
	if len(lines) != d.lines {
		return Vocabulary{}, fmt.Errorf("paddleocr: the dictionary of %s has %d lines and %d were recorded", id, len(lines), d.lines)
	}
	return NewVocabulary(lines, true), nil
}
