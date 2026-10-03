package feature_test

import (
	"context"
	"errors"
	"image"
	"image/draw"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"

	onnx "github.com/hyz-is/arandu-onnx"
	paddleocr "github.com/hyz-is/arandu-paddleocr"
)

// These helpers load the real models and draw the pages the tests read.
//
// The models are not in the repository. The tests look for them in the
// directory KUAA_MODELS_DIR names, or in the KUAA workspace's model cache two
// levels above this checkout, and skip, saying where they looked, when neither
// has them. The pages are drawn with the Go fonts (BSD licence) and contain
// only invented text.

// modelsDir is the directory holding the model files, or a skip.
func modelsDir(t testing.TB) string {
	t.Helper()
	dir := os.Getenv("KUAA_MODELS_DIR")
	if dir == "" {
		dir = filepath.Join(packageRoot(t), "..", "..", "models", "ocr_official")
	}
	manifest, err := paddleocr.LookupManifest(paddleocr.ModelMobileDetector)
	if err != nil {
		t.Fatalf("looking up the detector's manifest: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, manifest.File)); err != nil {
		t.Skipf("the PP-OCRv5 models are not in %s (set KUAA_MODELS_DIR, or fill it with go run ./cmd/paddleocr-models -dir %s): %v", dir, dir, err)
	}
	return dir
}

// newRuntime returns an initialized ONNX runtime, or a skip when the library
// is not on this machine.
func newRuntime(t testing.TB) *onnx.Runtime {
	t.Helper()
	rt, err := onnx.NewRuntime(onnx.Config{})
	if err != nil {
		t.Fatalf("building the runtime: %v", err)
	}
	if err := rt.Init(); err != nil {
		if errors.Is(err, onnx.ErrLibraryUnavailable) {
			t.Skipf("ONNX Runtime is not available here, so nothing that runs a model can be tested: %v", err)
		}
		t.Fatalf("initialising the runtime: %v", err)
	}
	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Errorf("closing the runtime: %v", err)
		}
	})
	return rt
}

// bootModule builds and boots the module on the cached models.
func bootModule(t testing.TB, cfg paddleocr.Config) *paddleocr.Module {
	t.Helper()
	cfg.ModelsDir = modelsDir(t)
	rt := newRuntime(t)
	m, err := paddleocr.New(cfg, rt)
	if err != nil {
		t.Fatalf("building the module: %v", err)
	}
	if err := m.Boot(context.Background()); err != nil {
		t.Fatalf("booting the module: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(context.Background()); err != nil {
			t.Errorf("closing the module: %v", err)
		}
	})
	return m
}

// drawPage writes the lines in black on a white page with Go Regular at size
// pixels, one under the other, and returns the page and each line's bounding
// rectangle.
func drawPage(t testing.TB, lines []string, size float64) (*image.RGBA, []image.Rectangle) {
	t.Helper()
	parsed, err := opentype.Parse(goregular.TTF)
	if err != nil {
		t.Fatalf("parsing Go Regular: %v", err)
	}
	face, err := opentype.NewFace(parsed, &opentype.FaceOptions{Size: size, DPI: 72, Hinting: font.HintingFull})
	if err != nil {
		t.Fatalf("building the face: %v", err)
	}
	defer face.Close()

	margin := int(size * 1.5)
	step := int(size * 2.2)
	width := 0
	for _, line := range lines {
		width = max(width, font.MeasureString(face, line).Ceil())
	}
	page := image.NewRGBA(image.Rect(0, 0, width+2*margin, margin*2+step*len(lines)))
	draw.Draw(page, page.Bounds(), image.White, image.Point{}, draw.Src)

	ascent := face.Metrics().Ascent.Ceil()
	boxes := make([]image.Rectangle, len(lines))
	for i, line := range lines {
		baseline := margin + i*step + ascent
		d := &font.Drawer{Dst: page, Src: image.Black, Face: face, Dot: fixed.P(margin, baseline)}
		bounds, _ := d.BoundString(line)
		d.DrawString(line)
		boxes[i] = image.Rect(bounds.Min.X.Floor(), bounds.Min.Y.Floor(), bounds.Max.X.Ceil(), bounds.Max.Y.Ceil())
	}
	return page, boxes
}

// tilt turns an image by degrees counter-clockwise about its centre on a
// canvas large enough to hold it, filling the corners with white.
func tilt(src *image.RGBA, degrees float64) *image.RGBA {
	theta := degrees * math.Pi / 180
	c, s := math.Cos(theta), math.Sin(theta)
	w, h := float64(src.Rect.Dx()), float64(src.Rect.Dy())
	nw := int(math.Ceil(math.Abs(w*c) + math.Abs(h*s)))
	nh := int(math.Ceil(math.Abs(w*s) + math.Abs(h*c)))
	out := image.NewRGBA(image.Rect(0, 0, nw, nh))
	draw.Draw(out, out.Bounds(), image.White, image.Point{}, draw.Src)
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			// Counter-clockwise on screen, where Y points down.
			dx, dy := float64(x)-float64(nw)/2, float64(y)-float64(nh)/2
			sx := c*dx - s*dy + w/2
			sy := s*dx + c*dy + h/2
			if sx < 0 || sy < 0 || sx > w-1 || sy > h-1 {
				continue
			}
			x0, y0 := int(sx), int(sy)
			x1, y1 := min(x0+1, int(w)-1), min(y0+1, int(h)-1)
			fx, fy := sx-float64(x0), sy-float64(y0)
			for ch := 0; ch < 4; ch++ {
				at := func(px, py int) float64 { return float64(src.Pix[py*src.Stride+px*4+ch]) }
				top := at(x0, y0) + (at(x1, y0)-at(x0, y0))*fx
				bottom := at(x0, y1) + (at(x1, y1)-at(x0, y1))*fx
				out.Pix[y*out.Stride+x*4+ch] = uint8(math.Round(top + (bottom-top)*fy))
			}
		}
	}
	return out
}

// turn rotates an image counter-clockwise by a multiple of 90 degrees.
func turn(t testing.TB, src *image.RGBA, degrees int) *image.RGBA {
	t.Helper()
	out, err := paddleocr.Rotate(src, degrees)
	if err != nil {
		t.Fatalf("rotating by %d: %v", degrees, err)
	}
	return out
}

// editDistance is the Levenshtein distance between two strings, by rune.
func editDistance(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	prev := make([]int, len(rb)+1)
	cur := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev, cur = cur, prev
	}
	return prev[len(rb)]
}

// ascii folds the accented capitals and small letters of Portuguese onto
// their base letters, so that a comparison does not hinge on an accent.
func ascii(s string) string {
	return strings.NewReplacer(
		"Á", "A", "À", "A", "Â", "A", "Ã", "A", "É", "E", "Ê", "E", "Í", "I",
		"Ó", "O", "Ô", "O", "Õ", "O", "Ú", "U", "Ç", "C",
		"á", "a", "à", "a", "â", "a", "ã", "a", "é", "e", "ê", "e", "í", "i",
		"ó", "o", "ô", "o", "õ", "o", "ú", "u", "ç", "c",
	).Replace(s)
}

func packageRoot(t testing.TB) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("finding package root")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}
