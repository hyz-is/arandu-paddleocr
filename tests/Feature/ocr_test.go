package feature_test

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"strings"
	"sync"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	paddleocr "github.com/hyz-is/arandu-paddleocr"
)

// These tests read synthetic pages with the real PP-OCRv5 models. The pages
// are drawn here, with invented text in the Go fonts.

var documentLines = []string{
	"REPÚBLICA FEDERATIVA DO BRASIL",
	"NOME: MARIA DA SILVA TESTE",
	"CPF 123.456.789-09",
	"IDBRA123456789<<<<<<<<<<<<<<<",
}

// mrzCharset is the alphabet of a machine-readable zone.
const mrzCharset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

// readable reports whether what was read is close enough to what was drawn:
// at most tolerance edits apart once accents are folded away. The Go fonts
// draw accents the recognizer usually, but not always, reads.
func readable(got, want string, tolerance int) bool {
	return editDistance(ascii(got), ascii(want)) <= tolerance
}

// tolerance is the edit distance a line may be off by: one per run of
// repeated '<' (CTC can merge or split a long run of one character) and one
// more for everything else.
func tolerance(line string) int {
	if strings.Contains(line, "<<") {
		return 2
	}
	return 1
}

// assertLines holds a page to the lines drawn on it, in order.
func assertLines(t *testing.T, label string, page paddleocr.Page, want []string) {
	t.Helper()
	if len(page.Lines) != len(want) {
		texts := []string{}
		for _, l := range page.Lines {
			texts = append(texts, l.Text)
		}
		t.Fatalf("%s: found %d lines %q, want %d", label, len(page.Lines), texts, len(want))
	}
	for i, line := range page.Lines {
		if line.Order != i {
			t.Errorf("%s: line %d has Order %d", label, i, line.Order)
		}
		if !readable(line.Text, want[i], tolerance(want[i])) {
			t.Errorf("%s: line %d read %q, want %q", label, i, line.Text, want[i])
		}
		if line.Confidence < 0.5 || line.Confidence > 1 {
			t.Errorf("%s: line %d %q has confidence %v", label, i, line.Text, line.Confidence)
		}
		if line.DetectionScore < 0.6 || line.DetectionScore > 1 {
			t.Errorf("%s: line %d has a detection score of %v, under the box threshold that kept it", label, i, line.DetectionScore)
		}
	}
}

// centre is the mean of a quadrilateral's corners.
func centre(q paddleocr.Quad) paddleocr.Point {
	var c paddleocr.Point
	for _, p := range q {
		c.X += p.X / 4
		c.Y += p.Y / 4
	}
	return c
}

func TestAPageOfDocumentLinesIsReadInOrder(t *testing.T) {
	m := bootModule(t, paddleocr.Config{})
	page, drawn := drawPage(t, documentLines, 40)

	got, err := m.Recognize(context.Background(), page, paddleocr.Options{})
	if err != nil {
		t.Fatalf("recognizing: %v", err)
	}
	assertLines(t, "upright page", got, documentLines)
	if got.Width != page.Rect.Dx() || got.Height != page.Rect.Dy() || got.Rotation != 0 {
		t.Errorf("page is %d×%d turned %d, want %d×%d turned 0", got.Width, got.Height, got.Rotation, page.Rect.Dx(), page.Rect.Dy())
	}
	// Every exact line except the accented one and the MRZ is read exactly.
	for _, i := range []int{1, 2} {
		if got.Lines[i].Text != documentLines[i] {
			t.Errorf("line %d read %q, want exactly %q", i, got.Lines[i].Text, documentLines[i])
		}
	}
	// Each box encloses the ink it was drawn with, with the margin the
	// detector grows it by, and not much more.
	for i, line := range got.Lines {
		ink := drawn[i]
		for _, p := range line.Box {
			if p.X < float64(ink.Min.X)-15 || p.X > float64(ink.Max.X)+15 || p.Y < float64(ink.Min.Y)-20 || p.Y > float64(ink.Max.Y)+20 {
				t.Errorf("line %d box %v strays from the ink at %v", i, line.Box, ink)
				break
			}
		}
		c := centre(line.Box)
		if !(image.Point{X: int(c.X), Y: int(c.Y)}).In(ink.Inset(-2)) {
			t.Errorf("line %d box centre %v is outside the ink at %v", i, c, ink)
		}
	}
}

func TestEveryPageCarriesTheModelsThatReadIt(t *testing.T) {
	m := bootModule(t, paddleocr.Config{UseDocOrientation: true, UseTextlineOrientation: true})
	page, _ := drawPage(t, documentLines[:1], 40)

	got, err := m.Recognize(context.Background(), page, paddleocr.Options{})
	if err != nil {
		t.Fatalf("recognizing: %v", err)
	}
	want := []string{paddleocr.ModelDocOrientation, paddleocr.ModelMobileDetector, paddleocr.ModelTextlineOrientation, paddleocr.ModelLatinRecognizer}
	if len(got.Models) != len(want) {
		t.Fatalf("Models = %+v, want %v", got.Models, want)
	}
	for i, info := range got.Models {
		manifest, err := paddleocr.LookupManifest(want[i])
		if err != nil {
			t.Fatal(err)
		}
		if info.ID != want[i] || info.SHA256 != manifest.SHA256 || info.Version != manifest.Version ||
			info.Status != onnx.StatusApproved || info.Precision != onnx.PrecisionFP32 || !strings.HasPrefix(info.Runtime, "onnxruntime ") {
			t.Errorf("Models[%d] = %+v, want the provenance of %s %s", i, info, manifest.ID, manifest.Version)
		}
	}
	if got.Timings.Total <= 0 || got.Timings.Detection <= 0 || got.Timings.Recognition <= 0 ||
		got.Timings.DocOrientation <= 0 || got.Timings.TextlineOrientation <= 0 {
		t.Errorf("Timings = %+v, want every stage that ran timed", got.Timings)
	}
}

func TestABlankPageHasNoLinesAndNoRecognizer(t *testing.T) {
	m := bootModule(t, paddleocr.Config{})
	blank := image.NewRGBA(image.Rect(0, 0, 300, 200))
	for i := range blank.Pix {
		blank.Pix[i] = 255
	}
	got, err := m.Recognize(context.Background(), blank, paddleocr.Options{})
	if err != nil {
		t.Fatalf("recognizing: %v", err)
	}
	if len(got.Lines) != 0 {
		t.Errorf("a blank page has %d lines: %+v", len(got.Lines), got.Lines)
	}
	if len(got.Models) != 1 || got.Models[0].ID != paddleocr.ModelMobileDetector {
		t.Errorf("Models = %+v, want the detector alone: nothing was recognized", got.Models)
	}
}

func TestASlightlyTiltedPageIsRead(t *testing.T) {
	m := bootModule(t, paddleocr.Config{})
	page, _ := drawPage(t, documentLines, 40)
	for _, degrees := range []float64{-4, 3} {
		got, err := m.Recognize(context.Background(), tilt(page, degrees), paddleocr.Options{})
		if err != nil {
			t.Fatalf("recognizing at %v degrees: %v", degrees, err)
		}
		assertLines(t, fmt.Sprintf("tilted %v degrees", degrees), got, documentLines)
		// The boxes follow the tilt: the top side of each line slopes by the
		// angle the page was turned, counter-clockwise being up on screen.
		for i, line := range got.Lines {
			slope := math.Atan2(-(line.Box[1].Y-line.Box[0].Y), line.Box[1].X-line.Box[0].X) * 180 / math.Pi
			if math.Abs(slope-degrees) > 1.5 {
				t.Errorf("tilted %v degrees: line %d top side slopes %.2f degrees", degrees, i, slope)
			}
		}
	}
}

func TestATurnedPageIsStoodUprightByDocumentOrientation(t *testing.T) {
	m := bootModule(t, paddleocr.Config{UseDocOrientation: true})
	page, drawn := drawPage(t, documentLines, 40)
	w, h := page.Rect.Dx(), page.Rect.Dy()

	for _, turned := range []int{90, 180, 270} {
		got, err := m.Recognize(context.Background(), turn(t, page, turned), paddleocr.Options{})
		if err != nil {
			t.Fatalf("recognizing a page turned %d: %v", turned, err)
		}
		label := fmt.Sprintf("turned %d", turned)
		if want := (360 - turned) % 360; got.Rotation != want {
			t.Errorf("%s: Rotation = %d, want %d", label, got.Rotation, want)
		}
		assertLines(t, label, got, documentLines)

		// The boxes are on the turned image as given: the centre of each is
		// where the centre of the ink went when the page was turned.
		for i, line := range got.Lines {
			ink := drawn[i]
			cx, cy := float64(ink.Min.X+ink.Max.X)/2, float64(ink.Min.Y+ink.Max.Y)/2
			var want paddleocr.Point
			switch turned {
			case 90:
				want = paddleocr.Point{X: cy, Y: float64(w-1) - cx}
			case 180:
				want = paddleocr.Point{X: float64(w-1) - cx, Y: float64(h-1) - cy}
			case 270:
				want = paddleocr.Point{X: float64(h-1) - cy, Y: cx}
			}
			if c := centre(line.Box); math.Hypot(c.X-want.X, c.Y-want.Y) > 12 {
				t.Errorf("%s: line %d box centre %v, want near %v", label, i, c, want)
			}
		}
	}
}

func TestUpsideDownLinesAreTurnedByLineOrientation(t *testing.T) {
	m := bootModule(t, paddleocr.Config{UseTextlineOrientation: true})
	page, _ := drawPage(t, documentLines, 40)

	got, err := m.Recognize(context.Background(), turn(t, page, 180), paddleocr.Options{})
	if err != nil {
		t.Fatalf("recognizing: %v", err)
	}
	// Without document orientation the page is read as it lies, so the last
	// line drawn is the first one met.
	reversed := make([]string, len(documentLines))
	for i, line := range documentLines {
		reversed[len(documentLines)-1-i] = line
	}
	assertLines(t, "upside down", got, reversed)
	for i, line := range got.Lines {
		if !line.Flipped {
			t.Errorf("line %d %q was not flipped", i, line.Text)
		}
	}

	upright, err := m.Recognize(context.Background(), page, paddleocr.Options{})
	if err != nil {
		t.Fatalf("recognizing: %v", err)
	}
	for i, line := range upright.Lines {
		if line.Flipped {
			t.Errorf("upright line %d %q was flipped", i, line.Text)
		}
	}
}

func TestACharsetRestrictsWhatIsRead(t *testing.T) {
	m := bootModule(t, paddleocr.Config{})
	page, _ := drawPage(t, []string{"P<BRATESTE<<MARIA<DA<SILVA<<<<<<<<<<<<<<<<<<", "IDBRA123456789<<<<<<<<<<<<<<<"}, 40)

	got, err := m.Recognize(context.Background(), page, paddleocr.Options{Charset: mrzCharset})
	if err != nil {
		t.Fatalf("recognizing: %v", err)
	}
	assertLines(t, "machine-readable zone", got, []string{"P<BRATESTE<<MARIA<DA<SILVA<<<<<<<<<<<<<<<<<<", "IDBRA123456789<<<<<<<<<<<<<<<"})
	for _, line := range got.Lines {
		for _, r := range line.Text {
			if !strings.ContainsRune(mrzCharset, r) {
				t.Errorf("%q has %q, outside the charset", line.Text, r)
			}
		}
	}

	// The same page read without the charset has the same text; the mask
	// only narrows the choice.
	free, err := m.Recognize(context.Background(), page, paddleocr.Options{})
	if err != nil {
		t.Fatalf("recognizing: %v", err)
	}
	assertLines(t, "unrestricted", free, []string{"P<BRATESTE<<MARIA<DA<SILVA<<<<<<<<<<<<<<<<<<", "IDBRA123456789<<<<<<<<<<<<<<<"})

	// A prose line read through the charset loses what the charset lacks.
	prose, _ := drawPage(t, []string{"NOME: MARIA"}, 40)
	restricted, err := m.Recognize(context.Background(), prose, paddleocr.Options{Charset: mrzCharset})
	if err != nil {
		t.Fatalf("recognizing: %v", err)
	}
	if len(restricted.Lines) != 1 || restricted.Lines[0].Text != "NOMEMARIA" {
		t.Errorf("read %+v through the charset, want the single line NOMEMARIA", restricted.Lines)
	}

	if _, err := m.Recognize(context.Background(), page, paddleocr.Options{Charset: "ABC中"}); err == nil {
		t.Error("a charset with a character the dictionary lacks was accepted")
	}
}

func TestTheOtherDetectorAndRecognizerRead(t *testing.T) {
	page, _ := drawPage(t, documentLines, 40)
	for _, cfg := range []paddleocr.Config{{Detector: paddleocr.DetectorServer}, {Recognizer: paddleocr.RecognizerEnglish}} {
		m := bootModule(t, cfg)
		got, err := m.Recognize(context.Background(), page, paddleocr.Options{})
		if err != nil {
			t.Fatalf("recognizing with %+v: %v", cfg, err)
		}
		assertLines(t, fmt.Sprintf("detector %q recognizer %q", cfg.Detector, cfg.Recognizer), got, documentLines)
	}
}

// TestTheSyntheticSetIsReadAccurately reads every document line at three
// sizes and three tilts and measures the character error rate. It is the
// accuracy figure the README reports.
func TestTheSyntheticSetIsReadAccurately(t *testing.T) {
	m := bootModule(t, paddleocr.Config{})
	lines, exact, edits, chars := 0, 0, 0, 0
	for _, size := range []float64{32, 40, 48} {
		page, _ := drawPage(t, documentLines, size)
		for _, degrees := range []float64{0, -2, 2} {
			got, err := m.Recognize(context.Background(), tilt(page, degrees), paddleocr.Options{})
			if err != nil {
				t.Fatalf("recognizing: %v", err)
			}
			assertLines(t, fmt.Sprintf("size %v tilt %v", size, degrees), got, documentLines)
			for i, line := range got.Lines {
				lines++
				d := editDistance(line.Text, documentLines[i])
				if d == 0 {
					exact++
				} else {
					t.Logf("size %v tilt %v: read %q for %q, %d edits", size, degrees, line.Text, documentLines[i], d)
				}
				edits += d
				chars += len([]rune(documentLines[i]))
			}
		}
	}
	cer := float64(edits) / float64(chars)
	t.Logf("synthetic set: %d lines, %d read exactly, %d edits over %d characters, character error rate %.4f", lines, exact, edits, chars, cer)
	if cer > 0.02 {
		t.Errorf("character error rate %.4f, want at most 0.02", cer)
	}
}

func TestRecognizeIsSafeForConcurrentUse(t *testing.T) {
	m := bootModule(t, paddleocr.Config{UseTextlineOrientation: true})
	page, _ := drawPage(t, documentLines, 40)
	want, err := m.Recognize(context.Background(), page, paddleocr.Options{})
	if err != nil {
		t.Fatalf("recognizing: %v", err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 2 {
				got, err := m.Recognize(context.Background(), page, paddleocr.Options{})
				if err != nil {
					errs <- err
					return
				}
				if len(got.Lines) != len(want.Lines) {
					errs <- fmt.Errorf("a concurrent call found %d lines, want %d", len(got.Lines), len(want.Lines))
					return
				}
				for i := range got.Lines {
					if got.Lines[i].Text != want.Lines[i].Text || got.Lines[i].Box != want.Lines[i].Box {
						errs <- fmt.Errorf("a concurrent call read line %d as %q at %v, want %q at %v",
							i, got.Lines[i].Text, got.Lines[i].Box, want.Lines[i].Text, want.Lines[i].Box)
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestACancelledContextStopsRecognition(t *testing.T) {
	m := bootModule(t, paddleocr.Config{})
	page, _ := drawPage(t, documentLines, 40)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Recognize(ctx, page, paddleocr.Options{}); !errors.Is(err, context.Canceled) {
		t.Errorf("Recognize with a cancelled context = %v, want context.Canceled", err)
	}
}

func TestAnEmptyImageIsRefused(t *testing.T) {
	m := bootModule(t, paddleocr.Config{})
	if _, err := m.Recognize(context.Background(), image.NewRGBA(image.Rect(0, 0, 0, 0)), paddleocr.Options{}); err == nil {
		t.Error("an empty image was accepted")
	}
	if _, err := m.Recognize(context.Background(), nil, paddleocr.Options{}); err == nil {
		t.Error("a nil image was accepted")
	}
}

func TestATinyImageIsReadWithoutFailing(t *testing.T) {
	m := bootModule(t, paddleocr.Config{})
	tiny := image.NewRGBA(image.Rect(0, 0, 3, 2))
	if _, err := m.Recognize(context.Background(), tiny, paddleocr.Options{}); err != nil {
		t.Errorf("a 3×2 image: %v", err)
	}
	// An image whose origin is not (0, 0) is read in its own coordinates
	// from (0, 0).
	page, _ := drawPage(t, documentLines[2:3], 40)
	shifted := page.SubImage(image.Rect(10, 10, page.Rect.Dx(), page.Rect.Dy()))
	got, err := m.Recognize(context.Background(), shifted, paddleocr.Options{})
	if err != nil {
		t.Fatalf("recognizing a sub-image: %v", err)
	}
	assertLines(t, "sub-image", got, documentLines[2:3])
	if got.Width != page.Rect.Dx()-10 {
		t.Errorf("Width = %d, want %d", got.Width, page.Rect.Dx()-10)
	}
}

// BenchmarkRecognize reads the four-line document page, 851×472, with each
// stage on, on the CPU.
func BenchmarkRecognize(b *testing.B) {
	for _, tc := range []struct {
		name string
		cfg  paddleocr.Config
	}{
		{"mobile", paddleocr.Config{}},
		{"mobile+orientation", paddleocr.Config{UseDocOrientation: true, UseTextlineOrientation: true}},
		{"server", paddleocr.Config{Detector: paddleocr.DetectorServer}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			m := bootModule(b, tc.cfg)
			page, _ := drawPage(b, documentLines, 40)
			b.ResetTimer()
			for b.Loop() {
				if _, err := m.Recognize(context.Background(), page, paddleocr.Options{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
