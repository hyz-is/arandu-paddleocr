package unit_test

import (
	"math"
	"testing"

	paddleocr "github.com/hyz-is/arandu-paddleocr"
)

// probabilityMap is a map of the given size whose every pixel is background.
func probabilityMap(width, height int) paddleocr.ProbabilityMap {
	return paddleocr.ProbabilityMap{Width: width, Height: height, Data: make([]float32, width*height)}
}

// fill sets the pixels of the rectangle x0..x1, y0..y1 (inclusive) to p.
func fill(m paddleocr.ProbabilityMap, x0, y0, x1, y1 int, p float32) {
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			m.Data[y*m.Width+x] = p
		}
	}
}

func TestDBFindsAnAxisAlignedRectangle(t *testing.T) {
	t.Parallel()

	m := probabilityMap(100, 50)
	fill(m, 20, 10, 79, 29, 0.9)

	// Pixel centres x 20..79, y 10..29: a 59×19 box about (49.5, 19.5). Its
	// unclip distance is 59·19·1.5 / (2·(59+19)) = 1681.5/156 = 10.778846…, so
	// it grows to half-sizes 40.278846 and 20.278846: x 9.221 to 89.779 and
	// y -0.779 to 39.779, rounded to 9, 90, -1 and 40, and -1 clipped to 0.
	boxes, err := paddleocr.DBPostProcess(m, 100, 50, paddleocr.DefaultDBParams())
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 {
		t.Fatalf("found %d boxes, want 1: %+v", len(boxes), boxes)
	}
	want := paddleocr.Quad{{X: 9, Y: 0}, {X: 90, Y: 0}, {X: 90, Y: 40}, {X: 9, Y: 40}}
	if boxes[0].Box != want {
		t.Errorf("box = %v, want %v", boxes[0].Box, want)
	}
	if !near(boxes[0].Score, 0.9, 1e-6) {
		t.Errorf("score = %v, want 0.9", boxes[0].Score)
	}

	// Onto a destination twice the size: x 18.44 to 179.56 and y -1.56 to
	// 79.56, rounded and clipped to 18, 180, 0 and 80.
	boxes, err = paddleocr.DBPostProcess(m, 200, 100, paddleocr.DefaultDBParams())
	if err != nil {
		t.Fatal(err)
	}
	if want := (paddleocr.Quad{{X: 18, Y: 0}, {X: 180, Y: 0}, {X: 180, Y: 80}, {X: 18, Y: 80}}); len(boxes) != 1 || boxes[0].Box != want {
		t.Errorf("scaled boxes = %+v, want %v", boxes, want)
	}
}

func TestDBFindsARotatedRectangle(t *testing.T) {
	t.Parallel()

	// A 80×16 rectangle turned 20 degrees about (100, 60).
	m := probabilityMap(200, 120)
	theta := 20 * math.Pi / 180
	c, s := math.Cos(theta), math.Sin(theta)
	for y := 0; y < m.Height; y++ {
		for x := 0; x < m.Width; x++ {
			dx, dy := float64(x)-100, float64(y)-60
			u, v := dx*c+dy*s, -dx*s+dy*c
			if math.Abs(u) <= 40 && math.Abs(v) <= 8 {
				m.Data[y*m.Width+x] = 0.95
			}
		}
	}
	boxes, err := paddleocr.DBPostProcess(m, 200, 120, paddleocr.DefaultDBParams())
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 {
		t.Fatalf("found %d boxes, want 1", len(boxes))
	}
	b := boxes[0].Box
	top := math.Atan2(b[1].Y-b[0].Y, b[1].X-b[0].X) * 180 / math.Pi
	if !near(top, 20, 1.5) {
		t.Errorf("top side at %.2f degrees, want about 20", top)
	}
	// The region is about 80×16; unclip adds 2·80·16·1.5/192 = 20 to each
	// side's length.
	w := math.Hypot(b[1].X-b[0].X, b[1].Y-b[0].Y)
	h := math.Hypot(b[3].X-b[0].X, b[3].Y-b[0].Y)
	if !near(w, 100, 3) || !near(h, 36, 3) {
		t.Errorf("box is %.1f×%.1f, want about 100×36", w, h)
	}
	if boxes[0].Score < 0.9 {
		t.Errorf("score = %v, want about 0.95", boxes[0].Score)
	}
}

func TestDBDropsWeakTinyAndSurplusRegions(t *testing.T) {
	t.Parallel()

	m := probabilityMap(120, 60)
	fill(m, 5, 5, 50, 20, 0.9)    // kept
	fill(m, 60, 5, 110, 20, 0.5)  // above thresh, under box thresh
	fill(m, 5, 40, 6, 41, 0.9)    // 2×2: under the minimum size
	fill(m, 60, 35, 110, 50, 0.8) // kept

	boxes, err := paddleocr.DBPostProcess(m, 120, 60, paddleocr.DefaultDBParams())
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 2 {
		t.Fatalf("found %d boxes, want the two strong ones: %+v", len(boxes), boxes)
	}
	if !near(boxes[0].Score, 0.9, 1e-6) || !near(boxes[1].Score, 0.8, 1e-6) {
		t.Errorf("scores = %v and %v, want 0.9 and 0.8 in scan order", boxes[0].Score, boxes[1].Score)
	}

	params := paddleocr.DefaultDBParams()
	params.MaxCandidates = 1
	boxes, err = paddleocr.DBPostProcess(m, 120, 60, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 || !near(boxes[0].Score, 0.9, 1e-6) {
		t.Errorf("with one candidate found %+v, want the first region met", boxes)
	}

	params = paddleocr.DefaultDBParams()
	params.BoxThresh = 0.4
	boxes, err = paddleocr.DBPostProcess(m, 120, 60, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 3 {
		t.Errorf("with a box threshold of 0.4 found %d boxes, want 3", len(boxes))
	}
}

func TestDBJoinsDiagonallyTouchingPixels(t *testing.T) {
	t.Parallel()

	// Two 10×10 blocks touching at one corner are one 8-connected region. Its
	// box is a diagonal band that is mostly background, so the threshold is
	// lowered for it to be kept at all.
	m := probabilityMap(60, 60)
	fill(m, 10, 10, 19, 19, 0.9)
	fill(m, 20, 20, 29, 29, 0.9)
	params := paddleocr.DefaultDBParams()
	params.BoxThresh = 0.1
	boxes, err := paddleocr.DBPostProcess(m, 60, 60, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 {
		t.Errorf("found %d boxes, want one region", len(boxes))
	}
}

func TestDBRefusesAMalformedMap(t *testing.T) {
	t.Parallel()

	if _, err := paddleocr.DBPostProcess(paddleocr.ProbabilityMap{Width: 4, Height: 4, Data: make([]float32, 15)}, 4, 4, paddleocr.DefaultDBParams()); err == nil {
		t.Error("a map with the wrong number of values was accepted")
	}
	if _, err := paddleocr.DBPostProcess(probabilityMap(4, 4), 0, 4, paddleocr.DefaultDBParams()); err == nil {
		t.Error("an empty destination was accepted")
	}
	boxes, err := paddleocr.DBPostProcess(probabilityMap(8, 8), 8, 8, paddleocr.DefaultDBParams())
	if err != nil || len(boxes) != 0 {
		t.Errorf("an empty map = %v, %v; want no boxes", boxes, err)
	}
}

func TestBoxScoreIsTheMeanInsideTheBox(t *testing.T) {
	t.Parallel()

	m := probabilityMap(10, 10)
	fill(m, 0, 0, 4, 9, 1) // left half 1, right half 0
	// The box covers columns 2..7 of rows 2..5: three columns at 1 and three
	// at 0, a mean of 0.5.
	if got := paddleocr.BoxScore(m, paddleocr.Quad{{X: 2, Y: 2}, {X: 7, Y: 2}, {X: 7, Y: 5}, {X: 2, Y: 5}}); !near(got, 0.5, 1e-12) {
		t.Errorf("BoxScore = %v, want 0.5", got)
	}
	// Corners are truncated to whole pixels: 2.9 is column 2.
	if got := paddleocr.BoxScore(m, paddleocr.Quad{{X: 2.9, Y: 2}, {X: 4.9, Y: 2}, {X: 4.9, Y: 5}, {X: 2.9, Y: 5}}); !near(got, 1, 1e-12) {
		t.Errorf("BoxScore of a truncated box = %v, want 1", got)
	}
}
