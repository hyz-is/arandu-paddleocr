package unit_test

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	paddleocr "github.com/kuaa-io/arandu-paddleocr"
)

// The golden values below are worked out by hand in the comments beside
// them, not by running the code under test.

func near(a, b, tolerance float64) bool { return math.Abs(a-b) <= tolerance }

func nearPoint(a, b paddleocr.Point, tolerance float64) bool {
	return near(a.X, b.X, tolerance) && near(a.Y, b.Y, tolerance)
}

func TestTheConvexHullDropsInteriorAndCollinearPoints(t *testing.T) {
	t.Parallel()

	points := []paddleocr.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 3}, {X: 0, Y: 3}, {X: 2, Y: 1}, {X: 1, Y: 2}, {X: 2, Y: 0}, {X: 4, Y: 1.5}, {X: 0, Y: 0}}
	hull := paddleocr.ConvexHull(points)
	want := []paddleocr.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 3}, {X: 0, Y: 3}}
	if !slices.Equal(hull, want) {
		t.Errorf("ConvexHull = %v, want %v", hull, want)
	}

	line := paddleocr.ConvexHull([]paddleocr.Point{{X: 3, Y: 3}, {X: 1, Y: 1}, {X: 2, Y: 2}, {X: 0, Y: 0}})
	if !slices.Equal(line, []paddleocr.Point{{X: 0, Y: 0}, {X: 3, Y: 3}}) {
		t.Errorf("hull of collinear points = %v, want its two ends", line)
	}
	if got := paddleocr.ConvexHull([]paddleocr.Point{{X: 5, Y: 5}, {X: 5, Y: 5}}); !slices.Equal(got, []paddleocr.Point{{X: 5, Y: 5}}) {
		t.Errorf("hull of one repeated point = %v", got)
	}
}

func TestTheMinimumAreaRectangleOfAnAxisAlignedRectangle(t *testing.T) {
	t.Parallel()

	// Pixels x 10..29, y 5..14: the rectangle through their centres is 19 by
	// 9, centred on (19.5, 9.5).
	points := []paddleocr.Point{}
	for y := 5; y <= 14; y++ {
		for x := 10; x <= 29; x++ {
			points = append(points, paddleocr.Point{X: float64(x), Y: float64(y)})
		}
	}
	box, side := paddleocr.MiniBox(points)
	want := paddleocr.Quad{{X: 10, Y: 5}, {X: 29, Y: 5}, {X: 29, Y: 14}, {X: 10, Y: 14}}
	for i := range box {
		if !nearPoint(box[i], want[i], 1e-9) {
			t.Fatalf("MiniBox = %v, want %v", box, want)
		}
	}
	if !near(side, 9, 1e-9) {
		t.Errorf("short side = %v, want 9", side)
	}
}

func TestTheMinimumAreaRectangleOfARotatedRectangle(t *testing.T) {
	t.Parallel()

	// A 40×10 rectangle turned by 30 degrees about (50, 50), and a scatter of
	// points inside it: the minimum rectangle is the rectangle itself.
	theta := math.Pi / 6
	c, s := math.Cos(theta), math.Sin(theta)
	at := func(u, v float64) paddleocr.Point {
		return paddleocr.Point{X: 50 + u*c - v*s, Y: 50 + u*s + v*c}
	}
	points := []paddleocr.Point{at(-20, -5), at(20, -5), at(20, 5), at(-20, 5)}
	r := rand.New(rand.NewPCG(1, 2))
	for range 200 {
		points = append(points, at(r.Float64()*40-20, r.Float64()*10-5))
	}

	rect := paddleocr.MinAreaRect(points)
	w, h := math.Max(rect.Width, rect.Height), math.Min(rect.Width, rect.Height)
	if !near(w, 40, 1e-6) || !near(h, 10, 1e-6) || !nearPoint(rect.Center, paddleocr.Point{X: 50, Y: 50}, 1e-6) {
		t.Errorf("MinAreaRect = %+v, want 40×10 about (50, 50)", rect)
	}
	// The long side runs at 30 degrees (or 210), whichever side was found.
	angle := math.Mod(math.Atan2(math.Sin(rect.Angle), math.Cos(rect.Angle))*180/math.Pi+360, 180)
	if rect.Width < rect.Height {
		angle = math.Mod(angle+90, 180)
	}
	if !near(angle, 30, 1e-6) {
		t.Errorf("long side at %v degrees, want 30", angle)
	}

	box, side := paddleocr.MiniBox(points)
	// Corners of the turned rectangle: (-20,-5) → (35.18, 35.67),
	// (20,-5) → (69.82, 55.67), (20,5) → (64.82, 64.33), (-20,5) → (30.18, 44.33).
	// Sorted by X the left pair is (30.18, 44.33) and (35.18, 35.67), of which
	// the second is higher: top-left. The right pair is (64.82, 64.33) and
	// (69.82, 55.67), of which the second is higher: top-right.
	want := paddleocr.Quad{at(-20, -5), at(20, -5), at(20, 5), at(-20, 5)}
	for i := range box {
		if !nearPoint(box[i], want[i], 1e-6) {
			t.Fatalf("MiniBox = %v, want %v", box, want)
		}
	}
	if !near(side, 10, 1e-6) {
		t.Errorf("short side = %v, want 10", side)
	}
}

// TestRotatingCalipersAgreeWithTryingEveryEdge checks the linear-time search
// against the quadratic one it replaces, on random point sets.
func TestRotatingCalipersAgreeWithTryingEveryEdge(t *testing.T) {
	t.Parallel()

	r := rand.New(rand.NewPCG(7, 11))
	for trial := range 300 {
		points := make([]paddleocr.Point, 3+r.IntN(40))
		for i := range points {
			points[i] = paddleocr.Point{X: math.Round(r.Float64() * 100), Y: math.Round(r.Float64() * 60)}
		}
		hull := paddleocr.ConvexHull(points)
		if len(hull) < 3 {
			continue
		}
		best := math.Inf(1)
		for i := range hull {
			a, b := hull[i], hull[(i+1)%len(hull)]
			l := math.Hypot(b.X-a.X, b.Y-a.Y)
			ux, uy := (b.X-a.X)/l, (b.Y-a.Y)/l
			minU, maxU, minV, maxV := math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)
			for _, p := range hull {
				u, v := p.X*ux+p.Y*uy, -p.X*uy+p.Y*ux
				minU, maxU = math.Min(minU, u), math.Max(maxU, u)
				minV, maxV = math.Min(minV, v), math.Max(maxV, v)
			}
			best = math.Min(best, (maxU-minU)*(maxV-minV))
		}
		rect := paddleocr.MinAreaRect(points)
		if !near(rect.Width*rect.Height, best, 1e-6*math.Max(1, best)) {
			t.Fatalf("trial %d: MinAreaRect area %v, every edge tried gives %v, for %v", trial, rect.Width*rect.Height, best, points)
		}
		// Every point lies inside the rectangle found.
		corners := rect.Corners()
		for _, p := range points {
			if !insideQuad(corners, p, 1e-6) {
				t.Fatalf("trial %d: %v lies outside %v", trial, p, corners)
			}
		}
	}
}

func insideQuad(q paddleocr.Quad, p paddleocr.Point, tolerance float64) bool {
	sign := 0.0
	for i := range q {
		a, b := q[i], q[(i+1)%4]
		c := (b.X-a.X)*(p.Y-a.Y) - (b.Y-a.Y)*(p.X-a.X)
		l := math.Hypot(b.X-a.X, b.Y-a.Y)
		if l > 0 {
			c /= l
		}
		if math.Abs(c) <= tolerance {
			continue
		}
		if sign == 0 {
			sign = math.Copysign(1, c)
		} else if math.Copysign(1, c) != sign {
			return false
		}
	}
	return true
}

func TestDegenerateInputsGiveDegenerateRectangles(t *testing.T) {
	t.Parallel()

	if r := paddleocr.MinAreaRect(nil); r != (paddleocr.RotatedRect{}) {
		t.Errorf("MinAreaRect(nil) = %+v", r)
	}
	if r := paddleocr.MinAreaRect([]paddleocr.Point{{X: 3, Y: 4}}); r.Center != (paddleocr.Point{X: 3, Y: 4}) || r.Width != 0 || r.Height != 0 {
		t.Errorf("MinAreaRect of a point = %+v", r)
	}
	r := paddleocr.MinAreaRect([]paddleocr.Point{{X: 0, Y: 0}, {X: 3, Y: 4}, {X: 6, Y: 8}})
	if !near(r.Width, 10, 1e-9) || r.Height != 0 || !nearPoint(r.Center, paddleocr.Point{X: 3, Y: 4}, 1e-9) {
		t.Errorf("MinAreaRect of a segment = %+v, want 10 long about (3, 4)", r)
	}
}

func TestUnclipGrowsARectangleByAreaTimesRatioOverPerimeter(t *testing.T) {
	t.Parallel()

	// 100×20: area 2000, perimeter 240, ratio 1.5 → distance 12.5, so the
	// rectangle becomes 125×45 about the same centre.
	q := paddleocr.Quad{{X: 0, Y: 0}, {X: 100, Y: 0}, {X: 100, Y: 20}, {X: 0, Y: 20}}
	if d := paddleocr.UnclipDistance(q, 1.5); !near(d, 12.5, 1e-12) {
		t.Errorf("UnclipDistance = %v, want 12.5", d)
	}
	got := paddleocr.Unclip(q, 1.5)
	want := paddleocr.Quad{{X: -12.5, Y: -12.5}, {X: 112.5, Y: -12.5}, {X: 112.5, Y: 32.5}, {X: -12.5, Y: 32.5}}
	for i := range got {
		if !nearPoint(got[i], want[i], 1e-9) {
			t.Fatalf("Unclip = %v, want %v", got, want)
		}
	}

	// The same rectangle listed the other way round grows the same way.
	reversed := paddleocr.Unclip(paddleocr.Quad{{X: 0, Y: 0}, {X: 0, Y: 20}, {X: 100, Y: 20}, {X: 100, Y: 0}}, 1.5)
	if !nearPoint(reversed[0], paddleocr.Point{X: -12.5, Y: -12.5}, 1e-9) || !nearPoint(reversed[2], paddleocr.Point{X: 112.5, Y: 32.5}, 1e-9) {
		t.Errorf("Unclip of the reversed rectangle = %v", reversed)
	}

	// Turned by 45 degrees it grows by the same distance along its own axes:
	// 125×45 about its centre.
	c := math.Sqrt2 / 2
	turnedQ := paddleocr.Quad{}
	for i, p := range q {
		turnedQ[i] = paddleocr.Point{X: p.X*c - p.Y*c, Y: p.X*c + p.Y*c}
	}
	grown := paddleocr.Unclip(turnedQ, 1.5)
	if w := math.Hypot(grown[1].X-grown[0].X, grown[1].Y-grown[0].Y); !near(w, 125, 1e-9) {
		t.Errorf("turned rectangle grew to width %v, want 125", w)
	}
	if h := math.Hypot(grown[3].X-grown[0].X, grown[3].Y-grown[0].Y); !near(h, 45, 1e-9) {
		t.Errorf("turned rectangle grew to height %v, want 45", h)
	}
}

func TestPolygonAreaAndPerimeter(t *testing.T) {
	t.Parallel()

	triangle := []paddleocr.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 0, Y: 3}}
	if a := paddleocr.PolygonArea(triangle); a != 6 {
		t.Errorf("area = %v, want 6", a)
	}
	if p := paddleocr.PolygonPerimeter(triangle); p != 12 {
		t.Errorf("perimeter = %v, want 12", p)
	}
}

func TestOrderClockwiseFindsTheCornersByTheirSums(t *testing.T) {
	t.Parallel()

	shuffled := paddleocr.Quad{{X: 90, Y: 40}, {X: 10, Y: 12}, {X: 10, Y: 40}, {X: 90, Y: 10}}
	want := paddleocr.Quad{{X: 10, Y: 12}, {X: 90, Y: 10}, {X: 90, Y: 40}, {X: 10, Y: 40}}
	if got := paddleocr.OrderClockwise(shuffled); got != want {
		t.Errorf("OrderClockwise = %v, want %v", got, want)
	}
}

func TestReadingOrderIsRowsThenColumns(t *testing.T) {
	t.Parallel()

	box := func(x, y float64) paddleocr.Quad {
		return paddleocr.Quad{{X: x, Y: y}, {X: x + 20, Y: y}, {X: x + 20, Y: y + 10}, {X: x, Y: y + 10}}
	}
	boxes := []paddleocr.Quad{
		box(200, 52), // 0: second row, right
		box(10, 0),   // 1: first row, left
		box(300, 5),  // 2: first row, right, 5 px lower than its neighbour
		box(10, 55),  // 3: second row, left
		box(150, 2),  // 4: first row, middle
		box(10, 100), // 5: third row
	}
	got := paddleocr.ReadingOrder(boxes, 10)
	want := []int{1, 4, 2, 3, 0, 5}
	if !slices.Equal(got, want) {
		t.Errorf("ReadingOrder = %v, want %v", got, want)
	}

	// With no tolerance the order is by the top-left corner's Y alone.
	if got := paddleocr.ReadingOrder(boxes, 0); !slices.Equal(got, []int{1, 4, 2, 0, 3, 5}) {
		t.Errorf("ReadingOrder without tolerance = %v", got)
	}
	if got := paddleocr.ReadingOrder(nil, 10); len(got) != 0 {
		t.Errorf("ReadingOrder(nil) = %v", got)
	}
}
