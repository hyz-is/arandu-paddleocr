package paddleocr

import (
	"errors"
	"fmt"
	"math"
)

// ProbabilityMap is the detector's output for one image: for every pixel of
// the resized input, the probability that it belongs to text, row by row.
type ProbabilityMap struct {
	// Width and Height are the map's dimensions, which are the resized
	// input's.
	Width, Height int
	// Data holds Width × Height probabilities.
	Data []float32
}

// DBParams are the constants of DB post-processing.
type DBParams struct {
	// Thresh binarizes the map: a pixel above it is text.
	Thresh float64
	// BoxThresh drops a box whose mean probability is below it.
	BoxThresh float64
	// UnclipRatio sets how far a box is grown back, as a multiple of its
	// area over its perimeter.
	UnclipRatio float64
	// MaxCandidates caps how many text regions are examined.
	MaxCandidates int
	// MinSize drops a region whose box is shorter than this on its short
	// side, before growing it. After growing, the floor is MinSize + 2.
	MinSize float64
}

// DefaultDBParams are the values the PP-OCRv5 detectors ship with.
func DefaultDBParams() DBParams {
	return DBParams{Thresh: 0.3, BoxThresh: 0.6, UnclipRatio: 1.5, MaxCandidates: 1000, MinSize: 3}
}

// TextBox is one region DB post-processing kept.
type TextBox struct {
	// Box is the region's rectangle in the destination's coordinates, ordered
	// top-left, top-right, bottom-right, bottom-left.
	Box Quad
	// Score is the mean probability inside the region's box before it was
	// grown.
	Score float64
}

// DBPostProcess turns a probability map into text boxes, the way PaddleOCR's
// DBPostProcess does.
//
// The map is binarized at Thresh and split into 8-connected regions. For each
// region, up to MaxCandidates of them in the order they are first met
// scanning row by row, the minimum-area rectangle of its pixels is taken; a
// rectangle whose short side is under MinSize is dropped, and so is one whose
// mean probability is under BoxThresh. The rectangle is grown by UnclipRatio,
// boxed again, dropped when its short side is under MinSize + 2, and scaled
// from the map onto a destination of destWidth × destHeight, rounded and
// clipped to it.
func DBPostProcess(m ProbabilityMap, destWidth, destHeight int, p DBParams) ([]TextBox, error) {
	if m.Width <= 0 || m.Height <= 0 || len(m.Data) != m.Width*m.Height {
		return nil, fmt.Errorf("paddleocr: a %d×%d probability map needs %d values and has %d", m.Width, m.Height, max(0, m.Width*m.Height), len(m.Data))
	}
	if destWidth <= 0 || destHeight <= 0 {
		return nil, errors.New("paddleocr: the destination of DB post-processing has to have a positive size")
	}

	regions := connectedRegions(m, float32(p.Thresh), p.MaxCandidates)
	boxes := make([]TextBox, 0, len(regions))
	sx := float64(destWidth) / float64(m.Width)
	sy := float64(destHeight) / float64(m.Height)
	for _, region := range regions {
		box, side := MiniBox(region)
		if side < p.MinSize {
			continue
		}
		score := BoxScore(m, box)
		if score < p.BoxThresh {
			continue
		}
		unclipped := Unclip(box, p.UnclipRatio)
		grown, side := MiniBox(unclipped[:])
		if side < p.MinSize+2 {
			continue
		}
		for i := range grown {
			grown[i].X = clampFloat(math.RoundToEven(grown[i].X*sx), 0, float64(destWidth))
			grown[i].Y = clampFloat(math.RoundToEven(grown[i].Y*sy), 0, float64(destHeight))
		}
		boxes = append(boxes, TextBox{Box: grown, Score: score})
	}
	return boxes, nil
}

// connectedRegions finds the 8-connected regions of pixels above thresh, in
// the order a row-by-row scan meets them, and returns for each the outline
// points its convex hull is taken from: the first and last pixel of every row
// it spans. At most limit regions are returned; limit <= 0 means no limit.
func connectedRegions(m ProbabilityMap, thresh float32, limit int) [][]Point {
	w, h := m.Width, m.Height
	label := make([]int32, w*h)
	regions := [][]Point{}
	stack := []int{}
	for start := range m.Data {
		if m.Data[start] <= thresh || label[start] != 0 {
			continue
		}
		if limit > 0 && len(regions) == limit {
			break
		}
		id := int32(len(regions) + 1)
		label[start] = id
		stack = append(stack[:0], start)

		// Per row, the leftmost and rightmost pixel of the region.
		top, bottom := start/w, start/w
		rowMin := map[int]int{}
		rowMax := map[int]int{}
		for len(stack) > 0 {
			at := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			x, y := at%w, at/w
			if v, ok := rowMin[y]; !ok || x < v {
				rowMin[y] = x
			}
			if v, ok := rowMax[y]; !ok || x > v {
				rowMax[y] = x
			}
			top, bottom = min(top, y), max(bottom, y)
			for dy := -1; dy <= 1; dy++ {
				ny := y + dy
				if ny < 0 || ny >= h {
					continue
				}
				for dx := -1; dx <= 1; dx++ {
					nx := x + dx
					if nx < 0 || nx >= w || (dx == 0 && dy == 0) {
						continue
					}
					n := ny*w + nx
					if label[n] == 0 && m.Data[n] > thresh {
						label[n] = id
						stack = append(stack, n)
					}
				}
			}
		}
		outline := make([]Point, 0, 2*(bottom-top+1))
		for y := top; y <= bottom; y++ {
			if x, ok := rowMin[y]; ok {
				outline = append(outline, Point{float64(x), float64(y)})
				if rowMax[y] != x {
					outline = append(outline, Point{float64(rowMax[y]), float64(y)})
				}
			}
		}
		regions = append(regions, outline)
	}
	return regions
}

// BoxScore is the mean probability of the pixels inside a convex
// quadrilateral on the map, boundary included, with the corners truncated to
// whole pixels as PaddleOCR's fast box score does. A box that covers no pixel
// scores 0.
func BoxScore(m ProbabilityMap, q Quad) float64 {
	var poly Quad
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	for i, p := range q {
		poly[i] = Point{math.Trunc(p.X), math.Trunc(p.Y)}
		minX, maxX = math.Min(minX, p.X), math.Max(maxX, p.X)
		minY, maxY = math.Min(minY, p.Y), math.Max(maxY, p.Y)
	}
	x0 := clampInt(int(math.Floor(minX)), 0, m.Width-1)
	x1 := clampInt(int(math.Ceil(maxX)), 0, m.Width-1)
	y0 := clampInt(int(math.Floor(minY)), 0, m.Height-1)
	y1 := clampInt(int(math.Ceil(maxY)), 0, m.Height-1)

	sum, count := 0.0, 0
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			if insideConvex(poly, Point{float64(x), float64(y)}) {
				sum += float64(m.Data[y*m.Width+x])
				count++
			}
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

// insideConvex reports whether p lies inside a convex quadrilateral or on its
// boundary, whichever way round its corners go.
func insideConvex(q Quad, p Point) bool {
	positive, negative := false, false
	for i := range q {
		c := cross(q[i], q[(i+1)%4], p)
		if c > 1e-9 {
			positive = true
		} else if c < -1e-9 {
			negative = true
		}
		if positive && negative {
			return false
		}
	}
	return true
}

func clampFloat(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}

func clampInt(v, lo, hi int) int {
	return max(lo, min(hi, v))
}
