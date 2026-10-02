package paddleocr

import (
	"math"
	"sort"
)

// Point is a position on an image, in pixels. X grows to the right and Y
// downwards; integer values are pixel indexes.
type Point struct {
	X, Y float64
}

// Quad is a quadrilateral given by its four corners. Where a function returns
// one, the corners are in the order top-left, top-right, bottom-right,
// bottom-left of the text it encloses.
type Quad [4]Point

// RotatedRect is a rectangle at any angle.
type RotatedRect struct {
	// Center is the rectangle's centre.
	Center Point
	// Width is the length of the side that runs along Angle.
	Width float64
	// Height is the length of the side perpendicular to it.
	Height float64
	// Angle is the direction of the Width side, in radians.
	Angle float64
}

// Corners returns the rectangle's four corners, walking around it.
func (r RotatedRect) Corners() Quad {
	ux, uy := math.Cos(r.Angle), math.Sin(r.Angle)
	vx, vy := -uy, ux
	hw, hh := r.Width/2, r.Height/2
	at := func(su, sv float64) Point {
		return Point{
			X: r.Center.X + su*hw*ux + sv*hh*vx,
			Y: r.Center.Y + su*hw*uy + sv*hh*vy,
		}
	}
	return Quad{at(-1, -1), at(1, -1), at(1, 1), at(-1, 1)}
}

// cross is the z component of (b-a) x (c-a): positive when a, b, c turn
// counter-clockwise in a frame whose Y axis points up.
func cross(a, b, c Point) float64 {
	return (b.X-a.X)*(c.Y-a.Y) - (b.Y-a.Y)*(c.X-a.X)
}

// ConvexHull returns the convex hull of the points with Andrew's monotone
// chain, as a closed polygon without the repeated first vertex. Collinear
// points on the hull's edges are dropped. Fewer than three distinct points
// come back as they are, deduplicated and sorted.
func ConvexHull(points []Point) []Point {
	ps := append([]Point(nil), points...)
	sort.Slice(ps, func(i, j int) bool {
		if ps[i].X != ps[j].X {
			return ps[i].X < ps[j].X
		}
		return ps[i].Y < ps[j].Y
	})
	unique := ps[:0]
	for i, p := range ps {
		if i == 0 || p != ps[i-1] {
			unique = append(unique, p)
		}
	}
	ps = unique
	if len(ps) < 3 {
		return append([]Point(nil), ps...)
	}

	hull := make([]Point, 0, 2*len(ps))
	for _, p := range ps {
		for len(hull) >= 2 && cross(hull[len(hull)-2], hull[len(hull)-1], p) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}
	lower := len(hull) + 1
	for i := len(ps) - 2; i >= 0; i-- {
		p := ps[i]
		for len(hull) >= lower && cross(hull[len(hull)-2], hull[len(hull)-1], p) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}
	hull = hull[:len(hull)-1]
	if len(hull) < 3 {
		// Every point was on one line: the two ends are the hull.
		return []Point{ps[0], ps[len(ps)-1]}
	}
	return hull
}

// MinAreaRect returns the rectangle of least area that encloses the points.
//
// It runs rotating calipers over the convex hull: one side of the minimum
// rectangle lies on an edge of the hull, and as the edge advances around the
// hull the three other extreme points only ever advance too, so every edge is
// tried in linear time. An empty input gives the zero RotatedRect.
func MinAreaRect(points []Point) RotatedRect {
	hull := ConvexHull(points)
	switch len(hull) {
	case 0:
		return RotatedRect{}
	case 1:
		return RotatedRect{Center: hull[0]}
	case 2:
		a, b := hull[0], hull[1]
		return RotatedRect{
			Center: Point{(a.X + b.X) / 2, (a.Y + b.Y) / 2},
			Width:  math.Hypot(b.X-a.X, b.Y-a.Y),
			Angle:  math.Atan2(b.Y-a.Y, b.X-a.X),
		}
	}

	n := len(hull)
	next := func(i int) int { return (i + 1) % n }
	dot := func(p Point, ux, uy float64) float64 { return p.X*ux + p.Y*uy }

	best := RotatedRect{}
	bestArea := math.Inf(1)
	// right is the vertex furthest along the edge, top the furthest from it,
	// left the furthest back along it.
	right, top, left := 1, 1, 1
	for i := 0; i < n; i++ {
		a, b := hull[i], hull[next(i)]
		length := math.Hypot(b.X-a.X, b.Y-a.Y)
		ux, uy := (b.X-a.X)/length, (b.Y-a.Y)/length
		// The hull is counter-clockwise in a Y-up frame, so the interior lies
		// to the left of each edge: the normal (-uy, ux).
		vx, vy := -uy, ux

		if i == 0 {
			right = next(i)
		}
		for dot(hull[next(right)], ux, uy) >= dot(hull[right], ux, uy)-1e-12 && next(right) != i {
			right = next(right)
		}
		if i == 0 {
			top = right
		}
		for dot(hull[next(top)], vx, vy) >= dot(hull[top], vx, vy)-1e-12 && next(top) != i {
			top = next(top)
		}
		if i == 0 {
			left = top
		}
		for dot(hull[next(left)], ux, uy) <= dot(hull[left], ux, uy)+1e-12 && next(left) != next(i) {
			left = next(left)
		}

		minU, maxU := dot(hull[left], ux, uy), dot(hull[right], ux, uy)
		minV, maxV := dot(a, vx, vy), dot(hull[top], vx, vy)
		area := (maxU - minU) * (maxV - minV)
		if area < bestArea-1e-9 {
			bestArea = area
			cu, cv := (minU+maxU)/2, (minV+maxV)/2
			best = RotatedRect{
				Center: Point{X: cu*ux + cv*vx, Y: cu*uy + cv*vy},
				Width:  maxU - minU,
				Height: maxV - minV,
				Angle:  math.Atan2(uy, ux),
			}
		}
	}
	return best
}

// MiniBox returns the minimum-area rectangle around the points with its
// corners ordered top-left, top-right, bottom-right, bottom-left, and the
// length of its shorter side.
//
// The ordering is the one PaddleOCR's DB post-processing applies: the two
// corners with the smallest X are the left side, and of each side the one
// with the smaller Y is the top.
func MiniBox(points []Point) (Quad, float64) {
	rect := MinAreaRect(points)
	corners := rect.Corners()
	ps := corners[:]
	sort.SliceStable(ps, func(i, j int) bool { return ps[i].X < ps[j].X })
	var tl, tr, br, bl Point
	if ps[1].Y > ps[0].Y {
		tl, bl = ps[0], ps[1]
	} else {
		tl, bl = ps[1], ps[0]
	}
	if ps[3].Y > ps[2].Y {
		tr, br = ps[2], ps[3]
	} else {
		tr, br = ps[3], ps[2]
	}
	return Quad{tl, tr, br, bl}, math.Min(rect.Width, rect.Height)
}

// PolygonArea is the absolute area of a simple polygon, by the shoelace
// formula.
func PolygonArea(polygon []Point) float64 {
	area := 0.0
	for i := range polygon {
		a, b := polygon[i], polygon[(i+1)%len(polygon)]
		area += a.X*b.Y - b.X*a.Y
	}
	return math.Abs(area) / 2
}

// PolygonPerimeter is the length of a closed polygon's boundary.
func PolygonPerimeter(polygon []Point) float64 {
	length := 0.0
	for i := range polygon {
		a, b := polygon[i], polygon[(i+1)%len(polygon)]
		length += math.Hypot(b.X-a.X, b.Y-a.Y)
	}
	return length
}

// UnclipDistance is how far DB post-processing pushes a box outwards:
// area × ratio / perimeter. A box with no perimeter does not move.
func UnclipDistance(q Quad, ratio float64) float64 {
	perimeter := PolygonPerimeter(q[:])
	if perimeter == 0 {
		return 0
	}
	return PolygonArea(q[:]) * ratio / perimeter
}

// Unclip grows a convex quadrilateral outwards by UnclipDistance(q, ratio),
// moving every side along its outward normal and joining the sides where they
// meet.
//
// DB post-processing offsets the shrunk text region with rounded corners and
// then takes the minimum-area rectangle of the result. For a rectangle, which
// is what the region's box always is, that rectangle is exactly the one this
// function returns: each side moved out by the distance. The corner order is
// kept.
func Unclip(q Quad, ratio float64) Quad {
	d := UnclipDistance(q, ratio)
	if d == 0 {
		return q
	}
	// The sign of the shoelace sum says which side of each edge is outside.
	orientation := 0.0
	for i := range q {
		a, b := q[i], q[(i+1)%4]
		orientation += a.X*b.Y - b.X*a.Y
	}
	sign := 1.0
	if orientation < 0 {
		sign = -1
	}

	type line struct{ p, dir Point }
	lines := [4]line{}
	for i := range q {
		a, b := q[i], q[(i+1)%4]
		length := math.Hypot(b.X-a.X, b.Y-a.Y)
		if length == 0 {
			return q
		}
		dx, dy := (b.X-a.X)/length, (b.Y-a.Y)/length
		// For a positive shoelace sum the interior is to the left of each edge,
		// (-dy, dx), so the outward normal is (dy, -dx).
		nx, ny := sign*dy, -sign*dx
		lines[i] = line{p: Point{a.X + nx*d, a.Y + ny*d}, dir: Point{dx, dy}}
	}

	var out Quad
	for i := range q {
		prev, cur := lines[(i+3)%4], lines[i]
		denominator := prev.dir.X*cur.dir.Y - prev.dir.Y*cur.dir.X
		if math.Abs(denominator) < 1e-12 {
			out[i] = cur.p
			continue
		}
		t := ((cur.p.X-prev.p.X)*cur.dir.Y - (cur.p.Y-prev.p.Y)*cur.dir.X) / denominator
		out[i] = Point{prev.p.X + t*prev.dir.X, prev.p.Y + t*prev.dir.Y}
	}
	return out
}

// OrderClockwise orders four corners top-left, top-right, bottom-right,
// bottom-left the way PaddleOCR does before it crops: the smallest X+Y is the
// top-left, the largest the bottom-right, and of the other two the smaller
// Y-X is the top-right.
func OrderClockwise(q Quad) Quad {
	sums := [4]float64{}
	for i, p := range q {
		sums[i] = p.X + p.Y
	}
	first, last := 0, 0
	for i := range q {
		if sums[i] < sums[first] {
			first = i
		}
		if sums[i] > sums[last] {
			last = i
		}
	}
	if first == last {
		return q
	}
	rest := make([]Point, 0, 2)
	for i := range q {
		if i != first && i != last {
			rest = append(rest, q[i])
		}
	}
	tr, bl := rest[0], rest[1]
	if rest[1].Y-rest[1].X < rest[0].Y-rest[0].X {
		tr, bl = rest[1], rest[0]
	}
	return Quad{q[first], tr, q[last], bl}
}

// ReadingOrder returns the indexes of the boxes in the order a reader meets
// them: top to bottom, and left to right among boxes whose top-left corners
// are less than tolerance pixels apart vertically. It is PaddleOCR's
// sorted_boxes rule, which uses a tolerance of 10.
func ReadingOrder(boxes []Quad, tolerance float64) []int {
	order := make([]int, len(boxes))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		a, b := boxes[order[i]][0], boxes[order[j]][0]
		if a.Y != b.Y {
			return a.Y < b.Y
		}
		return a.X < b.X
	})
	for i := 0; i < len(order)-1; i++ {
		for j := i; j >= 0; j-- {
			upper, lower := boxes[order[j]][0], boxes[order[j+1]][0]
			if math.Abs(lower.Y-upper.Y) < tolerance && lower.X < upper.X {
				order[j], order[j+1] = order[j+1], order[j]
				continue
			}
			break
		}
	}
	return order
}
