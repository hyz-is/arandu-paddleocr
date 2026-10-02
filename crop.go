package paddleocr

import (
	"errors"
	"fmt"
	"image"
	"math"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// Homography is a 3×3 projective transform, row by row. It maps (x, y) to
// ((h0·x + h1·y + h2) / w, (h3·x + h4·y + h5) / w) with w = h6·x + h7·y + h8.
type Homography [9]float64

// Apply maps a point through the transform.
func (h Homography) Apply(p Point) Point {
	w := h[6]*p.X + h[7]*p.Y + h[8]
	return Point{
		X: (h[0]*p.X + h[1]*p.Y + h[2]) / w,
		Y: (h[3]*p.X + h[4]*p.Y + h[5]) / w,
	}
}

// ErrDegenerateQuad is a quadrilateral no projective transform can map onto
// another: three of its corners on one line, or two in one place.
var ErrDegenerateQuad = errors.New("paddleocr: degenerate quadrilateral")

// PerspectiveTransform returns the homography that maps each corner of from
// onto the corner of to at the same index, with h8 = 1, solving the
// eight-unknown linear system by Gaussian elimination with partial pivoting.
func PerspectiveTransform(from, to Quad) (Homography, error) {
	var a [8][9]float64
	for i := 0; i < 4; i++ {
		x, y := from[i].X, from[i].Y
		u, v := to[i].X, to[i].Y
		a[2*i] = [9]float64{x, y, 1, 0, 0, 0, -x * u, -y * u, u}
		a[2*i+1] = [9]float64{0, 0, 0, x, y, 1, -x * v, -y * v, v}
	}
	for col := 0; col < 8; col++ {
		pivot := col
		for row := col + 1; row < 8; row++ {
			if math.Abs(a[row][col]) > math.Abs(a[pivot][col]) {
				pivot = row
			}
		}
		if math.Abs(a[pivot][col]) < 1e-10 {
			return Homography{}, ErrDegenerateQuad
		}
		a[col], a[pivot] = a[pivot], a[col]
		for row := 0; row < 8; row++ {
			if row == col {
				continue
			}
			f := a[row][col] / a[col][col]
			for k := col; k < 9; k++ {
				a[row][k] -= f * a[col][k]
			}
		}
	}
	var h Homography
	for i := 0; i < 8; i++ {
		h[i] = a[i][8] / a[i][i]
	}
	h[8] = 1
	return h, nil
}

// CropSize is the size PaddleOCR gives the upright crop of a quadrilateral
// ordered top-left, top-right, bottom-right, bottom-left: the longer of its
// top and bottom sides by the longer of its left and right sides, each
// truncated to whole pixels.
func CropSize(q Quad) (width, height int) {
	dist := func(a, b Point) float64 { return math.Hypot(b.X-a.X, b.Y-a.Y) }
	width = int(math.Max(dist(q[0], q[1]), dist(q[2], q[3])))
	height = int(math.Max(dist(q[0], q[3]), dist(q[1], q[2])))
	return width, height
}

// WarpQuad cuts the quadrilateral q, ordered top-left, top-right,
// bottom-right, bottom-left, out of the image and straightens it into an
// upright rectangle of CropSize(q).
//
// Each output pixel (x, y) reads the source at the point the homography from
// the rectangle (0,0) (w,0) (w,h) (0,h) onto q gives for it, with bilinear
// interpolation, and positions beyond the image's edge repeat the edge pixel.
// That is OpenCV's warpPerspective with BORDER_REPLICATE, which PaddleOCR
// uses, except that it interpolates bicubically.
func WarpQuad(src image.Image, q Quad) (*image.RGBA, error) {
	width, height := CropSize(q)
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("%w: its crop would be %d×%d", ErrDegenerateQuad, width, height)
	}
	rect := Quad{{0, 0}, {float64(width), 0}, {float64(width), float64(height)}, {0, float64(height)}}
	toSource, err := PerspectiveTransform(rect, q)
	if err != nil {
		return nil, err
	}
	in, err := asRGBA(src)
	if err != nil {
		return nil, err
	}
	out := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			p := toSource.Apply(Point{float64(x), float64(y)})
			o := out.PixOffset(x, y)
			sampleBilinear(in, p.X, p.Y, out.Pix[o:o+4])
		}
	}
	return out, nil
}

// sampleBilinear writes the bilinear interpolation of img at (x, y) into
// dst, clamping coordinates to the image so its edge repeats.
func sampleBilinear(img *image.RGBA, x, y float64, dst []uint8) {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	x = clampFloat(x, 0, float64(w-1))
	y = clampFloat(y, 0, float64(h-1))
	x0, y0 := int(math.Floor(x)), int(math.Floor(y))
	x1, y1 := min(x0+1, w-1), min(y0+1, h-1)
	fx, fy := x-float64(x0), y-float64(y0)
	p00 := img.Pix[y0*img.Stride+x0*4:]
	p10 := img.Pix[y0*img.Stride+x1*4:]
	p01 := img.Pix[y1*img.Stride+x0*4:]
	p11 := img.Pix[y1*img.Stride+x1*4:]
	for c := 0; c < 4; c++ {
		top := float64(p00[c]) + (float64(p10[c])-float64(p00[c]))*fx
		bottom := float64(p01[c]) + (float64(p11[c])-float64(p01[c]))*fx
		v := math.Round(top + (bottom-top)*fy)
		dst[c] = uint8(clampFloat(v, 0, 255))
	}
}

// asRGBA returns the image as an *image.RGBA with its origin at (0, 0),
// copying it only when it is not one already.
func asRGBA(src image.Image) (*image.RGBA, error) {
	if rgba, ok := src.(*image.RGBA); ok && rgba.Rect.Min == (image.Point{}) {
		return rgba, nil
	}
	return onnx.Crop(src, src.Bounds())
}

// Rotate turns an image counter-clockwise by a multiple of 90 degrees. Any
// other angle is refused.
func Rotate(src image.Image, degrees int) (*image.RGBA, error) {
	in, err := asRGBA(src)
	if err != nil {
		return nil, err
	}
	degrees = ((degrees % 360) + 360) % 360
	w, h := in.Rect.Dx(), in.Rect.Dy()
	var out *image.RGBA
	var from func(x, y int) (int, int)
	switch degrees {
	case 0:
		out = image.NewRGBA(image.Rect(0, 0, w, h))
		from = func(x, y int) (int, int) { return x, y }
	case 90:
		out = image.NewRGBA(image.Rect(0, 0, h, w))
		from = func(x, y int) (int, int) { return w - 1 - y, x }
	case 180:
		out = image.NewRGBA(image.Rect(0, 0, w, h))
		from = func(x, y int) (int, int) { return w - 1 - x, h - 1 - y }
	case 270:
		out = image.NewRGBA(image.Rect(0, 0, h, w))
		from = func(x, y int) (int, int) { return y, h - 1 - x }
	default:
		return nil, fmt.Errorf("paddleocr: rotation by %d degrees: only multiples of 90 are supported", degrees)
	}
	ow, oh := out.Rect.Dx(), out.Rect.Dy()
	for y := 0; y < oh; y++ {
		for x := 0; x < ow; x++ {
			sx, sy := from(x, y)
			copy(out.Pix[y*out.Stride+x*4:y*out.Stride+x*4+4], in.Pix[sy*in.Stride+sx*4:sy*in.Stride+sx*4+4])
		}
	}
	return out, nil
}

// unrotatePoint maps a point on an image Rotate turned by degrees back onto
// the image before the turn, whose size was width × height.
func unrotatePoint(p Point, degrees, width, height int) Point {
	w, h := float64(width-1), float64(height-1)
	switch ((degrees % 360) + 360) % 360 {
	case 90:
		return Point{X: w - p.Y, Y: p.X}
	case 180:
		return Point{X: w - p.X, Y: h - p.Y}
	case 270:
		return Point{X: p.Y, Y: h - p.X}
	default:
		return p
	}
}
