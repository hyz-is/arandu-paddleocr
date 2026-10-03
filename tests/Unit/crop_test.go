package unit_test

import (
	"image"
	"testing"

	paddleocr "github.com/hyz-is/arandu-paddleocr"
)

// gradient is an image whose pixel (x, y) has red x, green y and blue x+y, so
// any pixel says where it came from.
func gradient(width, height int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			o := img.PixOffset(x, y)
			img.Pix[o], img.Pix[o+1], img.Pix[o+2], img.Pix[o+3] = uint8(x), uint8(y), uint8(x+y), 255
		}
	}
	return img
}

func pixel(img *image.RGBA, x, y int) [4]uint8 {
	o := img.PixOffset(x, y)
	return [4]uint8{img.Pix[o], img.Pix[o+1], img.Pix[o+2], img.Pix[o+3]}
}

func TestPerspectiveTransformGoldenValues(t *testing.T) {
	t.Parallel()

	unit := paddleocr.Quad{{X: 0, Y: 0}, {X: 1, Y: 0}, {X: 1, Y: 1}, {X: 0, Y: 1}}

	// x' = 2x + 3, y' = 3y + 1 maps the unit square onto (3,1) (5,1) (5,4)
	// (3,4).
	h, err := paddleocr.PerspectiveTransform(unit, paddleocr.Quad{{X: 3, Y: 1}, {X: 5, Y: 1}, {X: 5, Y: 4}, {X: 3, Y: 4}})
	if err != nil {
		t.Fatal(err)
	}
	assertHomography(t, "affine", h, paddleocr.Homography{2, 0, 3, 0, 3, 1, 0, 0, 1})

	// x' = x/(x+1), y' = y/(x+1) maps (0,0) (1,0) (1,1) (0,1) onto (0,0)
	// (0.5,0) (0.5,0.5) (0,1): the matrix [1 0 0; 0 1 0; 1 0 1].
	h, err = paddleocr.PerspectiveTransform(unit, paddleocr.Quad{{X: 0, Y: 0}, {X: 0.5, Y: 0}, {X: 0.5, Y: 0.5}, {X: 0, Y: 1}})
	if err != nil {
		t.Fatal(err)
	}
	assertHomography(t, "projective", h, paddleocr.Homography{1, 0, 0, 0, 1, 0, 1, 0, 1})
	// The midpoint (0.5, 0.5) goes to (0.5/1.5, 0.5/1.5).
	if p := h.Apply(paddleocr.Point{X: 0.5, Y: 0.5}); !nearPoint(p, paddleocr.Point{X: 1.0 / 3, Y: 1.0 / 3}, 1e-12) {
		t.Errorf("Apply(0.5, 0.5) = %v, want (1/3, 1/3)", p)
	}

	if _, err := paddleocr.PerspectiveTransform(unit, paddleocr.Quad{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 2, Y: 2}, {X: 3, Y: 3}}); err == nil {
		t.Error("a quadrilateral on one line was accepted")
	}
}

func assertHomography(t *testing.T, label string, got, want paddleocr.Homography) {
	t.Helper()
	for i := range got {
		if !near(got[i], want[i], 1e-9) {
			t.Errorf("%s: homography = %v, want %v", label, got, want)
			return
		}
	}
}

func TestCropSizeIsTheLongerOfOppositeSides(t *testing.T) {
	t.Parallel()

	// Top 30, bottom 40 (from (0,30) to (40,30)); left 30, right √(10²+30²) ≈ 31.6.
	w, h := paddleocr.CropSize(paddleocr.Quad{{X: 0, Y: 0}, {X: 30, Y: 0}, {X: 40, Y: 30}, {X: 0, Y: 30}})
	if w != 40 || h != 31 {
		t.Errorf("CropSize = %d×%d, want 40×31", w, h)
	}
}

func TestWarpQuadOfAnAxisAlignedBoxIsACopy(t *testing.T) {
	t.Parallel()

	src := gradient(40, 30)
	// A 10×8 box from (10, 5): every output pixel (x, y) reads (10+x, 5+y)
	// exactly, so the crop is a copy.
	crop, err := paddleocr.WarpQuad(src, paddleocr.Quad{{X: 10, Y: 5}, {X: 20, Y: 5}, {X: 20, Y: 13}, {X: 10, Y: 13}})
	if err != nil {
		t.Fatal(err)
	}
	if crop.Rect.Dx() != 10 || crop.Rect.Dy() != 8 {
		t.Fatalf("crop is %v, want 10×8", crop.Rect)
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 10; x++ {
			if got, want := pixel(crop, x, y), pixel(src, 10+x, 5+y); got != want {
				t.Fatalf("crop(%d, %d) = %v, want %v", x, y, got, want)
			}
		}
	}
}

func TestWarpQuadStraightensATurnedBox(t *testing.T) {
	t.Parallel()

	src := gradient(40, 30)
	// The box (20,4) (20,14) (12,14) (12,4) lists its top-left at (20, 4) and
	// runs down: its "top" side is the column x = 20 from y 4 to 14. Output
	// (x, y) reads (20 - y, 4 + x), so the crop is 10×8.
	crop, err := paddleocr.WarpQuad(src, paddleocr.Quad{{X: 20, Y: 4}, {X: 20, Y: 14}, {X: 12, Y: 14}, {X: 12, Y: 4}})
	if err != nil {
		t.Fatal(err)
	}
	if crop.Rect.Dx() != 10 || crop.Rect.Dy() != 8 {
		t.Fatalf("crop is %v, want 10×8", crop.Rect)
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 10; x++ {
			if got, want := pixel(crop, x, y), pixel(src, 20-y, 4+x); got != want {
				t.Fatalf("crop(%d, %d) = %v, want %v", x, y, got, want)
			}
		}
	}
}

func TestWarpQuadInterpolatesAndRepeatsTheEdge(t *testing.T) {
	t.Parallel()

	src := gradient(40, 30)
	// Shifted by half a pixel: output (x, 0) reads (x + 0.5, 0.5), whose red
	// is x + 0.5, rounded half away from zero to x + 1.
	crop, err := paddleocr.WarpQuad(src, paddleocr.Quad{{X: 0.5, Y: 0.5}, {X: 4.5, Y: 0.5}, {X: 4.5, Y: 2.5}, {X: 0.5, Y: 2.5}})
	if err != nil {
		t.Fatal(err)
	}
	for x := 0; x < 4; x++ {
		if got := pixel(crop, x, 0)[0]; got != uint8(x+1) {
			t.Errorf("crop(%d, 0) red = %d, want %d", x, got, x+1)
		}
	}
	// A box hanging off the left edge repeats column 0.
	crop, err = paddleocr.WarpQuad(src, paddleocr.Quad{{X: -5, Y: 0}, {X: 5, Y: 0}, {X: 5, Y: 4}, {X: -5, Y: 4}})
	if err != nil {
		t.Fatal(err)
	}
	for x := 0; x <= 5; x++ {
		if got := pixel(crop, x, 0)[0]; got != 0 {
			t.Errorf("crop(%d, 0) red = %d, want 0 from the repeated edge", x, got)
		}
	}
	if _, err := paddleocr.WarpQuad(src, paddleocr.Quad{{X: 1, Y: 1}, {X: 1, Y: 1}, {X: 1, Y: 1}, {X: 1, Y: 1}}); err == nil {
		t.Error("a box with no area was cropped")
	}
}

func TestRotateTurnsCounterClockwise(t *testing.T) {
	t.Parallel()

	// 3×2:  a b c
	//       d e f
	src := image.NewRGBA(image.Rect(0, 0, 3, 2))
	for i := 0; i < 6; i++ {
		src.Pix[i*4] = uint8('a' + i)
	}
	read := func(img *image.RGBA) string {
		out := ""
		for y := 0; y < img.Rect.Dy(); y++ {
			for x := 0; x < img.Rect.Dx(); x++ {
				out += string(rune(pixel(img, x, y)[0]))
			}
			out += "/"
		}
		return out
	}
	for degrees, want := range map[int]string{
		0:   "abc/def/",
		90:  "cf/be/ad/", // the right column becomes the top row
		180: "fed/cba/",
		270: "da/eb/fc/",
		-90: "da/eb/fc/",
	} {
		got, err := paddleocr.Rotate(src, degrees)
		if err != nil {
			t.Fatalf("Rotate(%d): %v", degrees, err)
		}
		if read(got) != want {
			t.Errorf("Rotate(%d) = %s, want %s", degrees, read(got), want)
		}
	}
	if _, err := paddleocr.Rotate(src, 45); err == nil {
		t.Error("a 45-degree rotation was accepted")
	}

	// A sub-image keeps its parent's stride: a b / d e.
	sub := src.SubImage(image.Rect(0, 0, 2, 2)).(*image.RGBA)
	for degrees, want := range map[int]string{0: "ab/de/", 90: "be/ad/"} {
		got, err := paddleocr.Rotate(sub, degrees)
		if err != nil {
			t.Fatal(err)
		}
		if read(got) != want {
			t.Errorf("Rotate(sub-image, %d) = %s, want %s", degrees, read(got), want)
		}
	}
}
