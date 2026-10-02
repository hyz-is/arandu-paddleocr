package paddleocr

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"sort"
	"time"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// Options are what one Recognize call may narrow.
type Options struct {
	// Charset, when not empty, is every character the text may contain, such
	// as "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<" for a machine-readable zone.
	// The recognizer then chooses among those characters only. A character
	// the recognizer's dictionary lacks is refused.
	Charset string
}

// Page is the text read on one image.
type Page struct {
	// Width and Height are the size of the image as it was given.
	Width, Height int
	// Rotation is how far the image was turned counter-clockwise, in degrees,
	// to stand it upright before detection: 0, 90, 180 or 270. It is always 0
	// when document orientation is off.
	Rotation int
	// Lines are the lines of text in reading order.
	Lines []Line
	// Models identifies every model that contributed to the page, in the
	// order they ran.
	Models []onnx.ModelInfo
	// Timings is how long each stage took.
	Timings Timings
}

// Line is one line of text.
type Line struct {
	// Text is what the recognizer read.
	Text string
	// Confidence is the recognizer's mean probability over the characters it
	// kept, between 0 and 1.
	Confidence float64
	// DetectionScore is the detector's mean probability inside the line's
	// box, between 0 and 1.
	DetectionScore float64
	// Box is the line's quadrilateral on the image as it was given: the
	// text's top-left, top-right, bottom-right and bottom-left corners, so on
	// a page that was turned the first corner is not the image's top-left.
	Box Quad
	// Order is the line's position in reading order, from 0.
	Order int
	// Flipped is whether the line was read upside down and turned by 180
	// degrees before recognition.
	Flipped bool
}

// Timings is how long each stage of one Recognize call took.
type Timings struct {
	// DocOrientation is classifying and turning the page.
	DocOrientation time.Duration
	// Detection is finding the lines, post-processing included.
	Detection time.Duration
	// TextlineOrientation is classifying and turning the lines.
	TextlineOrientation time.Duration
	// Recognition is cropping and reading the lines.
	Recognition time.Duration
	// Total is the whole call.
	Total time.Duration
}

// The normalizations each model was trained with.
var (
	// imageNet is the classifiers': scale 1/255, ImageNet mean and standard
	// deviation, channels red, green, blue.
	imageNet = onnx.Normalization{Mean: [3]float32{0.485, 0.456, 0.406}, Std: [3]float32{0.229, 0.224, 0.225}}
	// detectorNorm is the detector's: the same constants applied to the
	// channels in blue, green, red order, which is how its inference.yml
	// decodes the image.
	detectorNorm = onnx.Normalization{Mean: [3]float32{0.485, 0.456, 0.406}, Std: [3]float32{0.229, 0.224, 0.225}, BGR: true}
	// recognizerNorm maps 0..255 onto -1..1, blue, green, red.
	recognizerNorm = onnx.Normalization{Mean: [3]float32{0.5, 0.5, 0.5}, Std: [3]float32{0.5, 0.5, 0.5}, BGR: true}
)

const (
	// recHeight is the height every line is resized to.
	recHeight = 48
	// recMinWidth is the narrowest padded width of a recognizer batch: the
	// 48×320 shape of the model's inference.yml.
	recMinWidth = 320
	// minCropSide drops a detected box whose sides are not longer than this,
	// as PaddleOCR's filter_tag_det_res does.
	minCropSide = 3
	// tallCrop is the height-to-width ratio from which a crop is read as
	// vertical text and turned 90 degrees.
	tallCrop = 1.5
)

// DetectorInputSize is the size an image of width × height is resized to for
// the detector: scaled down so its longer side is at most limit, never up,
// then each side rounded to the nearest multiple of 32 and at least 32. Halves
// round to even, as Python's round does.
func DetectorInputSize(width, height, limit int) (int, int) {
	ratio := 1.0
	if longest := max(width, height); longest > limit {
		ratio = float64(limit) / float64(longest)
	}
	round32 := func(side int) int {
		scaled := int(float64(side) * ratio)
		return max(int(math.RoundToEven(float64(scaled)/32))*32, 32)
	}
	return round32(width), round32(height)
}

// RecognizerInputWidth is the width a line image of width × height is
// resized to at a height of 48: ceil(48 × width / height), at most maxWidth
// and at least 1.
func RecognizerInputWidth(width, height, maxWidth int) int {
	w := int(math.Ceil(recHeight * float64(width) / float64(height)))
	return max(1, min(w, maxWidth))
}

// Recognize reads the text on an image.
//
// The image is turned upright when document orientation is on, its lines
// are found and post-processed, each line is cut out and straightened,
// turned when line orientation is on and it reads upside down, and read. The
// boxes come back in the image's own coordinates, in reading order.
//
// It is safe for concurrent use. The context is checked between stages and
// handed to every model run.
func (m *Module) Recognize(ctx context.Context, img image.Image, opts Options) (Page, error) {
	started := time.Now()
	m.mu.RLock()
	defer m.mu.RUnlock()
	if err := m.usable(); err != nil {
		return Page{}, err
	}
	if img == nil || img.Bounds().Empty() {
		return Page{}, errors.New("paddleocr: the image is empty")
	}
	mask, err := m.vocabulary.Mask(opts.Charset)
	if err != nil {
		return Page{}, err
	}

	source, err := onnx.Crop(img, img.Bounds())
	if err != nil {
		return Page{}, err
	}
	page := Page{Width: source.Rect.Dx(), Height: source.Rect.Dy(), Lines: []Line{}, Models: []onnx.ModelInfo{}}

	upright := source
	if m.docOrient != nil {
		t := time.Now()
		angle, info, err := m.pageAngle(ctx, source)
		if err != nil {
			return Page{}, err
		}
		page.Models = append(page.Models, info)
		if angle != 0 {
			if upright, err = Rotate(source, angle); err != nil {
				return Page{}, err
			}
		}
		page.Rotation = angle
		page.Timings.DocOrientation = time.Since(t)
	}

	t := time.Now()
	boxes, info, err := m.detect(ctx, upright)
	if err != nil {
		return Page{}, err
	}
	page.Models = append(page.Models, info)
	page.Timings.Detection = time.Since(t)

	t = time.Now()
	crops := make([]*image.RGBA, len(boxes))
	for i, box := range boxes {
		crop, err := WarpQuad(upright, box.Box)
		if err != nil {
			return Page{}, fmt.Errorf("paddleocr: cropping line %d: %w", i, err)
		}
		if w, h := crop.Rect.Dx(), crop.Rect.Dy(); float64(h)/float64(w) >= tallCrop {
			if crop, err = Rotate(crop, 90); err != nil {
				return Page{}, err
			}
		}
		crops[i] = crop
	}
	cropping := time.Since(t)

	flipped := make([]bool, len(crops))
	if m.textline != nil && len(crops) > 0 {
		t := time.Now()
		if flipped, info, err = m.lineFlips(ctx, crops); err != nil {
			return Page{}, err
		}
		page.Models = append(page.Models, info)
		for i, flip := range flipped {
			if flip {
				if crops[i], err = Rotate(crops[i], 180); err != nil {
					return Page{}, err
				}
			}
		}
		page.Timings.TextlineOrientation = time.Since(t)
	}

	t = time.Now()
	if len(crops) > 0 {
		texts, info, err := m.read(ctx, crops, mask)
		if err != nil {
			return Page{}, err
		}
		page.Models = append(page.Models, info)
		for i, box := range boxes {
			var corners Quad
			for k, p := range box.Box {
				corners[k] = unrotatePoint(p, page.Rotation, page.Width, page.Height)
			}
			page.Lines = append(page.Lines, Line{
				Text:           texts[i].Text,
				Confidence:     texts[i].Confidence,
				DetectionScore: box.Score,
				Box:            corners,
				Order:          i,
				Flipped:        flipped[i],
			})
		}
	}
	page.Timings.Recognition = cropping + time.Since(t)
	page.Timings.Total = time.Since(started)
	return page, nil
}

// pageAngle classifies how far the page is turned, and returns how far it has
// to be turned counter-clockwise to stand upright.
func (m *Module) pageAngle(ctx context.Context, page *image.RGBA) (int, onnx.ModelInfo, error) {
	w, h := page.Rect.Dx(), page.Rect.Dy()
	scale := 256 / float64(min(w, h))
	rw := max(224, int(math.RoundToEven(float64(w)*scale)))
	rh := max(224, int(math.RoundToEven(float64(h)*scale)))
	resized, err := onnx.Resize(page, rw, rh)
	if err != nil {
		return 0, onnx.ModelInfo{}, err
	}
	x0, y0 := (rw-224)/2, (rh-224)/2
	square, err := onnx.Crop(resized, image.Rect(x0, y0, x0+224, y0+224))
	if err != nil {
		return 0, onnx.ModelInfo{}, err
	}
	input, err := onnx.NCHW(square, imageNet)
	if err != nil {
		return 0, onnx.ModelInfo{}, err
	}
	result, err := m.docOrient.Run(ctx, map[string]onnx.Tensor{"x": input})
	if err != nil {
		return 0, onnx.ModelInfo{}, err
	}
	scores := result.Outputs["fetch_name_0"].Float32
	if len(scores) != 4 {
		return 0, onnx.ModelInfo{}, fmt.Errorf("paddleocr: %s returned %d scores, want 4", result.Model.ID, len(scores))
	}
	return argmax(scores) * 90, result.Model, nil
}

// detect finds the lines on an upright page and returns their boxes on it, in
// reading order.
func (m *Module) detect(ctx context.Context, page *image.RGBA) ([]TextBox, onnx.ModelInfo, error) {
	w, h := page.Rect.Dx(), page.Rect.Dy()
	rw, rh := DetectorInputSize(w, h, m.cfg.LimitSideLen)
	resized, err := onnx.Resize(page, rw, rh)
	if err != nil {
		return nil, onnx.ModelInfo{}, err
	}
	input, err := onnx.NCHW(resized, detectorNorm)
	if err != nil {
		return nil, onnx.ModelInfo{}, err
	}
	result, err := m.detector.Run(ctx, map[string]onnx.Tensor{"x": input})
	if err != nil {
		return nil, onnx.ModelInfo{}, err
	}
	out := result.Outputs["fetch_name_0"]
	if len(out.Shape) != 4 || out.Shape[2] != int64(rh) || out.Shape[3] != int64(rw) {
		return nil, onnx.ModelInfo{}, fmt.Errorf("paddleocr: %s returned a map of shape %v for a %d×%d input", result.Model.ID, out.Shape, rw, rh)
	}
	found, err := DBPostProcess(ProbabilityMap{Width: rw, Height: rh, Data: out.Float32}, w, h, DBParams{
		Thresh:        m.cfg.Thresh,
		BoxThresh:     m.cfg.BoxThresh,
		UnclipRatio:   m.cfg.UnclipRatio,
		MaxCandidates: m.cfg.MaxCandidates,
		MinSize:       3,
	})
	if err != nil {
		return nil, onnx.ModelInfo{}, err
	}

	kept := make([]TextBox, 0, len(found))
	for _, box := range found {
		q := OrderClockwise(box.Box)
		for i := range q {
			q[i].X = clampFloat(q[i].X, 0, float64(w-1))
			q[i].Y = clampFloat(q[i].Y, 0, float64(h-1))
		}
		if int(math.Hypot(q[1].X-q[0].X, q[1].Y-q[0].Y)) <= minCropSide ||
			int(math.Hypot(q[3].X-q[0].X, q[3].Y-q[0].Y)) <= minCropSide {
			continue
		}
		kept = append(kept, TextBox{Box: q, Score: box.Score})
	}
	quads := make([]Quad, len(kept))
	for i, box := range kept {
		quads[i] = box.Box
	}
	ordered := make([]TextBox, len(kept))
	for i, index := range ReadingOrder(quads, DefaultReadingOrderTolerance) {
		ordered[i] = kept[index]
	}
	return ordered, result.Model, nil
}

// lineFlips classifies every line as upright or upside down, and reports the
// ones whose probability of 180 degrees reaches the configured threshold.
func (m *Module) lineFlips(ctx context.Context, crops []*image.RGBA) ([]bool, onnx.ModelInfo, error) {
	const width, height = 160, 80
	flips := make([]bool, len(crops))
	var info onnx.ModelInfo
	for start := 0; start < len(crops); start += m.cfg.RecBatchSize {
		end := min(start+m.cfg.RecBatchSize, len(crops))
		data := make([]float32, 0, (end-start)*3*width*height)
		for _, crop := range crops[start:end] {
			resized, err := onnx.Resize(crop, width, height)
			if err != nil {
				return nil, onnx.ModelInfo{}, err
			}
			t, err := onnx.NCHW(resized, imageNet)
			if err != nil {
				return nil, onnx.ModelInfo{}, err
			}
			data = append(data, t.Float32...)
		}
		result, err := m.textline.Run(ctx, map[string]onnx.Tensor{
			"x": {Shape: []int64{int64(end - start), 3, height, width}, Float32: data},
		})
		if err != nil {
			return nil, onnx.ModelInfo{}, err
		}
		info = result.Model
		scores := result.Outputs["fetch_name_0"].Float32
		if len(scores) != 2*(end-start) {
			return nil, onnx.ModelInfo{}, fmt.Errorf("paddleocr: %s returned %d scores for %d lines", info.ID, len(scores), end-start)
		}
		for i := start; i < end; i++ {
			flips[i] = float64(scores[2*(i-start)+1]) >= m.cfg.TextlineThreshold
		}
	}
	return flips, info, nil
}

// read recognizes every line, batching lines of similar aspect ratio
// together so that little of each batch is padding.
func (m *Module) read(ctx context.Context, crops []*image.RGBA, mask []bool) ([]Decoded, onnx.ModelInfo, error) {
	ratios := make([]float64, len(crops))
	order := make([]int, len(crops))
	for i, crop := range crops {
		ratios[i] = float64(crop.Rect.Dx()) / float64(crop.Rect.Dy())
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return ratios[order[a]] < ratios[order[b]] })

	out := make([]Decoded, len(crops))
	var info onnx.ModelInfo
	classes := m.vocabulary.Classes()
	for start := 0; start < len(order); start += m.cfg.RecBatchSize {
		batch := order[start:min(start+m.cfg.RecBatchSize, len(order))]
		widest := float64(recMinWidth) / recHeight
		for _, i := range batch {
			widest = math.Max(widest, ratios[i])
		}
		batchWidth := min(int(recHeight*widest), m.cfg.RecMaxWidth)

		plane := recHeight * batchWidth
		data := make([]float32, len(batch)*3*plane)
		for b, i := range batch {
			crop := crops[i]
			w := RecognizerInputWidth(crop.Rect.Dx(), crop.Rect.Dy(), batchWidth)
			resized, err := onnx.Resize(crop, w, recHeight)
			if err != nil {
				return nil, onnx.ModelInfo{}, err
			}
			t, err := onnx.NCHW(resized, recognizerNorm)
			if err != nil {
				return nil, onnx.ModelInfo{}, err
			}
			// Right-pad with zeros to the batch's width, channel by channel.
			for c := 0; c < 3; c++ {
				for y := 0; y < recHeight; y++ {
					copy(data[b*3*plane+c*plane+y*batchWidth:], t.Float32[c*recHeight*w+y*w:c*recHeight*w+(y+1)*w])
				}
			}
		}
		result, err := m.reader.Run(ctx, map[string]onnx.Tensor{
			"x": {Shape: []int64{int64(len(batch)), 3, recHeight, int64(batchWidth)}, Float32: data},
		})
		if err != nil {
			return nil, onnx.ModelInfo{}, err
		}
		info = result.Model
		scores := result.Outputs["fetch_name_0"]
		if len(scores.Shape) != 3 || scores.Shape[0] != int64(len(batch)) || scores.Shape[2] != int64(classes) {
			return nil, onnx.ModelInfo{}, fmt.Errorf("paddleocr: %s returned shape %v for %d lines of %d classes", info.ID, scores.Shape, len(batch), classes)
		}
		steps := int(scores.Shape[1])
		for b, i := range batch {
			decoded, err := DecodeCTC(scores.Float32[b*steps*classes:(b+1)*steps*classes], steps, m.vocabulary, mask)
			if err != nil {
				return nil, onnx.ModelInfo{}, err
			}
			out[i] = decoded
		}
	}
	return out, info, nil
}

// argmax is the index of the largest value, the first of equal ones.
func argmax(values []float32) int {
	best := 0
	for i, v := range values {
		if v > values[best] {
			best = i
		}
	}
	return best
}
