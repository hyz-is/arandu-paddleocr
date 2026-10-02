package paddleocr

import (
	"fmt"
	"math"
)

// The detectors a Config may name.
const (
	// DetectorMobile is PP-OCRv5_mobile_det.
	DetectorMobile = "mobile"
	// DetectorServer is PP-OCRv5_server_det.
	DetectorServer = "server"
)

// The recognizers a Config may name.
const (
	// RecognizerLatin is latin_PP-OCRv5_mobile_rec, which reads accented
	// Latin script.
	RecognizerLatin = "latin"
	// RecognizerEnglish is en_PP-OCRv5_mobile_rec, which reads unaccented
	// Latin script.
	RecognizerEnglish = "en"
)

// The defaults Config falls back to for a zero value.
const (
	// DefaultLimitSideLen is the longest side the detector sees.
	DefaultLimitSideLen = 960
	// DefaultThresh is the probability a pixel needs to count as text.
	DefaultThresh = 0.3
	// DefaultBoxThresh is the mean probability a box needs to be kept.
	DefaultBoxThresh = 0.6
	// DefaultUnclipRatio is how far a detected box is grown back.
	DefaultUnclipRatio = 1.5
	// DefaultMaxCandidates is how many text regions a page may have examined.
	DefaultMaxCandidates = 1000
	// DefaultRecBatchSize is how many lines are recognized in one run.
	DefaultRecBatchSize = 6
	// DefaultRecMaxWidth is the widest line image the recognizer is given, in
	// pixels at a height of 48.
	DefaultRecMaxWidth = 3200
	// DefaultTextlineThreshold is the probability of 180 degrees a line needs
	// before it is turned.
	DefaultTextlineThreshold = 0.9
	// DefaultReadingOrderTolerance is how far apart vertically, in pixels, two
	// boxes may start and still be read as one row.
	DefaultReadingOrderTolerance = 10
)

// Config is what the application passes when it wires this package.
//
// Every value comes from the application's own typed configuration. This
// package reads no environment variable. A numeric field left at zero takes
// the default named beside it.
type Config struct {
	// ModelsDir is the directory the model files are in, laid out the way
	// their manifests' file fields say: one directory per model, named after
	// its source repository. cmd/paddleocr-models fills one.
	ModelsDir string

	// Detector is DetectorMobile (the default) or DetectorServer.
	Detector string
	// Recognizer is RecognizerLatin (the default) or RecognizerEnglish.
	Recognizer string

	// UseDocOrientation classifies the page as turned by 0, 90, 180 or 270
	// degrees and turns it upright before detection.
	UseDocOrientation bool
	// UseTextlineOrientation classifies every line as upright or upside down
	// and turns the upside-down ones before recognition.
	UseTextlineOrientation bool

	// LimitSideLen caps the longer side of the image the detector sees; a
	// larger image is scaled down to it. Default DefaultLimitSideLen, between
	// 32 and 4000.
	LimitSideLen int
	// Thresh is the probability a pixel of the detector's map needs to count
	// as text. Default DefaultThresh.
	Thresh float64
	// BoxThresh is the mean probability a detected box needs to be kept.
	// Default DefaultBoxThresh.
	BoxThresh float64
	// UnclipRatio is how far a detected box is grown back. Default
	// DefaultUnclipRatio, at most 10.
	UnclipRatio float64
	// MaxCandidates caps the text regions examined on one page. Default
	// DefaultMaxCandidates, at most 100000.
	MaxCandidates int

	// RecBatchSize is how many lines are recognized in one run. Default
	// DefaultRecBatchSize, at most 64.
	RecBatchSize int
	// RecMaxWidth is the widest line image the recognizer is given; a longer
	// line is squeezed into it. Default DefaultRecMaxWidth, between 320 and
	// 8000.
	RecMaxWidth int

	// TextlineThreshold is the probability of 180 degrees a line needs before
	// it is turned. Default DefaultTextlineThreshold.
	TextlineThreshold float64
}

// Validate reports what the configuration cannot be used with. New calls it,
// so a setting that cannot work fails where it is wired.
func (c Config) Validate() error {
	if c.ModelsDir == "" {
		return fmt.Errorf("paddleocr: Config.ModelsDir is required: it is the directory the model files are in")
	}
	switch c.Detector {
	case "", DetectorMobile, DetectorServer:
	default:
		return fmt.Errorf("paddleocr: Config.Detector is %q, and has to be %q or %q", c.Detector, DetectorMobile, DetectorServer)
	}
	switch c.Recognizer {
	case "", RecognizerLatin, RecognizerEnglish:
	default:
		return fmt.Errorf("paddleocr: Config.Recognizer is %q, and has to be %q or %q", c.Recognizer, RecognizerLatin, RecognizerEnglish)
	}
	if c.LimitSideLen != 0 && (c.LimitSideLen < 32 || c.LimitSideLen > 4000) {
		return fmt.Errorf("paddleocr: Config.LimitSideLen is %d, and has to be between 32 and 4000, or 0 for %d", c.LimitSideLen, DefaultLimitSideLen)
	}
	for _, unit := range []struct {
		name  string
		value float64
	}{
		{"Thresh", c.Thresh},
		{"BoxThresh", c.BoxThresh},
		{"TextlineThreshold", c.TextlineThreshold},
	} {
		if math.IsNaN(unit.value) || unit.value < 0 || unit.value > 1 {
			return fmt.Errorf("paddleocr: Config.%s is %v, and has to be between 0 and 1", unit.name, unit.value)
		}
	}
	if math.IsNaN(c.UnclipRatio) || c.UnclipRatio < 0 || c.UnclipRatio > 10 {
		return fmt.Errorf("paddleocr: Config.UnclipRatio is %v, and has to be between 0 and 10", c.UnclipRatio)
	}
	if c.MaxCandidates < 0 || c.MaxCandidates > 100000 {
		return fmt.Errorf("paddleocr: Config.MaxCandidates is %d, and has to be between 0 and 100000", c.MaxCandidates)
	}
	if c.RecBatchSize < 0 || c.RecBatchSize > 64 {
		return fmt.Errorf("paddleocr: Config.RecBatchSize is %d, and has to be between 0 and 64", c.RecBatchSize)
	}
	if c.RecMaxWidth != 0 && (c.RecMaxWidth < 320 || c.RecMaxWidth > 8000) {
		return fmt.Errorf("paddleocr: Config.RecMaxWidth is %d, and has to be between 320 and 8000, or 0 for %d", c.RecMaxWidth, DefaultRecMaxWidth)
	}
	return nil
}

// WithDefaults returns the configuration with every zero field set to its
// default. It does not validate.
func (c Config) WithDefaults() Config {
	if c.Detector == "" {
		c.Detector = DetectorMobile
	}
	if c.Recognizer == "" {
		c.Recognizer = RecognizerLatin
	}
	if c.LimitSideLen == 0 {
		c.LimitSideLen = DefaultLimitSideLen
	}
	if c.Thresh == 0 {
		c.Thresh = DefaultThresh
	}
	if c.BoxThresh == 0 {
		c.BoxThresh = DefaultBoxThresh
	}
	if c.UnclipRatio == 0 {
		c.UnclipRatio = DefaultUnclipRatio
	}
	if c.MaxCandidates == 0 {
		c.MaxCandidates = DefaultMaxCandidates
	}
	if c.RecBatchSize == 0 {
		c.RecBatchSize = DefaultRecBatchSize
	}
	if c.RecMaxWidth == 0 {
		c.RecMaxWidth = DefaultRecMaxWidth
	}
	if c.TextlineThreshold == 0 {
		c.TextlineThreshold = DefaultTextlineThreshold
	}
	return c
}

// detectorModel is the manifest id of the configured detector.
func (c Config) detectorModel() string {
	if c.Detector == DetectorServer {
		return ModelServerDetector
	}
	return ModelMobileDetector
}

// recognizerModel is the manifest id of the configured recognizer.
func (c Config) recognizerModel() string {
	if c.Recognizer == RecognizerEnglish {
		return ModelEnglishRecognizer
	}
	return ModelLatinRecognizer
}
