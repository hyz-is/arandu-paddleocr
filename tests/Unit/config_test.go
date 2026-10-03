package unit_test

import (
	"math"
	"strings"
	"testing"

	paddleocr "github.com/hyz-is/arandu-paddleocr"
)

func TestConfigValidateRefusesWhatCannotWork(t *testing.T) {
	t.Parallel()

	valid := paddleocr.Config{ModelsDir: "models"}
	if err := valid.Validate(); err != nil {
		t.Fatalf("a minimal config is refused: %v", err)
	}
	for name, tc := range map[string]struct {
		mutate func(*paddleocr.Config)
		field  string
	}{
		"no models dir":          {func(c *paddleocr.Config) { c.ModelsDir = "" }, "ModelsDir"},
		"unknown detector":       {func(c *paddleocr.Config) { c.Detector = "tiny" }, "Detector"},
		"unknown recognizer":     {func(c *paddleocr.Config) { c.Recognizer = "pt" }, "Recognizer"},
		"limit too small":        {func(c *paddleocr.Config) { c.LimitSideLen = 31 }, "LimitSideLen"},
		"limit too large":        {func(c *paddleocr.Config) { c.LimitSideLen = 4001 }, "LimitSideLen"},
		"thresh above one":       {func(c *paddleocr.Config) { c.Thresh = 1.1 }, "Thresh"},
		"box thresh negative":    {func(c *paddleocr.Config) { c.BoxThresh = -0.1 }, "BoxThresh"},
		"box thresh NaN":         {func(c *paddleocr.Config) { c.BoxThresh = math.NaN() }, "BoxThresh"},
		"unclip too large":       {func(c *paddleocr.Config) { c.UnclipRatio = 11 }, "UnclipRatio"},
		"negative candidates":    {func(c *paddleocr.Config) { c.MaxCandidates = -1 }, "MaxCandidates"},
		"batch too large":        {func(c *paddleocr.Config) { c.RecBatchSize = 65 }, "RecBatchSize"},
		"max width too small":    {func(c *paddleocr.Config) { c.RecMaxWidth = 100 }, "RecMaxWidth"},
		"textline threshold > 1": {func(c *paddleocr.Config) { c.TextlineThreshold = 2 }, "TextlineThreshold"},
	} {
		c := valid
		tc.mutate(&c)
		err := c.Validate()
		if err == nil {
			t.Errorf("%s: accepted", name)
			continue
		}
		if !strings.Contains(err.Error(), tc.field) {
			t.Errorf("%s: %v does not name %s", name, err, tc.field)
		}
	}
}

func TestConfigDefaultsFillOnlyZeroFields(t *testing.T) {
	t.Parallel()

	got := paddleocr.Config{ModelsDir: "m", LimitSideLen: 1280, Detector: paddleocr.DetectorServer}.WithDefaults()
	if got.LimitSideLen != 1280 || got.Detector != paddleocr.DetectorServer {
		t.Errorf("WithDefaults overwrote a set field: %+v", got)
	}
	if got.Recognizer != paddleocr.RecognizerLatin || got.Thresh != paddleocr.DefaultThresh ||
		got.BoxThresh != paddleocr.DefaultBoxThresh || got.UnclipRatio != paddleocr.DefaultUnclipRatio ||
		got.MaxCandidates != paddleocr.DefaultMaxCandidates || got.RecBatchSize != paddleocr.DefaultRecBatchSize ||
		got.RecMaxWidth != paddleocr.DefaultRecMaxWidth || got.TextlineThreshold != paddleocr.DefaultTextlineThreshold {
		t.Errorf("WithDefaults left a zero field: %+v", got)
	}
	if got.UseDocOrientation || got.UseTextlineOrientation {
		t.Error("an orientation stage was switched on by default")
	}
}

func TestTheDetectorInputSize(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ w, h, limit, ww, wh int }{
		// 851/32 = 26.6 → 27·32; 472/32 = 14.75 → 15·32. Not scaled: under 960.
		{851, 472, 960, 864, 480},
		// Scaled by 960/2000 = 0.48: 960 and 480, already multiples of 32.
		{2000, 1000, 960, 960, 480},
		// A tiny image is never smaller than 32.
		{10, 3, 960, 32, 32},
		// 48/32 = 1.5 rounds to even, 2; 80/32 = 2.5 rounds to even, 2.
		{48, 80, 960, 64, 64},
		// int(1000·0.96) = 960 → 960; int(700·0.96) = 672 → 21·32 = 672.
		{1000, 700, 960, 960, 672},
	} {
		w, h := paddleocr.DetectorInputSize(tc.w, tc.h, tc.limit)
		if w != tc.ww || h != tc.wh {
			t.Errorf("DetectorInputSize(%d, %d, %d) = %d×%d, want %d×%d", tc.w, tc.h, tc.limit, w, h, tc.ww, tc.wh)
		}
	}
}

func TestTheRecognizerInputWidth(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ w, h, max, want int }{
		{100, 50, 3200, 96},    // ceil(48·2) = 96
		{101, 50, 3200, 97},    // ceil(96.96) = 97
		{1000, 10, 3200, 3200}, // 4800, capped
		{1, 1000, 3200, 1},     // ceil(0.048) = 1
		{640, 48, 320, 320},    // 640, capped at the batch width
	} {
		if got := paddleocr.RecognizerInputWidth(tc.w, tc.h, tc.max); got != tc.want {
			t.Errorf("RecognizerInputWidth(%d, %d, %d) = %d, want %d", tc.w, tc.h, tc.max, got, tc.want)
		}
	}
}
