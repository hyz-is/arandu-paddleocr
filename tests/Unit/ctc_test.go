package unit_test

import (
	"math"
	"slices"
	"testing"

	paddleocr "github.com/kuaa-io/arandu-paddleocr"
)

// steps builds recognizer output in which each step puts p on one class and
// spreads the rest evenly over the others.
func steps(classes int, p float32, best ...int) []float32 {
	out := make([]float32, 0, len(best)*classes)
	rest := (1 - p) / float32(classes-1)
	for _, b := range best {
		for c := 0; c < classes; c++ {
			if c == b {
				out = append(out, p)
			} else {
				out = append(out, rest)
			}
		}
	}
	return out
}

// The vocabulary of the dictionary a, b, a: class 0 blank, 1 "a", 2 "b",
// 3 "a" again, 4 the space.
func vocabulary() paddleocr.Vocabulary {
	return paddleocr.NewVocabulary([]string{"a", "b", "a"}, true)
}

func TestTheVocabularyNumbersClassesFromOne(t *testing.T) {
	t.Parallel()

	v := vocabulary()
	if v.Classes() != 5 {
		t.Fatalf("Classes = %d, want 5", v.Classes())
	}
	for class, want := range []string{"", "a", "b", "a", " "} {
		if got := v.Symbol(class); got != want {
			t.Errorf("Symbol(%d) = %q, want %q", class, got, want)
		}
	}
	if v.Symbol(5) != "" || v.Symbol(-1) != "" {
		t.Error("a class out of range has a symbol")
	}
	if paddleocr.NewVocabulary([]string{"a"}, false).Classes() != 2 {
		t.Error("a vocabulary without the space has the space class")
	}
}

func TestCTCCollapsesRepeatsAndDropsTheBlank(t *testing.T) {
	t.Parallel()

	// a a _ a b b ␣ a′: the repeated a collapses, the blank separates the
	// next a, b b collapses, and a′ (class 3) is kept although it reads like
	// a, because it is another class from the space before it.
	scores := steps(5, 0.9, 1, 1, 0, 1, 2, 2, 4, 3)
	got, err := paddleocr.DecodeCTC(scores, 8, vocabulary(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Text != "aab a" {
		t.Errorf("Text = %q, want %q", got.Text, "aab a")
	}
	if !slices.Equal(got.Classes, []int{1, 1, 2, 4, 3}) {
		t.Errorf("Classes = %v, want [1 1 2 4 3]", got.Classes)
	}
	if !near(got.Confidence, 0.9, 1e-6) {
		t.Errorf("Confidence = %v, want 0.9", got.Confidence)
	}
}

func TestCTCDecodesDuplicateCharactersByIndex(t *testing.T) {
	t.Parallel()

	// Classes 1 and 3 are both "a". Back to back they are two classes, so
	// both are kept: "aa", not the one "a" a lookup by character would give.
	got, err := paddleocr.DecodeCTC(steps(5, 0.8, 1, 3), 2, vocabulary(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Text != "aa" || !slices.Equal(got.Classes, []int{1, 3}) {
		t.Errorf("decoded %q %v, want \"aa\" [1 3]", got.Text, got.Classes)
	}
}

func TestCTCAppliesASoftmaxToScoresThatAreNotProbabilities(t *testing.T) {
	t.Parallel()

	// Logits 0 5 0 0 0: softmax gives e^5 / (e^5 + 4) = 148.4132 / 152.4132
	// = 0.973756 for class 1.
	logits := []float32{0, 5, 0, 0, 0, 0, 0, 5, 0, 0}
	got, err := paddleocr.DecodeCTC(logits, 2, vocabulary(), nil)
	if err != nil {
		t.Fatal(err)
	}
	want := math.Exp(5) / (math.Exp(5) + 4)
	if got.Text != "ab" || !near(got.Confidence, want, 1e-6) {
		t.Errorf("decoded %q at %v, want \"ab\" at %v", got.Text, got.Confidence, want)
	}
}

func TestAMaskLimitsTheClassesThatCompete(t *testing.T) {
	t.Parallel()

	v := vocabulary()
	mask, err := v.Mask("b")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(mask, []bool{true, false, true, false, false}) {
		t.Fatalf("Mask(b) = %v, want blank and b", mask)
	}
	// A step that reads a at 0.6 and b at 0.3 reads b once a is masked, and
	// its confidence is b's 0.3, not renormalized.
	scores := []float32{0.05, 0.6, 0.3, 0.025, 0.025}
	got, err := paddleocr.DecodeCTC(scores, 1, v, mask)
	if err != nil {
		t.Fatal(err)
	}
	if got.Text != "b" || !near(got.Confidence, 0.3, 1e-6) {
		t.Errorf("decoded %q at %v, want \"b\" at 0.3", got.Text, got.Confidence)
	}
	// Without the mask it reads a.
	if got, _ := paddleocr.DecodeCTC(scores, 1, v, nil); got.Text != "a" {
		t.Errorf("unmasked decode = %q, want a", got.Text)
	}

	// Every class carrying a masked character is allowed, duplicates included.
	mask, err = v.Mask("a ")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(mask, []bool{true, true, false, true, true}) {
		t.Errorf("Mask(\"a \") = %v", mask)
	}
	if mask, err := v.Mask(""); mask != nil || err != nil {
		t.Errorf("Mask(\"\") = %v, %v; want no mask", mask, err)
	}
	if _, err := v.Mask("z"); err == nil {
		t.Error("a character with no class was accepted")
	}
}

func TestCTCEdgeCases(t *testing.T) {
	t.Parallel()

	v := vocabulary()
	got, err := paddleocr.DecodeCTC(steps(5, 0.99, 0, 0, 0), 3, v, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Text != "" || got.Confidence != 0 || len(got.Classes) != 0 {
		t.Errorf("all blanks = %+v, want empty text at confidence 0", got)
	}
	if got, err := paddleocr.DecodeCTC(nil, 0, v, nil); err != nil || got.Text != "" {
		t.Errorf("no steps = %+v, %v", got, err)
	}
	if _, err := paddleocr.DecodeCTC(make([]float32, 9), 2, v, nil); err == nil {
		t.Error("scores of the wrong length were accepted")
	}
	if _, err := paddleocr.DecodeCTC(make([]float32, 5), 1, v, []bool{true}); err == nil {
		t.Error("a mask of the wrong length was accepted")
	}
}

func TestTheLatinVocabularyMatchesTheRecognizer(t *testing.T) {
	t.Parallel()

	v, err := paddleocr.RecognizerVocabulary(paddleocr.ModelLatinRecognizer)
	if err != nil {
		t.Fatal(err)
	}
	if v.Classes() != 838 {
		t.Fatalf("Classes = %d, want 838: blank, 836 lines and the space", v.Classes())
	}
	// Line 1 of the dictionary is "0", so class 1 is "0"; the class after the
	// 836th line is the space.
	if v.Symbol(1) != "0" || v.Symbol(11) != "A" || v.Symbol(837) != " " {
		t.Errorf("Symbol(1), (11), (837) = %q %q %q, want \"0\" \"A\" \" \"", v.Symbol(1), v.Symbol(11), v.Symbol(837))
	}
	// "<" is line 480; "Ú" is on lines 89 and 727, and both classes carry it.
	mask, err := v.Mask("<Ú")
	if err != nil {
		t.Fatal(err)
	}
	allowed := []int{}
	for class, ok := range mask {
		if ok {
			allowed = append(allowed, class)
		}
	}
	if !slices.Equal(allowed, []int{0, 89, 480, 727}) {
		t.Errorf("Mask(\"<Ú\") allows %v, want [0 89 480 727]", allowed)
	}

	en, err := paddleocr.RecognizerVocabulary(paddleocr.ModelEnglishRecognizer)
	if err != nil {
		t.Fatal(err)
	}
	if en.Classes() != 438 {
		t.Errorf("the English vocabulary has %d classes, want 438", en.Classes())
	}
	if _, err := paddleocr.RecognizerVocabulary(paddleocr.ModelMobileDetector); err == nil {
		t.Error("a detector has a vocabulary")
	}
}
