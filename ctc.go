package paddleocr

import (
	"fmt"
	"math"
	"strings"
)

// Vocabulary maps a recognizer's output classes onto text.
//
// Class 0 is the CTC blank. Class i, for i from 1 to the dictionary's length,
// is the dictionary's line i-1, and when the model was trained with a space
// the class after the last line is " ". Classes are decoded by index, never by
// looking a character up: a dictionary may hold the same character twice, and
// each occurrence is its own class.
type Vocabulary struct {
	symbols []string // symbols[0] is the blank and is never emitted
}

// NewVocabulary builds the vocabulary of a dictionary, with a trailing space
// class when space is true.
func NewVocabulary(dictionary []string, space bool) Vocabulary {
	symbols := make([]string, 0, len(dictionary)+2)
	symbols = append(symbols, "")
	symbols = append(symbols, dictionary...)
	if space {
		symbols = append(symbols, " ")
	}
	return Vocabulary{symbols: symbols}
}

// Classes is how many classes the recognizer has to output for this
// vocabulary, blank included.
func (v Vocabulary) Classes() int { return len(v.symbols) }

// Symbol is the text of a class; the blank and a class out of range are "".
func (v Vocabulary) Symbol(class int) string {
	if class <= 0 || class >= len(v.symbols) {
		return ""
	}
	return v.symbols[class]
}

// Mask returns which classes may be emitted when the text is known to be
// written in charset: every class whose symbol is one of charset's
// characters, and the blank. An empty charset allows everything and returns
// nil. A character of charset no class carries is refused, because a mask
// that silently lacks it can never read it.
func (v Vocabulary) Mask(charset string) ([]bool, error) {
	if charset == "" {
		return nil, nil
	}
	allowed := make([]bool, len(v.symbols))
	allowed[0] = true
	for _, r := range charset {
		found := false
		for class := 1; class < len(v.symbols); class++ {
			if v.symbols[class] == string(r) {
				allowed[class] = true
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("paddleocr: the charset has %q, which the recognizer's dictionary does not", r)
		}
	}
	return allowed, nil
}

// Decoded is the text CTC decoding read from one sequence.
type Decoded struct {
	// Text is the characters of the kept classes, in order.
	Text string
	// Confidence is the mean probability of the kept steps, 0 when none was
	// kept.
	Confidence float64
	// Classes is the class of every kept step.
	Classes []int
}

// DecodeCTC decodes one sequence of recognizer scores, steps × classes values
// row by row, with CTC greedy decoding: the best class of each step, repeats
// of the previous step's class collapsed, and the blank dropped.
//
// Each step is used as probabilities when it already is one -- every value in
// [0, 1], summing to 1 -- and passed through a softmax first otherwise. When
// allowed is not nil, only the classes it marks compete for each step, and the
// confidence of the step is the probability of the class chosen among them,
// not renormalized: a step the model would rather have read as a class
// outside the mask scores low.
func DecodeCTC(scores []float32, steps int, v Vocabulary, allowed []bool) (Decoded, error) {
	classes := v.Classes()
	if steps < 0 || len(scores) != steps*classes {
		return Decoded{}, fmt.Errorf("paddleocr: %d steps of %d classes need %d scores and there are %d", steps, classes, steps*classes, len(scores))
	}
	if allowed != nil && len(allowed) != classes {
		return Decoded{}, fmt.Errorf("paddleocr: the mask has %d classes and the vocabulary %d", len(allowed), classes)
	}

	var text strings.Builder
	out := Decoded{Classes: []int{}}
	total := 0.0
	previous := -1
	probabilities := make([]float64, classes)
	for s := 0; s < steps; s++ {
		toProbabilities(scores[s*classes:(s+1)*classes], probabilities)
		best := -1
		for c, p := range probabilities {
			if allowed != nil && !allowed[c] {
				continue
			}
			if best < 0 || p > probabilities[best] {
				best = c
			}
		}
		if best > 0 && best != previous {
			out.Classes = append(out.Classes, best)
			text.WriteString(v.symbols[best])
			total += probabilities[best]
		}
		previous = best
	}
	out.Text = text.String()
	if len(out.Classes) > 0 {
		out.Confidence = total / float64(len(out.Classes))
	}
	return out, nil
}

// toProbabilities copies one step into dst, through a softmax unless it
// already is a probability distribution.
func toProbabilities(step []float32, dst []float64) {
	sum, isDistribution := 0.0, true
	for i, s := range step {
		v := float64(s)
		dst[i] = v
		sum += v
		if v < 0 || v > 1 || math.IsNaN(v) {
			isDistribution = false
		}
	}
	if isDistribution && math.Abs(sum-1) <= 1e-3 {
		return
	}
	top := math.Inf(-1)
	for _, v := range dst {
		top = math.Max(top, v)
	}
	sum = 0
	for i, v := range dst {
		dst[i] = math.Exp(v - top)
		sum += dst[i]
	}
	for i := range dst {
		dst[i] /= sum
	}
}
