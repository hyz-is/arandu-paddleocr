// Package paddleocr is an Arandu module that reads the text on an image with
// the PaddleOCR PP-OCRv5 models, run through ONNX Runtime by arandu-onnx.
//
// It finds the lines of text, straightens them, reads them and returns each
// with its box, its confidence and the models that produced it. Everything
// around the models -- resizing, DB post-processing, the perspective crop, CTC
// decoding -- is Go. It owns no table, serves no route and decides nothing:
// what the text means is the application's business.
//
// An application builds it once in bootstrap/app.go, after arandu-onnx, and
// registers it there. There is no service provider, no container and no
// discovery.
package paddleocr

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/arandu-io/framework/foundation"
	fhttp "github.com/arandu-io/framework/http"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// The errors a Module reports about its own life. Each is wrapped with the
// detail of the case, so callers test for them with errors.Is.
var (
	// ErrNotBooted is a module used before Boot opened its models.
	ErrNotBooted = errors.New("paddleocr: module not booted")
	// ErrClosed is a module used after Close.
	ErrClosed = errors.New("paddleocr: module closed")
)

// moduleState is where a Module is in its life.
type moduleState int

const (
	stateNew moduleState = iota
	stateReady
	stateClosed
)

// Module is what the application registers, and what reads text.
//
// Besides foundation.Module it implements foundation.Bootable, which opens the
// models, foundation.Health, which reports whether they are open, and
// foundation.Closable, which closes them. Recognize is safe for concurrent
// use; Close waits for the calls in flight.
type Module struct {
	cfg        Config
	runtime    *onnx.Runtime
	manifests  map[string]onnx.Manifest
	vocabulary Vocabulary

	mu        sync.RWMutex
	state     moduleState
	detector  *onnx.Session
	reader    *onnx.Session
	textline  *onnx.Session
	docOrient *onnx.Session
}

// Compile-time proof that the module honors the contracts it claims.
var (
	_ foundation.Module   = (*Module)(nil)
	_ foundation.Bootable = (*Module)(nil)
	_ foundation.Health   = (*Module)(nil)
	_ foundation.Closable = (*Module)(nil)
)

// New returns the module, or the reason it cannot be built.
//
// The runtime is the one arandu-onnx's module hands out. New validates the
// configuration and the embedded manifests and dictionary, and opens nothing:
// Boot does, so building the module in a test or a command that never boots
// it needs neither ONNX Runtime nor the model files.
func New(cfg Config, runtime *onnx.Runtime) (*Module, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if runtime == nil {
		return nil, errors.New("paddleocr: New needs the ONNX runtime, from the arandu-onnx module's Runtime()")
	}
	cfg = cfg.WithDefaults()

	all, err := Manifests()
	if err != nil {
		return nil, err
	}
	manifests := make(map[string]onnx.Manifest, len(all))
	for _, m := range all {
		manifests[m.ID] = m
	}
	vocabulary, err := RecognizerVocabulary(cfg.recognizerModel())
	if err != nil {
		return nil, err
	}
	reader := manifests[cfg.recognizerModel()]
	if classes := reader.Outputs[0].Shape[2]; int64(vocabulary.Classes()) != classes {
		return nil, fmt.Errorf("paddleocr: %s outputs %d classes and its dictionary makes %d", reader.ID, classes, vocabulary.Classes())
	}
	return &Module{cfg: cfg, runtime: runtime, manifests: manifests, vocabulary: vocabulary}, nil
}

// Config returns the configuration the module uses, with defaults filled in.
func (m *Module) Config() Config { return m.cfg }

// Vocabulary returns the configured recognizer's vocabulary, which is what
// Options.Charset is checked against.
func (m *Module) Vocabulary() Vocabulary { return m.vocabulary }

// Name is the module identifier: a lowercase slug, stable, no spaces.
func (m *Module) Name() string { return "paddleocr" }

// Routes registers nothing.
//
// Text is read through the application's own controllers, behind its Grants.
// A route here would read any image for anybody who sent one.
func (m *Module) Routes(*fhttp.Router) {}

// Boot opens the configured models on the runtime, initializing the runtime
// first when arandu-onnx has not booted yet. Each file is verified against its
// manifest before ONNX Runtime reads it; a missing or altered file stops the
// boot, naming the model.
func (m *Module) Boot(context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	switch m.state {
	case stateReady:
		return nil
	case stateClosed:
		return ErrClosed
	}
	if err := m.runtime.Init(); err != nil {
		return fmt.Errorf("paddleocr: initializing the ONNX runtime: %w", err)
	}

	type slot struct {
		id     string
		target **onnx.Session
	}
	wanted := []slot{
		{m.cfg.detectorModel(), &m.detector},
		{m.cfg.recognizerModel(), &m.reader},
	}
	if m.cfg.UseTextlineOrientation {
		wanted = append(wanted, slot{ModelTextlineOrientation, &m.textline})
	}
	if m.cfg.UseDocOrientation {
		wanted = append(wanted, slot{ModelDocOrientation, &m.docOrient})
	}

	opened := []*onnx.Session{}
	for _, w := range wanted {
		session, err := m.runtime.Open(m.manifests[w.id], m.cfg.ModelsDir)
		if err != nil {
			for _, s := range opened {
				_ = s.Close()
			}
			m.detector, m.reader, m.textline, m.docOrient = nil, nil, nil, nil
			return fmt.Errorf("paddleocr: opening %s from %s: %w", w.id, m.cfg.ModelsDir, err)
		}
		opened = append(opened, session)
		*w.target = session
	}
	m.state = stateReady
	return nil
}

// Health reports whether the models are open and the runtime under them is
// usable: nil after Boot, ErrNotBooted before, ErrClosed after Close, and the
// runtime's own error when it was closed underneath.
func (m *Module) Health(context.Context) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.usable()
}

// usable is Health for a caller already holding mu.
func (m *Module) usable() error {
	switch m.state {
	case stateNew:
		return ErrNotBooted
	case stateClosed:
		return ErrClosed
	}
	if err := m.runtime.Health(); err != nil {
		return fmt.Errorf("paddleocr: the ONNX runtime is not usable: %w", err)
	}
	return nil
}

// Close closes the module's sessions. It is idempotent, and it waits for the
// Recognize calls in flight.
func (m *Module) Close(context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.state == stateClosed {
		return nil
	}
	m.state = stateClosed
	var errs []error
	for _, s := range []*onnx.Session{m.detector, m.reader, m.textline, m.docOrient} {
		if s != nil {
			errs = append(errs, s.Close())
		}
	}
	m.detector, m.reader, m.textline, m.docOrient = nil, nil, nil, nil
	return errors.Join(errs...)
}
