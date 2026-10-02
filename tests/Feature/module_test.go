package feature_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/arandu-io/framework/foundation"
	fhttp "github.com/arandu-io/framework/http"

	onnx "github.com/kuaa-io/arandu-onnx"
	paddleocr "github.com/kuaa-io/arandu-paddleocr"
)

// These tests drive the module the way an application does: build it,
// register it, boot it, ask its health and close it.

// unbootedModule builds the module on a runtime that loads nothing.
func unbootedModule(t *testing.T, cfg paddleocr.Config) *paddleocr.Module {
	t.Helper()
	rt, err := onnx.NewRuntime(onnx.Config{})
	if err != nil {
		t.Fatalf("building the runtime: %v", err)
	}
	m, err := paddleocr.New(cfg, rt)
	if err != nil {
		t.Fatalf("building the module: %v", err)
	}
	return m
}

func TestTheModuleIsRegisteredAsAModule(t *testing.T) {
	t.Parallel()

	var module foundation.Module = unbootedModule(t, paddleocr.Config{ModelsDir: t.TempDir()})
	if module.Name() != "paddleocr" {
		t.Fatalf("Name() = %q, want paddleocr: route names and the boot log are keyed by it", module.Name())
	}
}

// TestTheModuleServesNoRoute holds the reason Routes is empty: text is read
// through the application's own controllers, and a route here would read any
// image for anybody.
func TestTheModuleServesNoRoute(t *testing.T) {
	t.Parallel()

	m := unbootedModule(t, paddleocr.Config{ModelsDir: t.TempDir()})
	router := fhttp.NewRouter()
	m.Routes(router.ForModule(m.Name()))
	if routes := router.Routes(); len(routes) != 0 {
		t.Fatalf("the module registered %d routes, want none: %+v", len(routes), routes)
	}
}

func TestNewRefusesWhatCannotWork(t *testing.T) {
	t.Parallel()

	rt, err := onnx.NewRuntime(onnx.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := paddleocr.New(paddleocr.Config{ModelsDir: t.TempDir()}, nil); err == nil {
		t.Error("New accepted a nil runtime")
	}
	if _, err := paddleocr.New(paddleocr.Config{}, rt); err == nil || !strings.Contains(err.Error(), "ModelsDir") {
		t.Errorf("New without a models directory = %v, want an error naming ModelsDir", err)
	}
	if _, err := paddleocr.New(paddleocr.Config{ModelsDir: "x", Detector: "tiny"}, rt); err == nil {
		t.Error("New accepted an unknown detector")
	}
}

func TestNewFillsTheDefaults(t *testing.T) {
	t.Parallel()

	m := unbootedModule(t, paddleocr.Config{ModelsDir: "models"})
	cfg := m.Config()
	if cfg.Detector != paddleocr.DetectorMobile || cfg.Recognizer != paddleocr.RecognizerLatin ||
		cfg.LimitSideLen != 960 || cfg.Thresh != 0.3 || cfg.BoxThresh != 0.6 || cfg.UnclipRatio != 1.5 ||
		cfg.MaxCandidates != 1000 || cfg.RecBatchSize != 6 || cfg.RecMaxWidth != 3200 || cfg.TextlineThreshold != 0.9 {
		t.Errorf("Config() = %+v, want the documented defaults", cfg)
	}
	if got := m.Vocabulary().Classes(); got != 838 {
		t.Errorf("the latin recognizer's vocabulary has %d classes, want 838", got)
	}
}

func TestTheModuleIsUnusableBeforeBootAndAfterClose(t *testing.T) {
	m := bootModule(t, paddleocr.Config{})
	ctx := context.Background()
	if err := m.Health(ctx); err != nil {
		t.Fatalf("Health after Boot = %v, want nil", err)
	}
	if err := m.Boot(ctx); err != nil {
		t.Errorf("a second Boot = %v, want nil", err)
	}
	if err := m.Close(ctx); err != nil {
		t.Fatalf("Close = %v", err)
	}
	if err := m.Close(ctx); err != nil {
		t.Errorf("a second Close = %v, want nil", err)
	}
	if err := m.Health(ctx); !errors.Is(err, paddleocr.ErrClosed) {
		t.Errorf("Health after Close = %v, want ErrClosed", err)
	}
	page, _ := drawPage(t, documentLines[:1], 40)
	if _, err := m.Recognize(ctx, page, paddleocr.Options{}); !errors.Is(err, paddleocr.ErrClosed) {
		t.Errorf("Recognize after Close = %v, want ErrClosed", err)
	}
	if err := m.Boot(ctx); !errors.Is(err, paddleocr.ErrClosed) {
		t.Errorf("Boot after Close = %v, want ErrClosed", err)
	}

	fresh := unbootedModule(t, paddleocr.Config{ModelsDir: t.TempDir()})
	if err := fresh.Health(ctx); !errors.Is(err, paddleocr.ErrNotBooted) {
		t.Errorf("Health before Boot = %v, want ErrNotBooted", err)
	}
	if _, err := fresh.Recognize(ctx, page, paddleocr.Options{}); !errors.Is(err, paddleocr.ErrNotBooted) {
		t.Errorf("Recognize before Boot = %v, want ErrNotBooted", err)
	}
}

func TestHealthFailsWhenTheRuntimeIsClosedUnderneath(t *testing.T) {
	cfg := paddleocr.Config{ModelsDir: modelsDir(t)}
	rt := newRuntime(t)
	m, err := paddleocr.New(cfg, rt)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Boot(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := rt.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Health(context.Background()); !errors.Is(err, onnx.ErrClosed) {
		t.Errorf("Health with the runtime closed = %v, want onnx.ErrClosed", err)
	}
}

func TestBootRefusesAMissingOrAlteredModel(t *testing.T) {
	rt := newRuntime(t)
	ctx := context.Background()

	empty := t.TempDir()
	m, err := paddleocr.New(paddleocr.Config{ModelsDir: empty}, rt)
	if err != nil {
		t.Fatal(err)
	}
	err = m.Boot(ctx)
	if err == nil || !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), paddleocr.ModelMobileDetector) {
		t.Errorf("Boot on an empty directory = %v, want a missing file naming %s", err, paddleocr.ModelMobileDetector)
	}
	if err := m.Health(ctx); !errors.Is(err, paddleocr.ErrNotBooted) {
		t.Errorf("Health after a failed Boot = %v, want ErrNotBooted", err)
	}

	altered := t.TempDir()
	manifest, err := paddleocr.LookupManifest(paddleocr.ModelMobileDetector)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(altered, manifest.File)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("not the detector"), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err = paddleocr.New(paddleocr.Config{ModelsDir: altered}, rt)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Boot(ctx); !errors.Is(err, onnx.ErrDigestMismatch) {
		t.Errorf("Boot on an altered file = %v, want onnx.ErrDigestMismatch", err)
	}
}
