package feature_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	paddleocr "github.com/hyz-is/arandu-paddleocr"
)

// The download command is exercised here only in -check mode, which reads the
// directory and connects to nothing. Downloading is never tested against the
// network.

// buildCommand compiles cmd/paddleocr-models into a temporary directory.
func buildCommand(t *testing.T) string {
	t.Helper()
	binary := filepath.Join(t.TempDir(), "paddleocr-models")
	build := exec.Command("go", "build", "-o", binary, "./cmd/paddleocr-models")
	build.Dir = packageRoot(t)
	build.Env = append(os.Environ(), "GOWORK=off")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the command: %v\n%s", err, out)
	}
	return binary
}

func TestTheCommandChecksTheCachedModels(t *testing.T) {
	dir := modelsDir(t)
	out, err := exec.Command(buildCommand(t), "-dir", dir, "-check").CombinedOutput()
	if err != nil {
		t.Fatalf("checking %s: %v\n%s", dir, err, out)
	}
	manifests, err := paddleocr.Manifests()
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range manifests {
		if !strings.Contains(string(out), "ok       "+m.ID+" "+m.Version) {
			t.Errorf("the check did not report %s %s as in place:\n%s", m.ID, m.Version, out)
		}
	}
}

func TestTheCommandReportsAMissingOrAlteredFile(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	manifest, err := paddleocr.LookupManifest(paddleocr.ModelTextlineOrientation)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, manifest.File)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("altered"), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(buildCommand(t), "-dir", dir, "-check", "-only", manifest.ID+","+paddleocr.ModelMobileDetector).CombinedOutput()
	if err == nil {
		t.Fatalf("the check passed on an altered and a missing file:\n%s", out)
	}
	for _, want := range []string{"missing  " + manifest.ID, "sha256 is", "missing  " + paddleocr.ModelMobileDetector, "2 of 2 models are not in place"} {
		if !strings.Contains(string(out), want) {
			t.Errorf("the check's output lacks %q:\n%s", want, out)
		}
	}

	out, err = exec.Command(buildCommand(t), "-dir", dir, "-check", "-only", "no-such-model").CombinedOutput()
	if err == nil || !strings.Contains(string(out), `no model is called "no-such-model"`) {
		t.Errorf("an unknown id = %v:\n%s", err, out)
	}
}
