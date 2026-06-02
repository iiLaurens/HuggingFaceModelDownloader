// Copyright 2025
// SPDX-License-Identifier: Apache-2.0

package hfdownloader

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// setupTestCache creates a minimal HFCache rooted at the given temp dir.
func setupTestCache(t *testing.T) *HFCache {
	t.Helper()
	root := t.TempDir()
	cache := NewHFCache(root, 0)
	return cache
}

// writeBlob creates a fake blob file and returns the path and blob filename.
func writeBlob(t *testing.T, repoDir *RepoDir, hash, content string) string {
	t.Helper()
	if err := os.MkdirAll(repoDir.BlobsDir(), 0755); err != nil {
		t.Fatalf("create blobs dir: %v", err)
	}
	blobPath := repoDir.BlobPath(hash)
	if err := os.WriteFile(blobPath, []byte(content), 0644); err != nil {
		t.Fatalf("write blob %s: %v", hash, err)
	}
	return blobPath
}

// TestReconcileManifest_SnapshotOnly verifies that files in the current
// snapshot are included in the manifest.
func TestReconcileManifest_SnapshotOnly(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks require elevated privileges on Windows")
	}

	cache := setupTestCache(t)
	repoDir, err := cache.Repo("owner/mymodel", RepoTypeModel)
	if err != nil {
		t.Fatalf("Repo: %v", err)
	}
	if err := repoDir.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	if err := repoDir.EnsureFriendlyDir(); err != nil {
		t.Fatalf("EnsureFriendlyDir: %v", err)
	}

	const commit = "abc123"
	const blobHash = "deadbeef1111"
	const fileName = "model.gguf"

	writeBlob(t, repoDir, blobHash, "fake model content")

	if err := repoDir.createSnapshotSymlink(commit, fileName, blobHash); err != nil {
		t.Fatalf("createSnapshotSymlink: %v", err)
	}
	if err := repoDir.CreateFriendlySymlink(commit, fileName, ""); err != nil {
		t.Fatalf("CreateFriendlySymlink: %v", err)
	}
	if err := repoDir.WriteRef("main", commit); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}

	if err := reconcileManifest(repoDir, commit); err != nil {
		t.Fatalf("reconcileManifest: %v", err)
	}

	manifest, err := ReadManifest(filepath.Join(repoDir.FriendlyPath(), ManifestFilename))
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if len(manifest.Files) != 1 {
		t.Fatalf("Files count = %d, want 1; files: %+v", len(manifest.Files), manifest.Files)
	}
	if manifest.Files[0].Name != fileName {
		t.Errorf("File name = %q, want %q", manifest.Files[0].Name, fileName)
	}
}

// TestReconcileManifest_OlderSnapshotFilesPickedUpViaFriendlyView verifies
// that files visible in the friendly view that point to an older snapshot
// (different from the current refs/main commit) are still recorded.
// This was the original bug: only the snapshot for the current commit was
// walked, so files from older commits were silently dropped.
func TestReconcileManifest_OlderSnapshotFilesPickedUpViaFriendlyView(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks require elevated privileges on Windows")
	}

	cache := setupTestCache(t)
	repoDir, err := cache.Repo("owner/mymodel", RepoTypeModel)
	if err != nil {
		t.Fatalf("Repo: %v", err)
	}
	if err := repoDir.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	if err := repoDir.EnsureFriendlyDir(); err != nil {
		t.Fatalf("EnsureFriendlyDir: %v", err)
	}

	const oldCommit = "old000"
	const newCommit = "new111"
	const blobHash = "cafebabe2222"
	const fileName = "model.gguf"

	// Blob and snapshot for the OLD commit.
	writeBlob(t, repoDir, blobHash, "fake model content for old snapshot")
	if err := repoDir.createSnapshotSymlink(oldCommit, fileName, blobHash); err != nil {
		t.Fatalf("createSnapshotSymlink (old): %v", err)
	}

	// Friendly-view symlink still pointing at the OLD snapshot.
	if err := repoDir.CreateFriendlySymlink(oldCommit, fileName, ""); err != nil {
		t.Fatalf("CreateFriendlySymlink: %v", err)
	}

	// refs/main now points to the NEW commit (which has no snapshot).
	if err := repoDir.WriteRef("main", newCommit); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}

	// reconcileManifest is called with the new commit (no snapshot).
	if err := reconcileManifest(repoDir, newCommit); err != nil {
		t.Fatalf("reconcileManifest: %v", err)
	}

	manifest, err := ReadManifest(filepath.Join(repoDir.FriendlyPath(), ManifestFilename))
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if len(manifest.Files) != 1 {
		t.Fatalf("Files count = %d, want 1 (file from older snapshot); files: %+v", len(manifest.Files), manifest.Files)
	}
	if manifest.Files[0].Name != fileName {
		t.Errorf("File name = %q, want %q", manifest.Files[0].Name, fileName)
	}
}

// TestReconcileManifest_NoDuplicatesWhenBothPhasesSeeSameFile verifies that
// a file present in both the snapshot directory and the friendly view is only
// listed once in the manifest.
func TestReconcileManifest_NoDuplicatesWhenBothPhasesSeeSameFile(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks require elevated privileges on Windows")
	}

	cache := setupTestCache(t)
	repoDir, err := cache.Repo("owner/mymodel", RepoTypeModel)
	if err != nil {
		t.Fatalf("Repo: %v", err)
	}
	if err := repoDir.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	if err := repoDir.EnsureFriendlyDir(); err != nil {
		t.Fatalf("EnsureFriendlyDir: %v", err)
	}

	const commit = "abc456"
	const blobHash = "11223344aabb"
	const fileName = "model.gguf"

	writeBlob(t, repoDir, blobHash, "model data")
	if err := repoDir.createSnapshotSymlink(commit, fileName, blobHash); err != nil {
		t.Fatalf("createSnapshotSymlink: %v", err)
	}
	if err := repoDir.CreateFriendlySymlink(commit, fileName, ""); err != nil {
		t.Fatalf("CreateFriendlySymlink: %v", err)
	}
	if err := repoDir.WriteRef("main", commit); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}

	if err := reconcileManifest(repoDir, commit); err != nil {
		t.Fatalf("reconcileManifest: %v", err)
	}

	manifest, err := ReadManifest(filepath.Join(repoDir.FriendlyPath(), ManifestFilename))
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if len(manifest.Files) != 1 {
		t.Fatalf("Files count = %d, want 1 (no duplicates); files: %+v", len(manifest.Files), manifest.Files)
	}
}

// TestReconcileManifest_FilterSubdirNoDuplicates verifies that when a file is
// downloaded with AppendFilterSubdir (friendly path has a subdir prefix) the
// file is still found by the snapshot walk and not duplicated by the friendly
// view walk.
func TestReconcileManifest_FilterSubdirNoDuplicates(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks require elevated privileges on Windows")
	}

	cache := setupTestCache(t)
	repoDir, err := cache.Repo("owner/mymodel", RepoTypeModel)
	if err != nil {
		t.Fatalf("Repo: %v", err)
	}
	if err := repoDir.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	if err := repoDir.EnsureFriendlyDir(); err != nil {
		t.Fatalf("EnsureFriendlyDir: %v", err)
	}

	const commit = "filt001"
	const blobHash = "aabbcc112233"
	const fileName = "model.gguf"
	const filterSubdir = "q4_k_m"

	writeBlob(t, repoDir, blobHash, "quantized model data")
	if err := repoDir.createSnapshotSymlink(commit, fileName, blobHash); err != nil {
		t.Fatalf("createSnapshotSymlink: %v", err)
	}
	// Friendly view: models/owner/mymodel/q4_k_m/model.gguf
	if err := repoDir.CreateFriendlySymlink(commit, fileName, filterSubdir); err != nil {
		t.Fatalf("CreateFriendlySymlink: %v", err)
	}
	if err := repoDir.WriteRef("main", commit); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}

	if err := reconcileManifest(repoDir, commit); err != nil {
		t.Fatalf("reconcileManifest: %v", err)
	}

	manifest, err := ReadManifest(filepath.Join(repoDir.FriendlyPath(), ManifestFilename))
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if len(manifest.Files) != 1 {
		t.Fatalf("Files count = %d, want 1 (no filter-subdir duplicates); files: %+v", len(manifest.Files), manifest.Files)
	}
	// The snapshot walk (Phase 1) names it "model.gguf"; the friendly view (Phase 2)
	// would name it "q4_k_m/model.gguf".  Phase 1 wins and deduplication prevents Phase 2
	// from adding a second entry.
	if manifest.Files[0].Name != fileName {
		t.Errorf("File name = %q, want %q (snapshot path takes precedence)", manifest.Files[0].Name, fileName)
	}
}

// TestReconcileManifest_BrokenSnapshotSymlinkSkipped verifies that a symlink
// in the snapshot directory whose blob is missing is not included in the manifest.
func TestReconcileManifest_BrokenSnapshotSymlinkSkipped(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks require elevated privileges on Windows")
	}

	cache := setupTestCache(t)
	repoDir, err := cache.Repo("owner/mymodel", RepoTypeModel)
	if err != nil {
		t.Fatalf("Repo: %v", err)
	}
	if err := repoDir.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	if err := repoDir.EnsureFriendlyDir(); err != nil {
		t.Fatalf("EnsureFriendlyDir: %v", err)
	}

	const commit = "brkn001"
	const blobHash = "00000000ffff"
	const fileName = "missing.gguf"

	// Create the snapshot symlink but NOT the actual blob.
	if err := repoDir.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	if err := repoDir.createSnapshotSymlink(commit, fileName, blobHash); err != nil {
		t.Fatalf("createSnapshotSymlink: %v", err)
	}
	if err := repoDir.WriteRef("main", commit); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}

	if err := reconcileManifest(repoDir, commit); err != nil {
		t.Fatalf("reconcileManifest: %v", err)
	}

	manifest, err := ReadManifest(filepath.Join(repoDir.FriendlyPath(), ManifestFilename))
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if len(manifest.Files) != 0 {
		t.Fatalf("Files count = %d, want 0 (broken symlink should be skipped); files: %+v", len(manifest.Files), manifest.Files)
	}
}

// TestReconcileManifest_MultipleFilesIncluded verifies that multiple files
// from both phases are all included.
func TestReconcileManifest_MultipleFilesIncluded(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks require elevated privileges on Windows")
	}

	cache := setupTestCache(t)
	repoDir, err := cache.Repo("owner/mymodel", RepoTypeModel)
	if err != nil {
		t.Fatalf("Repo: %v", err)
	}
	if err := repoDir.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	if err := repoDir.EnsureFriendlyDir(); err != nil {
		t.Fatalf("EnsureFriendlyDir: %v", err)
	}

	const currentCommit = "curr001"
	const oldCommit = "old002"

	// File A: in current snapshot and friendly view.
	writeBlob(t, repoDir, "blobAAAA", "file A content")
	if err := repoDir.createSnapshotSymlink(currentCommit, "fileA.bin", "blobAAAA"); err != nil {
		t.Fatalf("createSnapshotSymlink A: %v", err)
	}
	if err := repoDir.CreateFriendlySymlink(currentCommit, "fileA.bin", ""); err != nil {
		t.Fatalf("CreateFriendlySymlink A: %v", err)
	}

	// File B: only in OLD snapshot, friendly view still points to old snapshot.
	writeBlob(t, repoDir, "blobBBBB", "file B content (from old snapshot)")
	if err := repoDir.createSnapshotSymlink(oldCommit, "fileB.bin", "blobBBBB"); err != nil {
		t.Fatalf("createSnapshotSymlink B: %v", err)
	}
	if err := repoDir.CreateFriendlySymlink(oldCommit, "fileB.bin", ""); err != nil {
		t.Fatalf("CreateFriendlySymlink B: %v", err)
	}

	if err := repoDir.WriteRef("main", currentCommit); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}

	if err := reconcileManifest(repoDir, currentCommit); err != nil {
		t.Fatalf("reconcileManifest: %v", err)
	}

	manifest, err := ReadManifest(filepath.Join(repoDir.FriendlyPath(), ManifestFilename))
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if len(manifest.Files) != 2 {
		t.Fatalf("Files count = %d, want 2; files: %+v", len(manifest.Files), manifest.Files)
	}
	names := make(map[string]bool)
	for _, f := range manifest.Files {
		names[f.Name] = true
	}
	if !names["fileA.bin"] {
		t.Error("fileA.bin missing from manifest")
	}
	if !names["fileB.bin"] {
		t.Error("fileB.bin missing from manifest (picked up via friendly view)")
	}
}
