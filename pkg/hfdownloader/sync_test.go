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

// TestReconcileManifest_MultipleSnapshotsAllFilesIncluded verifies that files
// from two separate snapshot directories (each from a different download
// session / filter) are both included via Phase 1 of reconcileManifest, even
// when there are NO friendly-view symlinks and NO refs entry pointing at the
// older snapshot.
//
// This is the core regression test for the "missing files" bug: user downloads
// Q4_K_M with commit A, then Q8_0 with commit B.  refs/main → B.  Without
// the fix, reconcileManifest only walked snapshot B and silently dropped the
// Q4 file.
func TestReconcileManifest_MultipleSnapshotsAllFilesIncluded(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks require elevated privileges on Windows")
	}

	cache := setupTestCache(t)
	repoDir, err := cache.Repo("owner/llama", RepoTypeModel)
	if err != nil {
		t.Fatalf("Repo: %v", err)
	}
	if err := repoDir.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	if err := repoDir.EnsureFriendlyDir(); err != nil {
		t.Fatalf("EnsureFriendlyDir: %v", err)
	}

	const commitA = "aaaa1111"
	const commitB = "bbbb2222"
	const blobQ4 = "blobQ4aaaa"
	const blobQ8 = "blobQ8bbbb"

	// First session: Q4 file downloaded at commit A.
	writeBlob(t, repoDir, blobQ4, "Q4 model data")
	if err := repoDir.createSnapshotSymlink(commitA, "model.Q4_K_M.gguf", blobQ4); err != nil {
		t.Fatalf("createSnapshotSymlink (A): %v", err)
	}

	// Second session: Q8 file downloaded at commit B.
	writeBlob(t, repoDir, blobQ8, "Q8 model data")
	if err := repoDir.createSnapshotSymlink(commitB, "model.Q8_0.gguf", blobQ8); err != nil {
		t.Fatalf("createSnapshotSymlink (B): %v", err)
	}

	// refs/main now points only to the newer commit B.
	if err := repoDir.WriteRef("main", commitB); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}

	// No friendly-view symlinks are created: Phase 2 must NOT be needed.
	// reconcileManifest is called with commit B (the current ref).
	if err := reconcileManifest(repoDir, commitB); err != nil {
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
	if !names["model.Q4_K_M.gguf"] {
		t.Error("model.Q4_K_M.gguf missing from manifest (file from older snapshot)")
	}
	if !names["model.Q8_0.gguf"] {
		t.Error("model.Q8_0.gguf missing from manifest (file from newer snapshot)")
	}
}

// TestSyncRepoFriendlyView_MultipleSnapshots verifies that syncRepoFriendlyView
// creates symlinks for files from ALL snapshot directories, not just the one
// the current ref points to.
func TestSyncRepoFriendlyView_MultipleSnapshots(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks require elevated privileges on Windows")
	}

	cache := setupTestCache(t)
	repoDir, err := cache.Repo("owner/llama", RepoTypeModel)
	if err != nil {
		t.Fatalf("Repo: %v", err)
	}
	if err := repoDir.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	if err := repoDir.EnsureFriendlyDir(); err != nil {
		t.Fatalf("EnsureFriendlyDir: %v", err)
	}

	const commitA = "aaaa3333"
	const commitB = "bbbb4444"
	const blobQ4 = "blobQ4cccc"
	const blobQ8 = "blobQ8dddd"

	// Session 1: Q4 file at commit A.
	writeBlob(t, repoDir, blobQ4, "Q4 content")
	if err := repoDir.createSnapshotSymlink(commitA, "model.Q4_K_M.gguf", blobQ4); err != nil {
		t.Fatalf("createSnapshotSymlink (A): %v", err)
	}

	// Session 2: Q8 file at commit B.
	writeBlob(t, repoDir, blobQ8, "Q8 content")
	if err := repoDir.createSnapshotSymlink(commitB, "model.Q8_0.gguf", blobQ8); err != nil {
		t.Fatalf("createSnapshotSymlink (B): %v", err)
	}

	// refs/main → commit B (newer session).
	if err := repoDir.WriteRef("main", commitB); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}

	created, _, err := cache.syncRepoFriendlyView(repoDir, SyncOptions{})
	if err != nil {
		t.Fatalf("syncRepoFriendlyView: %v", err)
	}
	// Both files should have been created (2 created, 0 updated).
	if created != 2 {
		t.Errorf("Created = %d, want 2", created)
	}

	// Verify both symlinks exist in the friendly view.
	friendlyPath := repoDir.FriendlyPath()
	for _, name := range []string{"model.Q4_K_M.gguf", "model.Q8_0.gguf"} {
		linkPath := filepath.Join(friendlyPath, name)
		if _, err := os.Lstat(linkPath); err != nil {
			t.Errorf("friendly symlink %s missing: %v", name, err)
		}
		// Ensure the symlink resolves to an existing blob.
		if _, err := os.Stat(linkPath); err != nil {
			t.Errorf("friendly symlink %s does not resolve: %v", name, err)
		}
	}
}
