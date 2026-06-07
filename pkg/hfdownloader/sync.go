// Copyright 2025
// SPDX-License-Identifier: Apache-2.0

package hfdownloader

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// SyncOptions configures the sync operation.
type SyncOptions struct {
	// Clean removes orphaned symlinks in the friendly view
	Clean bool
	// Verbose prints detailed progress
	Verbose bool
}

// SyncResult contains statistics from a sync operation.
type SyncResult struct {
	ReposScanned    int
	SymlinksCreated int
	SymlinksUpdated int
	OrphansRemoved  int
	Errors          []error
}

// Sync regenerates the friendly view (models/, datasets/) from the hub cache.
// It scans all repos in hub/, reads their refs to find current commits,
// and creates symlinks in the friendly view pointing to snapshot files.
func (c *HFCache) Sync(opts SyncOptions) (*SyncResult, error) {
	result := &SyncResult{}

	hubDir := c.HubDir()
	if _, err := os.Stat(hubDir); errors.Is(err, os.ErrNotExist) {
		return result, nil // Nothing to sync
	}

	// Scan hub/ for repo directories
	entries, err := os.ReadDir(hubDir)
	if err != nil {
		return nil, fmt.Errorf("read hub directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		// Parse repo directory name (models--owner--name or datasets--owner--name)
		repoType, owner, name, ok := parseRepoDirName(entry.Name())
		if !ok {
			continue // Not a valid repo directory
		}

		result.ReposScanned++

		repoDir, err := c.Repo(owner+"/"+name, repoType)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Errorf("parse repo %s: %w", entry.Name(), err))
			continue
		}

		// Sync this repo's friendly view
		created, updated, err := c.syncRepoFriendlyView(repoDir, opts)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Errorf("sync %s: %w", entry.Name(), err))
			continue
		}

		result.SymlinksCreated += created
		result.SymlinksUpdated += updated
	}

	// Clean orphaned symlinks if requested
	if opts.Clean {
		removed, err := c.cleanOrphanedSymlinks(opts)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Errorf("clean orphans: %w", err))
		}
		result.OrphansRemoved = removed
	}

	return result, nil
}

// parseRepoDirName extracts repo type, owner, and name from a hub directory name.
// Format: models--owner--name or datasets--owner--name
func parseRepoDirName(dirName string) (RepoType, string, string, bool) {
	var repoType RepoType
	var rest string

	if strings.HasPrefix(dirName, "models--") {
		repoType = RepoTypeModel
		rest = strings.TrimPrefix(dirName, "models--")
	} else if strings.HasPrefix(dirName, "datasets--") {
		repoType = RepoTypeDataset
		rest = strings.TrimPrefix(dirName, "datasets--")
	} else {
		return "", "", "", false
	}

	// Split owner--name
	parts := strings.SplitN(rest, "--", 2)
	if len(parts) != 2 {
		return "", "", "", false
	}

	return repoType, parts[0], parts[1], true
}

// syncRepoFriendlyView syncs a single repo's friendly view.
// Returns (created, updated, error).
func (c *HFCache) syncRepoFriendlyView(repoDir *RepoDir, opts SyncOptions) (int, int, error) {
	created := 0
	updated := 0

	// Find the current commit from ALL available refs (prefer main/master).
	commit, _, err := repoDir.ReadBestRef()
	if err != nil {
		return 0, 0, fmt.Errorf("read refs: %w", err)
	}

	// Enumerate every snapshot directory present on disk, not just the one
	// for the current ref.  Users often download different filters or
	// revisions in separate runs, each creating its own snapshot directory.
	// Limiting the walk to a single snapshot would silently drop those files
	// from the friendly view.
	snapshots, err := repoDir.ListSnapshots()
	if err != nil {
		return 0, 0, fmt.Errorf("list snapshots: %w", err)
	}
	if len(snapshots) == 0 {
		return 0, 0, nil // Nothing to sync
	}

	// Process the current commit first so its version of a file wins when
	// the same relative path appears in multiple snapshots.
	orderedSnapshots := OrderedSnapshotList(snapshots, commit)

	// Ensure friendly directory exists
	if err := repoDir.EnsureFriendlyDir(); err != nil {
		return 0, 0, fmt.Errorf("ensure friendly dir: %w", err)
	}

	// seenPaths prevents creating a second symlink for a path already handled
	// by a higher-priority (earlier) snapshot.
	seenPaths := make(map[string]bool)

	for _, snapshotCommit := range orderedSnapshots {
		snapshotDir := repoDir.SnapshotDir(snapshotCommit)
		if _, err := os.Stat(snapshotDir); errors.Is(err, os.ErrNotExist) {
			continue
		}

		walkErr := filepath.Walk(snapshotDir, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return nil
			}

			relPath, err := filepath.Rel(snapshotDir, path)
			if err != nil {
				return err
			}

			// A higher-priority snapshot already owns this path.
			if seenPaths[relPath] {
				return nil
			}
			seenPaths[relPath] = true

			// Check if the friendly symlink already exists and points to the
			// right place; skip the write if so.
			friendlyPath := filepath.Join(repoDir.FriendlyPath(), relPath)
			existingTarget, readErr := os.Readlink(friendlyPath)
			snapshotPath := repoDir.SnapshotPath(snapshotCommit, relPath)
			expectedTarget, _ := filepath.Rel(filepath.Dir(friendlyPath), snapshotPath)
			if readErr == nil && existingTarget == expectedTarget {
				return nil
			}

			if err := repoDir.CreateFriendlySymlink(snapshotCommit, relPath, ""); err != nil {
				return fmt.Errorf("create symlink for %s: %w", relPath, err)
			}
			if errors.Is(readErr, os.ErrNotExist) {
				created++
			} else {
				updated++
			}
			return nil
		})

		if walkErr != nil {
			return created, updated, fmt.Errorf("walk snapshot %s: %w", snapshotCommit, walkErr)
		}
	}

	// Reconcile the hfd.yaml manifest so it accurately reflects the files
	// that are actually present on disk.  This corrects stale manifests
	// written by older versions of the software or by external tools.
	if reconcileErr := reconcileManifest(repoDir, commit); reconcileErr != nil {
		// Non-fatal: log the error but don't fail the whole sync.
		_ = reconcileErr
	}

	return created, updated, nil
}

// orderedSnapshotList returns a snapshot list in which currentCommit (if
// non-empty) comes first, followed by all remaining commits in their
// original order.  This ensures the current commit's files take priority
// when the same relative path exists in multiple snapshots.
func OrderedSnapshotList(snapshots []string, currentCommit string) []string {
	if currentCommit == "" {
		return snapshots
	}
	ordered := make([]string, 0, len(snapshots))
	// Put current commit first (may not be in the snapshot list if the ref
	// points to an as-yet-undownloaded revision; the walk will just skip it).
	ordered = append(ordered, currentCommit)
	for _, s := range snapshots {
		if s != currentCommit {
			ordered = append(ordered, s)
		}
	}
	return ordered
}

// reconcileManifest rebuilds hfd.yaml so it accurately reflects the files
// that are actually present on disk.  Three scan phases are performed so that
// no accessible file is missed:
//
//  1. ALL hub snapshot directories (hub/models--…/snapshots/*/): the
//     canonical source of truth.  The current commit's snapshot is walked
//     first so that its version of a file takes priority when the same
//     relative path appears in multiple snapshots (e.g. a file that was
//     updated between two download sessions).  Each symlink whose blob target
//     exists on disk is included.
//
//  2. Friendly-view directory (models/{owner}/{repo}/): picks up files that
//     are visible to the user but were not captured by the snapshot walk —
//     for example files placed directly by an external tool, or symlinks that
//     resolve through a snapshot not found in phase 1.
//
// Duplicates are detected by blob hash so that the same underlying file is
// never counted twice even if it appears under different names in the two
// directory trees (e.g. when AppendFilterSubdir is used).
//
// If a manifest already exists its repository metadata (branch, commit,
// command, timing) is preserved; only the file list and totals are updated.
// If no manifest exists a new one is created with the information available
// from disk.  The function is best-effort: errors are silently ignored so a
// manifest problem never aborts a rebuild/sync.
func reconcileManifest(repoDir *RepoDir, commit string) error {
	var files []ManifestFile
	// seenBlobs tracks blob filenames (sha256 hashes) already added so that
	// the friendly-view walk does not double-count files found in the
	// snapshot walk.
	seenBlobs := make(map[string]bool)
	// seenPaths tracks snapshot-relative paths so that the same relative path
	// is only recorded once (current commit's version wins).
	seenPaths := make(map[string]bool)

	// ── Phase 1: ALL snapshot directories ──────────────────────────────────
	// Walk every snapshot present on disk, not just the one for the current
	// ref.  When users download different filters or revisions in separate
	// sessions, each run creates its own snapshot directory.  Restricting the
	// walk to a single snapshot silently drops those files from the manifest.
	allSnapshots, _ := repoDir.ListSnapshots()
	orderedSnapshots := OrderedSnapshotList(allSnapshots, commit)

	for _, snapshotCommit := range orderedSnapshots {
		snapshotDir := repoDir.SnapshotDir(snapshotCommit)
		if _, err := os.Stat(snapshotDir); err != nil {
			continue
		}
		_ = filepath.Walk(snapshotDir, func(path string, info os.FileInfo, err error) error {
			if err != nil || info == nil || info.IsDir() {
				return nil
			}
			if info.Mode()&os.ModeSymlink == 0 {
				return nil
			}
			target, readErr := os.Readlink(path)
			if readErr != nil {
				return nil
			}
			if !filepath.IsAbs(target) {
				target = filepath.Join(filepath.Dir(path), target)
			}
			blobPath := filepath.Clean(target)
			blobInfo, statErr := os.Stat(blobPath)
			if statErr != nil {
				// Blob doesn't exist on disk — skip this file.
				return nil
			}
			relPath, relErr := filepath.Rel(snapshotDir, path)
			if relErr != nil {
				return nil
			}
			// The current commit's snapshot is walked first; skip if this
			// relative path was already recorded by an earlier iteration.
			if seenPaths[relPath] {
				return nil
			}
			blobName := filepath.Base(blobPath)
			seenBlobs[blobName] = true
			seenPaths[relPath] = true
			files = append(files, ManifestFile{
				Name: relPath,
				Blob: "blobs/" + blobName,
				Size: blobInfo.Size(),
			})
			return nil
		})
	}

	// ── Phase 2: friendly-view directory ───────────────────────────────────
	// Walk the models/datasets directory so that files visible to the user
	// but absent from any snapshot (e.g. files placed directly by an
	// external tool) are also recorded.
	friendlyPath := repoDir.FriendlyPath()
	if _, statErr := os.Stat(friendlyPath); statErr == nil {
		_ = filepath.Walk(friendlyPath, func(path string, info os.FileInfo, err error) error {
			if err != nil || info == nil || info.IsDir() {
				return nil
			}
			// Never include the manifest file itself.
			if info.Name() == ManifestFilename {
				return nil
			}

			relPath, relErr := filepath.Rel(friendlyPath, path)
			if relErr != nil {
				return nil
			}

			if info.Mode()&os.ModeSymlink != 0 {
				// Follow the complete symlink chain (friendly → snapshot → blob).
				realPath, evalErr := filepath.EvalSymlinks(path)
				if evalErr != nil {
					// Broken symlink — skip.
					return nil
				}
				blobName := filepath.Base(realPath)
				if seenBlobs[blobName] {
					// Already recorded during the snapshot walk.
					return nil
				}
				fileInfo, statErr := os.Stat(realPath)
				if statErr != nil {
					return nil
				}
				seenBlobs[blobName] = true
				seenPaths[relPath] = true
				files = append(files, ManifestFile{
					Name: relPath,
					Blob: "blobs/" + blobName,
					Size: fileInfo.Size(),
				})
			} else {
				// Regular (non-symlink) file placed directly in the friendly view.
				if seenPaths[relPath] {
					return nil
				}
				seenPaths[relPath] = true
				files = append(files, ManifestFile{
					Name: relPath,
					Size: info.Size(),
				})
			}
			return nil
		})
	}

	// If nothing was found at all there is nothing to reconcile.
	if len(files) == 0 {
		if _, friendlyErr := os.Stat(friendlyPath); errors.Is(friendlyErr, os.ErrNotExist) {
			return nil
		}
	}

	// Read the existing manifest, if any.
	manifestPath := filepath.Join(friendlyPath, ManifestFilename)
	existing, _ := ReadManifest(manifestPath)

	// Build the updated manifest.
	repoType := "model"
	if repoDir.Type() == RepoTypeDataset {
		repoType = "dataset"
	}

	// Try to find the branch name from ALL refs (not just main/master).
	_, branch, _ := repoDir.ReadBestRef()
	if branch == "" {
		branch = "main"
	}

	now := time.Now().UTC()
	manifest := &DownloadManifest{
		Version:     "1.0",
		Type:        repoType,
		Repo:        repoDir.RepoID(),
		Branch:      branch,
		Commit:      commit,
		RepoPath:    "hub/" + RepoTypeName(repoDir.Type() == RepoTypeDataset) + "--" + strings.ReplaceAll(repoDir.RepoID(), "/", "--"),
		StartedAt:   now,
		CompletedAt: now,
		Files:       files,
	}
	// Compute totals from actual files.
	for _, f := range files {
		manifest.TotalFiles++
		manifest.TotalSize += f.Size
	}

	// Preserve metadata from the existing manifest where possible.
	if existing != nil {
		if existing.Branch != "" {
			manifest.Branch = existing.Branch
		}
		if existing.Commit != "" {
			manifest.Commit = existing.Commit
		}
		if existing.Command != "" {
			manifest.Command = existing.Command
		}
		if !existing.StartedAt.IsZero() {
			manifest.StartedAt = existing.StartedAt
		}
		if !existing.CompletedAt.IsZero() {
			manifest.CompletedAt = existing.CompletedAt
		}
	}

	_, writeErr := manifest.Write(friendlyPath)
	return writeErr
}

// cleanOrphanedSymlinks removes symlinks in friendly view that point to non-existent files.
func (c *HFCache) cleanOrphanedSymlinks(opts SyncOptions) (int, error) {
	removed := 0

	// Clean models/
	modelsRemoved, err := cleanOrphansInDir(c.ModelsDir())
	if err != nil {
		return removed, fmt.Errorf("clean models: %w", err)
	}
	removed += modelsRemoved

	// Clean datasets/
	datasetsRemoved, err := cleanOrphansInDir(c.DatasetsDir())
	if err != nil {
		return removed, fmt.Errorf("clean datasets: %w", err)
	}
	removed += datasetsRemoved

	return removed, nil
}

// cleanOrphansInDir removes broken symlinks in a directory tree.
func cleanOrphansInDir(dir string) (int, error) {
	removed := 0

	if _, err := os.Stat(dir); errors.Is(err, os.ErrNotExist) {
		return 0, nil
	}

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// Skip errors (e.g., permission denied)
			return nil
		}

		// Check if it's a symlink
		if info.Mode()&os.ModeSymlink == 0 {
			return nil
		}

		// Check if symlink target exists
		target, err := os.Readlink(path)
		if err != nil {
			return nil // Can't read symlink, skip
		}

		// Resolve relative to symlink location
		if !filepath.IsAbs(target) {
			target = filepath.Join(filepath.Dir(path), target)
		}

		if _, err := os.Stat(target); errors.Is(err, os.ErrNotExist) {
			// Broken symlink, remove it
			if err := os.Remove(path); err != nil {
				return nil // Skip errors
			}
			removed++
		}

		return nil
	})

	if err != nil {
		return removed, err
	}

	// Clean up empty directories
	cleanEmptyDirs(dir)

	return removed, nil
}

// cleanEmptyDirs removes empty directories from bottom up.
func cleanEmptyDirs(dir string) {
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || !info.IsDir() {
			return nil
		}

		// Try to remove (will fail if not empty)
		os.Remove(path)
		return nil
	})
}

// ListRepos returns all repositories in the cache.
func (c *HFCache) ListRepos() ([]*RepoDir, error) {
	var repos []*RepoDir

	hubDir := c.HubDir()
	if _, err := os.Stat(hubDir); errors.Is(err, os.ErrNotExist) {
		return repos, nil
	}

	entries, err := os.ReadDir(hubDir)
	if err != nil {
		return nil, fmt.Errorf("read hub directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		repoType, owner, name, ok := parseRepoDirName(entry.Name())
		if !ok {
			continue
		}

		repoDir, err := c.Repo(owner+"/"+name, repoType)
		if err != nil {
			continue
		}

		repos = append(repos, repoDir)
	}

	return repos, nil
}
