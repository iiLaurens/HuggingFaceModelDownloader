// Copyright 2025
// SPDX-License-Identifier: Apache-2.0

package hfdownloader

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// progressReader wraps an io.Reader and emits progress events during reads.
type progressReader struct {
	reader     io.Reader
	total      int64
	downloaded int64
	path       string
	emit       func(ProgressEvent)
	lastEmit   time.Time
	interval   time.Duration
}

func newProgressReader(r io.Reader, total int64, path string, emit func(ProgressEvent)) *progressReader {
	return &progressReader{
		reader:   r,
		total:    total,
		path:     path,
		emit:     emit,
		lastEmit: time.Now(),
		interval: 200 * time.Millisecond, // Emit at most 5 times per second
	}
}

func (pr *progressReader) Read(p []byte) (n int, err error) {
	n, err = pr.reader.Read(p)
	if n > 0 {
		pr.downloaded += int64(n)
		// Throttle emissions to avoid flooding
		if time.Since(pr.lastEmit) >= pr.interval || err == io.EOF {
			pr.emit(ProgressEvent{
				Event:      "file_progress",
				Path:       pr.path,
				Downloaded: pr.downloaded,
				Total:      pr.total,
			})
			pr.lastEmit = time.Now()
		}
	}
	return n, err
}

// countingReader wraps an io.Reader and atomically increments a counter by
// the number of bytes read. Used to track in-flight download bytes across
// multiple concurrent workers without holding any locks.
type countingReader struct {
	r       io.Reader
	counter *int64
}

func (cr *countingReader) Read(p []byte) (n int, err error) {
	n, err = cr.r.Read(p)
	if n > 0 {
		atomic.AddInt64(cr.counter, int64(n))
	}
	return
}

// Download scans and downloads files from a HuggingFace repo.
//
// v3.0+: Files are stored in HuggingFace Hub cache structure by default:
//   - Blobs: hub/models--{owner}--{repo}/blobs/{sha256}
//   - Snapshots: hub/models--{owner}--{repo}/snapshots/{commit}/{path} (symlinks)
//   - Friendly: models/{owner}/{repo}/{path} (symlinks)
//
// Legacy mode (OutputDir set): Falls back to flat directory structure.
//
// Cancellation: all loops/sleeps/requests are tied to ctx for fast abort.
func Download(ctx context.Context, job Job, cfg Settings, progress ProgressFunc) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := validate(job, cfg); err != nil {
		return err
	}

	// Apply defaults
	if job.Revision == "" {
		job.Revision = "main"
	}
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 8
	}
	if cfg.MaxActiveDownloads <= 0 {
		cfg.MaxActiveDownloads = runtime.GOMAXPROCS(0)
	}

	// Determine storage mode: HF cache (new) vs flat directory (legacy)
	// Use HF cache mode when:
	// 1. --cache-dir is explicitly set, OR
	// 2. --output is NOT set (default to HF cache)
	useHFCache := cfg.CacheDir != "" || cfg.OutputDir == ""
	var hfCache *HFCache
	var repoDir *RepoDir

	if useHFCache {
		var err error
		hfCache, err = cfg.BuildHFCache()
		if err != nil {
			return fmt.Errorf("build hf cache: %w", err)
		}
		repoType := RepoTypeModel
		if job.IsDataset {
			repoType = RepoTypeDataset
		}
		repoDir, err = hfCache.Repo(job.Repo, repoType)
		if err != nil {
			return fmt.Errorf("create repo dir: %w", err)
		}
		if err := repoDir.EnsureDirs(); err != nil {
			return fmt.Errorf("ensure repo dirs: %w", err)
		}
	} else {
		// Legacy mode: use OutputDir
		if cfg.OutputDir == "" {
			cfg.OutputDir = "Storage"
		}
	}

	thresholdBytes, err := parseSizeString(cfg.MultipartThreshold, 256<<20)
	if err != nil {
		return fmt.Errorf("invalid multipart-threshold: %w", err)
	}

	partSize, err := parseSizeString(cfg.PartSize, 32<<20)
	if err != nil {
		return fmt.Errorf("invalid part-size: %w", err)
	}
	if partSize < 1<<20 {
		partSize = 1 << 20 // floor at 1 MiB to avoid degenerate behaviour
	}

	httpc := buildHTTPClientWithProxy(cfg.Proxy)

	emit := func(ev ProgressEvent) {
		if progress != nil {
			if ev.Time.IsZero() {
				ev.Time = time.Now()
			}
			if ev.Repo == "" {
				ev.Repo = job.Repo
			}
			if ev.Revision == "" {
				ev.Revision = job.Revision
			}
			progress(ev)
		}
	}

	emit(ProgressEvent{Event: "scan_start", Message: "scanning repo"})

	plan, err := scanRepo(ctx, httpc, cfg.Token, job, cfg)
	if err != nil {
		return err
	}

	// Emit ALL plan_item events upfront so TUI knows total size immediately
	for _, item := range plan.Items {
		displayRel := item.RelativePath
		if job.AppendFilterSubdir && item.Subdir != "" {
			displayRel = filepath.ToSlash(filepath.Join(item.Subdir, item.RelativePath))
		}
		emit(ProgressEvent{Event: "plan_item", Path: displayRel, Total: item.Size})
	}

	// Ensure destination root exists (only for legacy mode)
	// HF cache mode already created directories via repoDir.EnsureDirs()
	if !useHFCache {
		if err := os.MkdirAll(destinationBase(job, cfg), 0o755); err != nil {
			return err
		}
	}

	// Overall concurrency limiter (ctx-aware acquisition)
	type token struct{}
	lim := make(chan token, cfg.MaxActiveDownloads)

	var wg sync.WaitGroup
	errCh := make(chan error, len(plan.Items))

	// To print "skip" only once per final path per run
	var skipOnce sync.Map

	var skippedCount int64
	var downloadedCount int64

	// Build manifest during download (thread-safe)
	// Manifest is always written unless explicitly disabled with NoManifest
	var manifestBuilder *ManifestBuilder
	var manifestMu sync.Mutex
	if useHFCache && !cfg.NoManifest {
		manifestBuilder = NewManifestBuilder(job, cfg.Command)
		manifestBuilder.SetCommit(plan.Commit)
	}

LOOP:
	for _, item := range plan.Items {
		// Stop scheduling more work once canceled
		select {
		case <-ctx.Done():
			break LOOP
		default:
		}

		it := item // capture for goroutine

		// Acquire a slot or abort if canceled
		select {
		case lim <- token{}:
		case <-ctx.Done():
			break LOOP
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-lim }()

			// Per-file context; ensures all inner loops stop on cancellation
			fileCtx, fileCancel := context.WithCancel(ctx)
			defer fileCancel()

			finalRel := it.RelativePath
			filterSubdir := ""
			if job.AppendFilterSubdir && it.Subdir != "" {
				filterSubdir = it.Subdir
				finalRel = filepath.ToSlash(filepath.Join(it.Subdir, it.RelativePath))
			}

			var dst string
			var skipCheck func() (bool, string, error)

			if useHFCache {
				// HF Cache mode: check blob existence
				skipCheck = func() (bool, string, error) {
					if it.SHA256 != "" {
						status, _, err := repoDir.CheckBlob(it.SHA256)
						if err != nil {
							return false, "", err
						}
						if status == BlobComplete {
							// Blob exists, but ensure symlinks are in place
							if err := repoDir.createSnapshotSymlink(plan.Commit, it.RelativePath, it.SHA256); err == nil {
								if !cfg.NoFriendlyView {
									repoDir.CreateFriendlySymlink(plan.Commit, it.RelativePath, filterSubdir)
								}
							}
							return true, "blob exists", nil
						}
						if status == BlobDownloading {
							return true, "downloading by another process", nil
						}
					}
					return false, "", nil
				}
				// Download to temp location, will be moved to blob later
				// Use SHA256 as temp name to avoid collisions (e.g., multiple config.json files)
				tmpName := "tmp-" + it.SHA256
				if it.SHA256 == "" {
					// Fallback: sanitize path to avoid collisions
					tmpName = "tmp-" + strings.ReplaceAll(it.RelativePath, "/", "_")
				}
				dst = filepath.Join(repoDir.BlobsDir(), tmpName)
			} else {
				// Legacy mode: flat directory structure
				base := destinationBase(job, cfg)
				dst = filepath.Join(base, finalRel)
				skipCheck = func() (bool, string, error) {
					return shouldSkipLocal(it, dst)
				}
			}

			if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
				select {
				case errCh <- err:
				default:
				}
				return
			}

			// Check if we can skip
			alreadyOK, reason, err := skipCheck()
			if err != nil {
				select {
				case errCh <- err:
				default:
				}
				return
			}
			if alreadyOK {
				if _, loaded := skipOnce.LoadOrStore(finalRel, struct{}{}); !loaded {
					emit(ProgressEvent{Event: "file_done", Path: finalRel, Message: "skip (" + reason + ")"})
					atomic.AddInt64(&skippedCount, 1)
					// Add to manifest (skipped files are still part of the download job)
					if manifestBuilder != nil {
						manifestMu.Lock()
						manifestBuilder.AddFile(it.RelativePath, it.SHA256, it.Size, it.LFS)
						manifestMu.Unlock()
					}
				}
				return
			}

			emit(ProgressEvent{Event: "file_start", Path: finalRel, Total: it.Size})

			// Create a copy with updated RelativePath for progress display
			itForIO := it
			itForIO.RelativePath = finalRel

			// Choose single/multipart path
			var computedSHA string
			var dlErr error
			if it.Size >= thresholdBytes && it.AcceptRanges {
				computedSHA, dlErr = downloadMultipart(fileCtx, httpc, cfg.Token, job, cfg, itForIO, dst, emit, partSize)
			} else {
				computedSHA, dlErr = downloadSingle(fileCtx, httpc, cfg.Token, job, cfg, itForIO, dst, emit)
			}
			if dlErr != nil {
				select {
				case errCh <- fmt.Errorf("download %s: %w", finalRel, dlErr):
				default:
				}
				return
			}

			// Verify after download — use the hash computed during streaming to
			// avoid a full second read of the file.
			if it.LFS && it.SHA256 != "" {
				if !strings.EqualFold(computedSHA, it.SHA256) {
					select {
					case errCh <- fmt.Errorf("sha256 verify failed: %s: expected %s got %s", finalRel, it.SHA256, computedSHA):
					default:
					}
					return
				}
			} else if cfg.Verify == "size" && it.Size > 0 {
				fi, err := os.Stat(dst)
				if err != nil || fi.Size() != it.Size {
					select {
					case errCh <- fmt.Errorf("size mismatch for %s", finalRel):
					default:
					}
					return
				}
			} else if cfg.Verify == "sha256" {
				_, remoteSha, _ := headForETag(fileCtx, httpc, cfg.Token, itForIO)
				if remoteSha != "" && !strings.EqualFold(computedSHA, remoteSha) {
					select {
					case errCh <- fmt.Errorf("sha256 verify failed: %s: expected %s got %s", finalRel, remoteSha, computedSHA):
					default:
					}
					return
				}
			}

			// For HF Cache mode: move to blob and create symlinks.
			// Pass computedSHA so StoreDownloadedFile skips its own re-read.
			var finalSHA256 string
			if useHFCache {
				result, err := repoDir.StoreDownloadedFile(dst, it.RelativePath, plan.Commit, computedSHA, filterSubdir, cfg.NoFriendlyView)
				if err != nil {
					select {
					case errCh <- fmt.Errorf("store file %s: %w", finalRel, err):
					default:
					}
					return
				}
				finalSHA256 = result.SHA256 // Use computed SHA256 from store result
			} else {
				finalSHA256 = it.SHA256
			}

			// Add to manifest with actual LFS info from API and final SHA256
			if manifestBuilder != nil {
				manifestMu.Lock()
				manifestBuilder.AddFile(it.RelativePath, finalSHA256, it.Size, it.LFS)
				manifestMu.Unlock()
			}

			emit(ProgressEvent{Event: "file_done", Path: finalRel})
			atomic.AddInt64(&downloadedCount, 1)
		}()
	}

	wg.Wait()
	close(errCh)

	// Drain errors
	var firstErr error
	for e := range errCh {
		if e != nil {
			firstErr = e
			break
		}
	}
	if firstErr != nil {
		emit(ProgressEvent{Level: "error", Event: "error", Message: firstErr.Error()})
		return firstErr
	}

	if ctx.Err() != nil {
		return ctx.Err()
	}

	// For HF Cache mode: write ref file and ensure friendly directory exists
	if useHFCache && repoDir != nil {
		// Write refs/main (or the revision used)
		ref := job.Revision
		if ref == "" {
			ref = "main"
		}
		if err := repoDir.WriteRef(ref, plan.Commit); err != nil {
			emit(ProgressEvent{Level: "warn", Event: "warning", Message: fmt.Sprintf("failed to write ref: %v", err)})
		}
		// Ensure friendly directory structure exists (unless disabled)
		if !cfg.NoFriendlyView {
			if err := repoDir.EnsureFriendlyDir(); err != nil {
				emit(ProgressEvent{Level: "warn", Event: "warning", Message: fmt.Sprintf("failed to create friendly dir: %v", err)})
			}
		}
	}

	// Write/update the rebuild shell script if using HF cache (unless friendly view disabled)
	if hfCache != nil && !cfg.NoFriendlyView {
		if _, err := hfCache.WriteRebuildScript(); err != nil {
			emit(ProgressEvent{Level: "warn", Event: "warning", Message: fmt.Sprintf("failed to write rebuild script: %v", err)})
		}
	}

	// Write manifest file (hfd.yaml) if using HF cache (unless friendly view disabled)
	if manifestBuilder != nil && repoDir != nil && !cfg.NoFriendlyView {
		manifest := manifestBuilder.Build()
		if _, err := manifest.Write(repoDir.FriendlyPath()); err != nil {
			emit(ProgressEvent{Level: "warn", Event: "warning", Message: fmt.Sprintf("failed to write manifest: %v", err)})
		}
	}

	emit(ProgressEvent{
		Event:   "done",
		Message: fmt.Sprintf("download complete (downloaded %d, skipped %d)", downloadedCount, skippedCount),
	})
	return nil
}

// downloadSingle downloads a file in a single request and returns the
// SHA-256 hash computed incrementally during the download.
//
// Resume behavior: if a .part file already exists from a previous interrupted
// run, its bytes are preserved and the HTTP request uses a Range header to
// fetch only the remaining bytes. If the server ignores the Range header and
// responds with 200 (full body), the .part file is truncated and the download
// restarts from zero.
func downloadSingle(ctx context.Context, httpc *http.Client, token string, job Job, cfg Settings, it PlanItem, dst string, emit func(ProgressEvent)) (string, error) {
	tmp := dst + ".part"
	out, err := os.OpenFile(tmp, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return "", err
	}
	defer out.Close()

	fi, err := out.Stat()
	if err != nil {
		return "", err
	}
	pos := fi.Size()

	// If the partial is already exactly the right size, compute SHA and finalize.
	if it.Size > 0 && pos == it.Size {
		h := sha256.New()
		if _, err := out.Seek(0, io.SeekStart); err != nil {
			return "", err
		}
		if _, err := io.Copy(h, out); err != nil {
			return "", err
		}
		out.Close()
		if err := os.Rename(tmp, dst); err != nil {
			return "", err
		}
		return hex.EncodeToString(h.Sum(nil)), nil
	}
	// If the partial is larger than expected (stale/corrupt), start over.
	if it.Size > 0 && pos > it.Size {
		if err := out.Truncate(0); err != nil {
			return "", err
		}
		pos = 0
	}
	if _, err := out.Seek(pos, io.SeekStart); err != nil {
		return "", err
	}
	if pos > 0 {
		emit(ProgressEvent{Event: "file_progress", Path: it.RelativePath, Downloaded: pos, Total: it.Size})
	}

	retry := newRetry(cfg)
	var lastErr error

	for attempt := 0; attempt <= cfg.Retries; attempt++ {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		req, _ := http.NewRequestWithContext(ctx, "GET", it.URL, nil)
		addAuth(req, token)
		if pos > 0 {
			if it.Size > 0 {
				req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", pos, it.Size-1))
			} else {
				req.Header.Set("Range", fmt.Sprintf("bytes=%d-", pos))
			}
		}

		resp, err := httpc.Do(req)
		if err != nil {
			lastErr = err
		} else {
			// If we asked for a range but the server returned the whole body,
			// throw away any existing partial bytes and start fresh.
			if pos > 0 && resp.StatusCode == http.StatusOK {
				if err := out.Truncate(0); err != nil {
					resp.Body.Close()
					return "", err
				}
				if _, err := out.Seek(0, io.SeekStart); err != nil {
					resp.Body.Close()
					return "", err
				}
				pos = 0
			}
			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				lastErr = fmt.Errorf("bad status: %s", resp.Status)
				resp.Body.Close()
			} else {
				// Seed the hasher with bytes already on disk (resume case).
				h := sha256.New()
				if pos > 0 {
					if _, err := out.Seek(0, io.SeekStart); err != nil {
						resp.Body.Close()
						return "", err
					}
					if _, err := io.CopyN(h, out, pos); err != nil {
						resp.Body.Close()
						return "", err
					}
					if _, err := out.Seek(pos, io.SeekStart); err != nil {
						resp.Body.Close()
						return "", err
					}
				}
				pr := newProgressReader(resp.Body, it.Size, it.RelativePath, emit)
				pr.downloaded = pos // emitted progress reflects cumulative bytes
				_, cerr := io.Copy(io.MultiWriter(out, h), pr)
				resp.Body.Close()
				if cerr == nil {
					out.Close()
					if err := os.Rename(tmp, dst); err != nil {
						return "", err
					}
					return hex.EncodeToString(h.Sum(nil)), nil
				}
				lastErr = cerr
				// Update pos to current file position so the next retry issues
				// a Range request for the remaining bytes instead of duplicating.
				if cur, serr := out.Seek(0, io.SeekCurrent); serr == nil {
					pos = cur
				}
			}
		}

		if attempt < cfg.Retries {
			emit(ProgressEvent{Event: "retry", Path: it.RelativePath, Attempt: attempt + 1, Message: lastErr.Error()})
			if d := retry.Next(); !sleepCtx(ctx, d) {
				return "", ctx.Err()
			}
		}
	}
	return "", lastErr
}

// downloadMultipart downloads a file using multiple parallel range requests and
// returns the SHA-256 hash computed during assembly.
// Concurrency is capped to cfg.Concurrency to avoid spawning thousands of
// goroutines (and exhausting the HTTP transport) when downloading large files
// or multiple repositories simultaneously.
func downloadMultipart(ctx context.Context, httpc *http.Client, token string, job Job, cfg Settings, it PlanItem, dst string, emit func(ProgressEvent), partSize int64) (string, error) {
	// HEAD to resolve size
	req, _ := http.NewRequestWithContext(ctx, "HEAD", it.URL, nil)
	addAuth(req, token)
	resp, err := httpc.Do(req)
	if err != nil {
		return "", err
	}
	resp.Body.Close()

	if it.Size == 0 {
		if clen := resp.Header.Get("Content-Length"); clen != "" {
			var n int64
			fmt.Sscan(clen, &n)
			it.Size = n
		}
	}
	if it.Size == 0 {
		return downloadSingle(ctx, httpc, token, job, cfg, it, dst, emit)
	}

	// Plan parts using configurable partSize
	numParts := int((it.Size + partSize - 1) / partSize)
	if numParts < 1 {
		numParts = 1
	}

	tmpParts := make([]string, numParts)
	for i := 0; i < numParts; i++ {
		tmpParts[i] = fmt.Sprintf("%s.part-%02d", dst, i)
	}

	// Seed the progress counter with bytes already on disk from any previous
	// interrupted run so the ticker reflects resumed progress immediately.
	var downloadedBytes int64
	for _, p := range tmpParts {
		if fi, err := os.Stat(p); err == nil {
			atomic.AddInt64(&downloadedBytes, fi.Size())
		}
	}

	// Download parts in parallel, capped to cfg.Concurrency goroutines.
	concurrency := cfg.Concurrency
	if concurrency <= 0 {
		concurrency = 32
	}
	if concurrency > numParts {
		concurrency = numParts
	}
	type semToken struct{}
	sem := make(chan semToken, concurrency)

	var wg sync.WaitGroup
	errCh := make(chan error, numParts)

	for i := 0; i < numParts; i++ {
		i := i
		start := int64(i) * partSize
		end := start + partSize - 1
		if end >= it.Size {
			end = it.Size - 1
		}

		// Acquire a concurrency slot before launching the goroutine.
		// If the context is cancelled stop scheduling new parts immediately.
		select {
		case sem <- semToken{}:
		case <-ctx.Done():
			goto partsLaunched
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			tmp := tmpParts[i]
			expected := end - start + 1

			// Open or create the part file without truncating; we may be
			// resuming from a previous interrupted run.
			out, err := os.OpenFile(tmp, os.O_RDWR|os.O_CREATE, 0o644)
			if err != nil {
				select {
				case errCh <- err:
				default:
				}
				return
			}
			defer out.Close()

			fi, err := out.Stat()
			if err != nil {
				select {
				case errCh <- err:
				default:
				}
				return
			}
			pos := fi.Size()
			// Already fully downloaded.
			if pos == expected {
				return
			}
			// Oversize (stale/corrupt) — reset. Un-count the stale bytes that
			// were included in the initial seed so the progress stays accurate.
			if pos > expected {
				atomic.AddInt64(&downloadedBytes, -pos)
				if err := out.Truncate(0); err != nil {
					select {
					case errCh <- err:
					default:
					}
					return
				}
				pos = 0
			}
			if _, err := out.Seek(pos, io.SeekStart); err != nil {
				select {
				case errCh <- err:
				default:
				}
				return
			}

			retry := newRetry(cfg)
			var lastErr error

			for attempt := 0; attempt <= cfg.Retries; attempt++ {
				select {
				case <-ctx.Done():
					return
				default:
				}

				rq, _ := http.NewRequestWithContext(ctx, "GET", it.URL, nil)
				addAuth(rq, token)
				rq.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start+pos, end))

				rs, err := httpc.Do(rq)
				if err != nil {
					lastErr = err
				} else if rs.StatusCode != 206 {
					lastErr = fmt.Errorf("range not supported (status %s)", rs.Status)
					rs.Body.Close()
				} else {
					// Count bytes as they arrive, so the progress ticker reflects
					// in-flight data rather than only bytes flushed to disk.
					cr := &countingReader{r: rs.Body, counter: &downloadedBytes}
					_, cerr := io.Copy(out, cr)
					rs.Body.Close()
					if cerr == nil {
						return
					}
					lastErr = cerr
					// Advance pos by what we actually wrote so the next retry
					// Range request picks up from the correct offset.
					if cur, serr := out.Seek(0, io.SeekCurrent); serr == nil {
						pos = cur
					}
				}

				if attempt < cfg.Retries {
					emit(ProgressEvent{Event: "retry", Path: it.RelativePath, Attempt: attempt + 1, Message: lastErr.Error()})
					if d := retry.Next(); !sleepCtx(ctx, d) {
						return
					}
				}
			}

			select {
			case errCh <- lastErr:
			default:
			}
		}()
	}
partsLaunched:

	// Emit periodic progress while parts download.
	tickerDone := make(chan struct{})
	var tickerWG sync.WaitGroup
	tickerWG.Add(1)
	go func() {
		defer tickerWG.Done()
		t := time.NewTicker(200 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tickerDone:
				return
			case <-t.C:
				emit(ProgressEvent{Event: "file_progress", Path: it.RelativePath, Downloaded: atomic.LoadInt64(&downloadedBytes), Total: it.Size})
			}
		}
	}()

	wg.Wait()

	// Stop the progress ticker before touching part files.
	close(tickerDone)
	tickerWG.Wait()

	// If the context was cancelled, return the cancellation error immediately.
	if ctx.Err() != nil {
		return "", ctx.Err()
	}

	select {
	case e := <-errCh:
		return "", e
	default:
	}

	// Emit one explicit full-progress reading so the caller's last observed
	// file_progress value is the full byte count.
	emit(ProgressEvent{Event: "file_progress", Path: it.RelativePath, Downloaded: it.Size, Total: it.Size})

	// Assemble parts, computing SHA-256 during assembly to avoid a second read.
	out, err := os.Create(dst + ".part")
	if err != nil {
		return "", err
	}

	h := sha256.New()
	mw := io.MultiWriter(out, h)

	for i := 0; i < numParts; i++ {
		p := tmpParts[i]
		in, err := os.Open(p)
		if err != nil {
			out.Close()
			return "", err
		}
		if _, err := io.Copy(mw, in); err != nil {
			in.Close()
			out.Close()
			return "", err
		}
		in.Close()
	}
	out.Close()

	if err := os.Rename(dst+".part", dst); err != nil {
		return "", err
	}

	for _, p := range tmpParts {
		_ = os.Remove(p)
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}
