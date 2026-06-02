// Copyright 2025
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"

	"github.com/bodaay/HuggingFaceModelDownloader/pkg/hfdownloader"
)

// JobStatus represents the state of a download job.
type JobStatus string

const (
	JobStatusQueued    JobStatus = "queued"
	JobStatusRunning   JobStatus = "running"
	JobStatusCompleted JobStatus = "completed"
	JobStatusFailed    JobStatus = "failed"
	JobStatusCancelled JobStatus = "cancelled"
)

// Job represents a download job.
type Job struct {
	ID        string            `json:"id"`
	Repo      string            `json:"repo"`
	Revision  string            `json:"revision"`
	IsDataset bool              `json:"isDataset,omitempty"`
	Filters   []string          `json:"filters,omitempty"`
	Excludes  []string          `json:"excludes,omitempty"`
	Paths     []string          `json:"paths,omitempty"`
	OutputDir string            `json:"outputDir"`
	Status    JobStatus         `json:"status"`
	Progress  JobProgress       `json:"progress"`
	Error     string            `json:"error,omitempty"`
	CreatedAt time.Time         `json:"createdAt"`
	StartedAt *time.Time        `json:"startedAt,omitempty"`
	EndedAt   *time.Time        `json:"endedAt,omitempty"`
	Files     []JobFileProgress `json:"files,omitempty"`

	cancel     context.CancelFunc `json:"-"`
	generation int                `json:"-"` // Tracks which runJob instance is current
}

// JobProgress holds aggregate progress info.
type JobProgress struct {
	TotalFiles      int   `json:"totalFiles"`
	CompletedFiles  int   `json:"completedFiles"`
	TotalBytes      int64 `json:"totalBytes"`
	DownloadedBytes int64 `json:"downloadedBytes"`
	BytesPerSecond  int64 `json:"bytesPerSecond"`
}

// JobFileProgress holds per-file progress.
type JobFileProgress struct {
	Path       string `json:"path"`
	TotalBytes int64  `json:"totalBytes"`
	Downloaded int64  `json:"downloaded"`
	Status     string `json:"status"` // pending, active, complete, skipped, error
}

// JobManager manages download jobs.
type JobManager struct {
	mu          sync.RWMutex
	jobs        map[string]*Job
	config      Config
	listeners   []chan *Job
	listenerMu  sync.RWMutex
	wsHub       *WSHub
	wsCoalescer *jobCoalescer
	// runWG tracks in-flight runJob goroutines so shutdown paths (and
	// tests) can wait for every download to actually unwind — not just
	// for Status to flip to Cancelled. Without this a t.TempDir cleanup
	// can race a still-in-flight mkdir inside the downloader and fail
	// with "directory not empty".
	runWG sync.WaitGroup
}

// wsBroadcastMinGap is the minimum interval between consecutive WebSocket
// broadcasts for the same job. Progress events arriving inside this window
// are coalesced — only the latest job state is flushed when the window
// elapses. Terminal status changes (completed, failed, cancelled)
// bypass this gate and are sent immediately. See github issue #62.
const wsBroadcastMinGap = 250 * time.Millisecond

// NewJobManager creates a new job manager.
func NewJobManager(cfg Config, wsHub *WSHub) *JobManager {
	m := &JobManager{
		jobs:   make(map[string]*Job),
		config: cfg,
		wsHub:  wsHub,
	}
	if wsHub != nil {
		m.wsCoalescer = newJobCoalescer(wsBroadcastMinGap, func(j *Job) {
			wsHub.BroadcastJob(j)
		})
	}
	return m
}

// generateID creates a short random ID.
func generateID() string {
	b := make([]byte, 6)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// cloneJobLocked returns a fully-independent copy of a Job. Must be called
// while m.mu is held (any lock, read or write) so the fields being copied
// are stable. The returned *Job can be safely handed to JSON encoders or
// WebSocket broadcasters without racing against runJob's in-place mutations
// of the live Job stored in m.jobs. Slice fields are deep-copied so that
// subsequent mutations of the live job's slices can't leak through a shared
// backing array.
func (m *JobManager) cloneJobLocked(j *Job) *Job {
	if j == nil {
		return nil
	}
	clone := *j
	clone.cancel = nil
	if j.Filters != nil {
		clone.Filters = append([]string(nil), j.Filters...)
	}
	if j.Excludes != nil {
		clone.Excludes = append([]string(nil), j.Excludes...)
	}
	if j.Paths != nil {
		clone.Paths = append([]string(nil), j.Paths...)
	}
	if j.Files != nil {
		clone.Files = append([]JobFileProgress(nil), j.Files...)
	}
	if j.StartedAt != nil {
		t := *j.StartedAt
		clone.StartedAt = &t
	}
	if j.EndedAt != nil {
		t := *j.EndedAt
		clone.EndedAt = &t
	}
	return &clone
}

// CreateJob creates a new download job.
// Returns existing job if same repo+revision+dataset+paths is already in progress.
// Jobs that specify explicit Paths are never de-duplicated against full-repo jobs
// (and vice versa), so that a targeted update always gets its own job entry.
func (m *JobManager) CreateJob(req DownloadRequest) (*Job, bool, error) {
	revision := req.Revision
	if revision == "" {
		revision = "main"
	}

	// Use HuggingFace cache directory (v3 mode)
	cacheDir := m.config.CacheDir
	if cacheDir == "" {
		cacheDir = hfdownloader.DefaultCacheDir()
	}

	// De-duplicate: only match when both jobs are full-repo downloads (no Paths)
	// OR when both have identical Paths slices.  A paths-based update is always
	// treated as distinct from a full-repo job so it gets its own visible entry.
	m.mu.Lock()
	for _, existing := range m.jobs {
		if existing.Repo == req.Repo &&
			existing.Revision == revision &&
			existing.IsDataset == req.Dataset &&
			(existing.Status == JobStatusQueued || existing.Status == JobStatusRunning) &&
			pathsEqual(existing.Paths, req.Paths) {
			snapshot := m.cloneJobLocked(existing)
			m.mu.Unlock()
			return snapshot, true, nil
		}
	}

	job := &Job{
		ID:        generateID(),
		Repo:      req.Repo,
		Revision:  revision,
		IsDataset: req.Dataset,
		Filters:   req.Filters,
		Excludes:  req.Excludes,
		Paths:     req.Paths,
		OutputDir: cacheDir, // HuggingFace cache directory
		Status:    JobStatusQueued,
		CreatedAt: time.Now(),
		Progress:  JobProgress{},
	}

	m.jobs[job.ID] = job
	snapshot := m.cloneJobLocked(job)
	m.mu.Unlock()

	// Start the job
	m.runWG.Add(1)
	go m.runJob(job)

	return snapshot, false, nil
}

// pathsEqual returns true when two Paths slices represent the same set of
// files.  Two nil/empty slices are considered equal (both mean "full repo").
func pathsEqual(a, b []string) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	if len(a) != len(b) {
		return false
	}
	seen := make(map[string]int, len(a))
	for _, p := range a {
		seen[p]++
	}
	for _, p := range b {
		seen[p]--
		if seen[p] < 0 {
			return false
		}
	}
	return true
}

// GetJob retrieves a snapshot of a job by ID. The returned pointer is a
// standalone copy; the caller can read its fields without racing against
// the runJob goroutine that owns the live version in m.jobs.
func (m *JobManager) GetJob(id string) (*Job, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	job, ok := m.jobs[id]
	if !ok {
		return nil, false
	}
	return m.cloneJobLocked(job), true
}

// ListJobs returns snapshots of all jobs. Each returned *Job is an
// independent copy — safe to JSON-encode or hand to the WebSocket hub
// without holding any lock.
func (m *JobManager) ListJobs() []*Job {
	m.mu.RLock()
	defer m.mu.RUnlock()

	jobs := make([]*Job, 0, len(m.jobs))
	for _, job := range m.jobs {
		jobs = append(jobs, m.cloneJobLocked(job))
	}
	return jobs
}

// HasActiveJobs returns true if any jobs are currently running or queued.
func (m *JobManager) HasActiveJobs() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, job := range m.jobs {
		switch job.Status {
		case JobStatusRunning, JobStatusQueued:
			return true
		}
	}
	return false
}

// CancelJob cancels a running or queued job.
func (m *JobManager) CancelJob(id string) bool {
	m.mu.Lock()
	job, ok := m.jobs[id]
	if !ok {
		m.mu.Unlock()
		return false
	}

	if job.Status != JobStatusQueued && job.Status != JobStatusRunning {
		m.mu.Unlock()
		return false
	}

	if job.cancel != nil {
		job.cancel()
	}
	job.Status = JobStatusCancelled
	now := time.Now()
	job.EndedAt = &now
	snapshot := m.cloneJobLocked(job)
	m.mu.Unlock()

	m.notifyListeners(snapshot)
	return true
}

// DeleteJob removes a job from the list.
func (m *JobManager) DeleteJob(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	job, ok := m.jobs[id]
	if !ok {
		return false
	}

	// Cancel if running
	if job.cancel != nil && (job.Status == JobStatusQueued || job.Status == JobStatusRunning) {
		job.cancel()
	}

	delete(m.jobs, id)
	return true
}

// WaitAll blocks until every in-flight runJob goroutine has returned or
// until timeout elapses. Returns true if all goroutines exited cleanly,
// false on timeout. Primarily for tests and graceful shutdown — lets
// callers observe actual goroutine exit rather than just Status==Cancelled,
// which is set before the downloader's filesystem operations fully unwind.
func (m *JobManager) WaitAll(timeout time.Duration) bool {
	done := make(chan struct{})
	go func() {
		m.runWG.Wait()
		close(done)
	}()
	select {
	case <-done:
		return true
	case <-time.After(timeout):
		return false
	}
}

// DismissJobResult distinguishes the three possible outcomes of a dismiss
// attempt so the HTTP layer can map them to appropriate status codes.
type DismissJobResult int

const (
	// DismissJobOK means the job was in a terminal state and has been removed.
	DismissJobOK DismissJobResult = iota
	// DismissJobNotFound means no job with that ID exists.
	DismissJobNotFound
	// DismissJobStillActive means the job is queued or running; it must be
	// cancelled first (or completed) before it can be dismissed.
	DismissJobStillActive
)

// DismissJob removes a job from the manager if and only if it is in a
// terminal state (completed, failed, cancelled). Dismissal is the
// user's way of hiding a finished job from the UI permanently, and the
// guarantee that matters for github issue #68 is that the job does not
// come back on the next page refresh — so the underlying storage drops it.
// Dismissing a queued or running job is rejected so a stray click can't
// wipe a live download.
func (m *JobManager) DismissJob(id string) bool {
	res, _ := m.DismissJobResult(id)
	return res == DismissJobOK
}

// DismissJobResult is the richer variant of DismissJob that returns the
// reason a dismissal failed, for use by the HTTP handler.
func (m *JobManager) DismissJobResult(id string) (DismissJobResult, *Job) {
	m.mu.Lock()
	defer m.mu.Unlock()
	job, ok := m.jobs[id]
	if !ok {
		return DismissJobNotFound, nil
	}
	if !isTerminalJobStatus(job.Status) {
		return DismissJobStillActive, job
	}
	delete(m.jobs, id)
	return DismissJobOK, job
}

// Subscribe adds a listener for job updates.
func (m *JobManager) Subscribe() chan *Job {
	ch := make(chan *Job, 100)
	m.listenerMu.Lock()
	m.listeners = append(m.listeners, ch)
	m.listenerMu.Unlock()
	return ch
}

// Unsubscribe removes a listener.
func (m *JobManager) Unsubscribe(ch chan *Job) {
	m.listenerMu.Lock()
	defer m.listenerMu.Unlock()

	for i, listener := range m.listeners {
		if listener == ch {
			m.listeners = append(m.listeners[:i], m.listeners[i+1:]...)
			close(ch)
			return
		}
	}
}

// notifyListeners forwards an already-snapshotted job update to channel
// listeners and the WebSocket broadcast path. The caller MUST pass in a
// snapshot (produced by cloneJobLocked while holding m.mu) — this function
// does not take m.mu itself, so it is safe to call from sites that already
// hold m.mu.Lock() (like CancelJob with a deferred unlock).
func (m *JobManager) notifyListeners(snapshot *Job) {
	// Notify channel listeners (tests and other internal subscribers see
	// every raw update; only the WebSocket path is throttled).
	m.listenerMu.RLock()
	for _, ch := range m.listeners {
		select {
		case ch <- snapshot:
		default:
			// Listener is slow, skip
		}
	}
	m.listenerMu.RUnlock()

	// Broadcast to WebSocket clients through the per-job coalescer so the
	// browser isn't asked to re-render at 5Hz × file-count.
	if m.wsCoalescer != nil {
		m.wsCoalescer.schedule(snapshot)
	} else if m.wsHub != nil {
		m.wsHub.BroadcastJob(snapshot)
	}
}

// runJob executes the download job.
func (m *JobManager) runJob(job *Job) {
	defer m.runWG.Done()

	ctx, cancel := context.WithCancel(context.Background())

	// Increment generation and store our generation number
	m.mu.Lock()
	job.cancel = cancel
	job.generation++
	myGeneration := job.generation // Track which generation we are
	job.Status = JobStatusRunning
	now := time.Now()
	job.StartedAt = &now
	startSnap := m.cloneJobLocked(job)
	m.mu.Unlock()
	m.notifyListeners(startSnap)

	// Create hfdownloader job and settings
	dlJob := hfdownloader.Job{
		Repo:               job.Repo,
		Revision:           job.Revision,
		IsDataset:          job.IsDataset,
		Filters:            job.Filters,
		Excludes:           job.Excludes,
		Paths:              job.Paths,
		AppendFilterSubdir: false,
	}

	// Use HuggingFace cache structure (v3 mode) instead of legacy OutputDir
	cacheDir := m.config.CacheDir
	if cacheDir == "" {
		cacheDir = hfdownloader.DefaultCacheDir()
	}

	settings := hfdownloader.Settings{
		CacheDir:           cacheDir, // Use HF cache structure
		Concurrency:        m.config.Concurrency,
		MaxActiveDownloads: m.config.MaxActive,
		Token:              m.config.Token,
		MultipartThreshold: m.config.MultipartThreshold,
		PartSize:           m.config.PartSize,
		Verify:             m.config.Verify,
		Retries:            m.config.Retries,
		BackoffInitial:     "400ms",
		BackoffMax:         "10s",
		Endpoint:           m.config.Endpoint,
		Proxy:              m.config.Proxy,
	}

	// Progress callback - NOTE: must not hold lock when calling notifyListeners
	progressFunc := func(evt hfdownloader.ProgressEvent) {
		m.mu.Lock()

		switch evt.Event {
		case "plan_item":
			job.Progress.TotalFiles++
			job.Progress.TotalBytes += evt.Total
			job.Files = append(job.Files, JobFileProgress{
				Path:       evt.Path,
				TotalBytes: evt.Total,
				Status:     "pending",
			})

		case "file_start":
			for i := range job.Files {
				if job.Files[i].Path == evt.Path {
					job.Files[i].Status = "active"
					break
				}
			}

		case "file_progress":
			for i := range job.Files {
				if job.Files[i].Path == evt.Path {
					job.Files[i].Downloaded = evt.Downloaded
					break
				}
			}
			// Update aggregate
			var total int64
			for _, f := range job.Files {
				total += f.Downloaded
			}
			job.Progress.DownloadedBytes = total

		case "file_done":
			for i := range job.Files {
				if job.Files[i].Path == evt.Path {
					job.Files[i].Status = "complete"
					job.Files[i].Downloaded = job.Files[i].TotalBytes
					break
				}
			}
			job.Progress.CompletedFiles++
			// Recalculate total downloaded
			var total int64
			for _, f := range job.Files {
				total += f.Downloaded
			}
			job.Progress.DownloadedBytes = total
		}

		progressSnap := m.cloneJobLocked(job)
		m.mu.Unlock() // Unlock BEFORE notifying to avoid deadlock
		m.notifyListeners(progressSnap)
	}

	// Run the download
	err := hfdownloader.Run(ctx, dlJob, settings, progressFunc)

	// Update final status
	m.mu.Lock()
	// Don't update status if we're a stale goroutine (a newer runJob has started).
	if job.generation != myGeneration {
		m.mu.Unlock()
		return
	}
	endTime := time.Now()
	job.EndedAt = &endTime
	if ctx.Err() != nil {
		job.Status = JobStatusCancelled
	} else if err != nil {
		job.Status = JobStatusFailed
		job.Error = err.Error()
	} else {
		job.Status = JobStatusCompleted
	}
	endSnap := m.cloneJobLocked(job)
	m.mu.Unlock()

	m.notifyListeners(endSnap)
}

