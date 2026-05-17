// Copyright 2025
// SPDX-License-Identifier: Apache-2.0

package hfdownloader

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"runtime"
	"time"
)

// ErrDownloadBackendNotImplemented marks the in-progress download backend rewrite.
var ErrDownloadBackendNotImplemented = errors.New("download backend rewrite in progress")

type downloadBackend struct {
	httpc    *http.Client
	settings Settings
}

type downloadExecutionPlan struct {
	plan               *Plan
	multipartThreshold int64
	partSize           int64
	useHFCache         bool
}

func newDownloadBackend(cfg Settings) *downloadBackend {
	return &downloadBackend{
		httpc:    buildHTTPClientWithProxy(cfg.Proxy),
		settings: cfg,
	}
}

// Download scans and downloads files from a HuggingFace repo.
//
// The low-level transfer backend is currently being rewritten. For now this
// function still validates input, computes the remote plan, and emits the same
// planning progress events that the active jobs UI depends on before returning
// a placeholder error for the execution phase.
func Download(ctx context.Context, job Job, cfg Settings, progress ProgressFunc) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := validate(job, cfg); err != nil {
		return err
	}

	job, cfg = normalizeDownloadInputs(job, cfg)
	emit := progressEmitter(job, progress)
	backend := newDownloadBackend(cfg)

	execPlan, err := backend.prepare(ctx, job, emit)
	if err != nil {
		return err
	}
	if err := backend.execute(ctx, job, execPlan, emit); err != nil {
		emit(ProgressEvent{
			Event:   "error",
			Level:   "error",
			Message: err.Error(),
		})
		return err
	}

	emit(ProgressEvent{
		Event:   "done",
		Message: "download complete",
	})
	return nil
}

func normalizeDownloadInputs(job Job, cfg Settings) (Job, Settings) {
	if job.Revision == "" {
		job.Revision = "main"
	}
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 8
	}
	if cfg.MaxActiveDownloads <= 0 {
		cfg.MaxActiveDownloads = runtime.GOMAXPROCS(0)
	}
	return job, cfg
}

func progressEmitter(job Job, progress ProgressFunc) func(ProgressEvent) {
	return func(ev ProgressEvent) {
		if progress == nil {
			return
		}
		if ev.Time.IsZero() {
			ev.Time = time.Now().UTC()
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

func (b *downloadBackend) prepare(ctx context.Context, job Job, emit func(ProgressEvent)) (*downloadExecutionPlan, error) {
	emit(ProgressEvent{
		Event:   "scan_start",
		Message: "scanning repo",
	})

	thresholdBytes, err := parseSizeString(b.settings.MultipartThreshold, 256<<20)
	if err != nil {
		return nil, fmt.Errorf("invalid multipart-threshold: %w", err)
	}

	partSize, err := parseSizeString(b.settings.PartSize, 32<<20)
	if err != nil {
		return nil, fmt.Errorf("invalid part-size: %w", err)
	}
	if partSize < 1<<20 {
		partSize = 1 << 20
	}

	plan, err := scanRepo(ctx, b.httpc, b.settings.Token, job, b.settings)
	if err != nil {
		return nil, err
	}

	for _, item := range plan.Items {
		emit(ProgressEvent{
			Event: "plan_item",
			Path:  displayPath(job, item),
			Total: item.Size,
			IsLFS: item.LFS,
		})
	}

	return &downloadExecutionPlan{
		plan:               plan,
		multipartThreshold: thresholdBytes,
		partSize:           partSize,
		useHFCache:         b.settings.CacheDir != "" || b.settings.OutputDir == "",
	}, nil
}

func (b *downloadBackend) execute(ctx context.Context, _ Job, execPlan *downloadExecutionPlan, _ func(ProgressEvent)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if execPlan == nil || execPlan.plan == nil {
		return fmt.Errorf("%w: missing execution plan", ErrDownloadBackendNotImplemented)
	}
	return fmt.Errorf("%w: transfer execution placeholder for %d planned files", ErrDownloadBackendNotImplemented, len(execPlan.plan.Items))
}

func displayPath(job Job, item PlanItem) string {
	if job.AppendFilterSubdir && item.Subdir != "" {
		return filepathToSlashJoin(item.Subdir, item.RelativePath)
	}
	return item.RelativePath
}

func filepathToSlashJoin(parts ...string) string {
	if len(parts) == 0 {
		return ""
	}
	joined := parts[0]
	for i := 1; i < len(parts); i++ {
		if joined == "" {
			joined = parts[i]
			continue
		}
		if parts[i] == "" {
			continue
		}
		joined += "/" + parts[i]
	}
	return joined
}

// downloadSingle is reserved for the rewritten single-stream transfer path.
func downloadSingle(context.Context, *http.Client, string, Job, Settings, PlanItem, string, func(ProgressEvent)) (string, error) {
	return "", fmt.Errorf("%w: single-part transfer path placeholder", ErrDownloadBackendNotImplemented)
}

// downloadMultipart is reserved for the rewritten multipart transfer path.
func downloadMultipart(context.Context, *http.Client, string, Job, Settings, PlanItem, string, func(ProgressEvent), int64) (string, error) {
	return "", fmt.Errorf("%w: multipart transfer path placeholder", ErrDownloadBackendNotImplemented)
}
