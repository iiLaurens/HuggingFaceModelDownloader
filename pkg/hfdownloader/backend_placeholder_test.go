// Copyright 2025
// SPDX-License-Identifier: Apache-2.0

package hfdownloader

import (
	"context"
	"errors"
	"testing"
)

func TestDownload_EmitsPlanBeforePlaceholderError(t *testing.T) {
	srv := mockHFAPI(t, map[string]interface{}{
		"/api/models/owner/repo/revision/main": RepoInfo{SHA: "commit-sha"},
		"/api/models/owner/repo/tree/main": []hfNode{
			{
				Type: "file",
				Path: "weights/model.bin",
				Size: 123,
			},
		},
	})
	defer srv.Close()

	var events []ProgressEvent
	err := Download(context.Background(), Job{Repo: "owner/repo"}, Settings{Endpoint: srv.URL}, func(ev ProgressEvent) {
		events = append(events, ev)
	})
	if !errors.Is(err, ErrDownloadBackendNotImplemented) {
		t.Fatalf("Download() error = %v, want ErrDownloadBackendNotImplemented", err)
	}

	if len(events) < 3 {
		t.Fatalf("expected scan_start, plan_item, and error events; got %d", len(events))
	}
	if events[0].Event != "scan_start" {
		t.Fatalf("first event = %q, want scan_start", events[0].Event)
	}
	if events[1].Event != "plan_item" {
		t.Fatalf("second event = %q, want plan_item", events[1].Event)
	}
	if events[1].Path != "weights/model.bin" {
		t.Fatalf("plan_item path = %q, want weights/model.bin", events[1].Path)
	}
	if events[len(events)-1].Event != "error" {
		t.Fatalf("last event = %q, want error", events[len(events)-1].Event)
	}
}
