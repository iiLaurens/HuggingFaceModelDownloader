// Copyright 2025
// SPDX-License-Identifier: Apache-2.0

package smartdl

import (
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// GGUF quantization quality ratings (1-5 stars).
//
// Entries prefixed/suffixed with UD-style modifiers (e.g. Q4_K_XL) are unsloth's
// "Unsloth Dynamic" quants — same bit-budget family as the corresponding K
// variant but with improved layer-by-layer quantization, rated at or above the
// closest standard K_M equivalent.
var quantQuality = map[string]int{
	// 1-bit (unsloth dynamic only)
	"IQ1_S": 1,
	"IQ1_M": 1,

	// 2-bit
	"Q2_K":    1,
	"Q2_K_S":  1,
	"Q2_K_L":  1,
	"Q2_K_XL": 2, // unsloth dynamic, better than plain Q2_K
	"IQ2_S":   1,
	"IQ2_M":   1,
	"IQ2_XS":  1,
	"IQ2_XXS": 1,

	// 3-bit
	"Q3_K_S":  2,
	"Q3_K_M":  2,
	"Q3_K_L":  2,
	"Q3_K_XL": 3, // unsloth dynamic
	"IQ3_S":   2,
	"IQ3_XS":  2,
	"IQ3_XXS": 2,
	"IQ3_M":   2,

	// 4-bit
	"Q4_0":    3,
	"Q4_1":    3,
	"Q4_K_S":  3,
	"Q4_K_M":  4,
	"Q4_K_XL": 4, // unsloth dynamic, comparable to Q4_K_M
	"IQ4_NL":  3,
	"IQ4_XS":  3,

	// 5-bit
	"Q5_0":    4,
	"Q5_1":    4,
	"Q5_K_S":  4,
	"Q5_K_M":  5,
	"Q5_K_XL": 5, // unsloth dynamic

	// 6-bit
	"Q6_K":    5,
	"Q6_K_XL": 5, // unsloth dynamic

	// 8-bit
	"Q8_0":    5,
	"Q8_K_XL": 5, // unsloth dynamic

	// Full / half precision
	"F16":  5,
	"F32":  5,
	"BF16": 5,
}

// quantDescriptions provides human-readable descriptions for quantization levels.
var quantDescriptions = map[string]string{
	// 1-bit
	"IQ1_S": "Unsloth dynamic 1-bit, small",
	"IQ1_M": "Unsloth dynamic 1-bit, medium",

	// 2-bit
	"Q2_K":    "Smallest, significant quality loss",
	"Q2_K_S":  "Smallest, significant quality loss",
	"Q2_K_L":  "Small 2-bit, noticeable quality loss",
	"Q2_K_XL": "Unsloth dynamic 2-bit, improved quality",
	"IQ2_S":   "Importance matrix 2-bit, small",
	"IQ2_M":   "Importance matrix 2-bit, medium",
	"IQ2_XS":  "Importance matrix 2-bit, extra small",
	"IQ2_XXS": "Importance matrix 2-bit, extra extra small",

	// 3-bit
	"Q3_K_S":  "Very small, noticeable quality loss",
	"Q3_K_M":  "Small, noticeable quality loss",
	"Q3_K_L":  "Small, noticeable quality loss",
	"Q3_K_XL": "Unsloth dynamic 3-bit, improved quality",
	"IQ3_S":   "Importance matrix 3-bit, small",
	"IQ3_XS":  "Importance matrix 3-bit, extra small",
	"IQ3_XXS": "Importance matrix 3-bit, extra extra small",
	"IQ3_M":   "Importance matrix 3-bit, medium",

	// 4-bit
	"Q4_0":    "Legacy 4-bit, good balance",
	"Q4_1":    "Legacy 4-bit with scales",
	"Q4_K_S":  "Small 4-bit, good quality",
	"Q4_K_M":  "Medium 4-bit, recommended",
	"Q4_K_XL": "Unsloth dynamic 4-bit, recommended",
	"IQ4_NL":  "Importance matrix 4-bit, non-linear",
	"IQ4_XS":  "Importance matrix 4-bit, extra small",

	// 5-bit
	"Q5_0":    "Legacy 5-bit, very good quality",
	"Q5_1":    "Legacy 5-bit with scales",
	"Q5_K_S":  "Small 5-bit, excellent quality",
	"Q5_K_M":  "Medium 5-bit, excellent quality",
	"Q5_K_XL": "Unsloth dynamic 5-bit, excellent quality",

	// 6-bit
	"Q6_K":    "6-bit, near-lossless",
	"Q6_K_XL": "Unsloth dynamic 6-bit, near-lossless",

	// 8-bit
	"Q8_0":    "8-bit, minimal loss",
	"Q8_K_XL": "Unsloth dynamic 8-bit, minimal loss",

	// Full / half precision
	"F16":  "Half precision, full quality",
	"F32":  "Full precision, original quality",
	"BF16": "Brain float 16, full quality",
}

// Regex patterns for parsing GGUF filenames.
//
// quantPattern captures the full quantization type from a filename segment.
// Supports:
//   - IQ1..IQ4 importance-matrix quants with XXS/XS/S/M/NL suffixes
//   - Q2..Q8 with legacy _0/_1 suffixes, plain _K, or _K with
//     S/M/L/XL/XXL suffixes (XL/XXL are unsloth "Unsloth Dynamic" quants)
//   - F16/F32/BF16 float precisions
//
// Alternation order inside the _K suffix group puts longer literals first
// (XXL before XL before L) so the longest applicable suffix is always captured.
var (
	quantPattern = regexp.MustCompile(`(?i)(IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[2-8]_(?:[01]|K(?:_(?:XXL|XL|L|M|S))?)|F(?:16|32)|BF16)`)

	// Match parameter count: 7B, 13B, 70B, 1.5B, etc.
	paramPattern = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)[Bb]`)

	// Match model name from filename (before quant type).
	modelNamePattern = regexp.MustCompile(`^(.+?)[-._](?:IQ|Q|F|BF)\d`)
)

// isMMProjFile reports whether a GGUF filename is a multimodal projector
// (vision encoder) file. These files live alongside LLM quantizations in
// multimodal GGUF repos and must be downloaded as a companion to the chosen
// LLM quant — they are not user-selectable quantizations themselves.
func isMMProjFile(name string) bool {
	base := strings.ToLower(filepath.Base(name))
	return strings.HasPrefix(base, "mmproj") || strings.Contains(base, "-mmproj")
}

// analyzeGGUF analyzes GGUF files and extracts quantization information.
// Multimodal projector ("mmproj") files are detected and kept separate from
// LLM quantizations so the picker shows clean quant options while still
// knowing to bundle the vision encoder with any quant download.
func analyzeGGUF(files []FileInfo) *GGUFInfo {
	info := &GGUFInfo{}

	// Partition .gguf files into LLM quants vs mmproj vision encoders.
	var llmFiles []FileInfo
	for _, f := range files {
		if !strings.HasSuffix(strings.ToLower(f.Name), ".gguf") {
			continue
		}
		if isMMProjFile(f.Name) {
			info.MMProjFiles = append(info.MMProjFiles, f)
			continue
		}
		llmFiles = append(llmFiles, f)
	}

	if len(llmFiles) == 0 && len(info.MMProjFiles) == 0 {
		return nil
	}

	// Extract model name and parameter count from the first LLM file (or the
	// first mmproj file if the repo is mmproj-only, which is rare).
	var nameSource string
	if len(llmFiles) > 0 {
		nameSource = llmFiles[0].Name
	} else {
		nameSource = info.MMProjFiles[0].Name
	}
	if matches := modelNamePattern.FindStringSubmatch(nameSource); len(matches) > 1 {
		info.ModelName = strings.ReplaceAll(matches[1], "-", " ")
		info.ModelName = strings.ReplaceAll(info.ModelName, "_", " ")
	}
	if matches := paramPattern.FindStringSubmatch(nameSource); len(matches) > 1 {
		info.ParameterCount = matches[1] + "B"
	}

	// Parse each LLM GGUF file (mmproj files are intentionally excluded).
	for _, f := range llmFiles {
		quant := parseGGUFQuantization(f)
		if quant != nil {
			info.Quantizations = append(info.Quantizations, *quant)
		}
	}

	// Sort by quality (descending) then by size (ascending)
	sort.Slice(info.Quantizations, func(i, j int) bool {
		if info.Quantizations[i].Quality != info.Quantizations[j].Quality {
			return info.Quantizations[i].Quality > info.Quantizations[j].Quality
		}
		return info.Quantizations[i].File.Size < info.Quantizations[j].File.Size
	})

	return info
}

// parseGGUFQuantization extracts quantization info from a GGUF file.
func parseGGUFQuantization(f FileInfo) *GGUFQuantization {
	name := strings.ToUpper(filepath.Base(f.Name))

	// Find quantization type
	matches := quantPattern.FindStringSubmatch(name)
	if len(matches) < 2 {
		// No recognized quantization, might be a split file or unknown format
		ram := estimateRAM(f.Size)
		return &GGUFQuantization{
			Name:             "Unknown",
			File:             f,
			Quality:          3,
			QualityStars:     qualityToStars(3),
			EstimatedRAM:     ram,
			EstimatedRAMHuman: humanSize(ram),
			Description:      "Unknown quantization format",
		}
	}

	quantType := strings.ToUpper(matches[1])
	quality := quantQuality[quantType]
	if quality == 0 {
		quality = 3 // Default to medium if not found
	}

	desc := quantDescriptions[quantType]
	if desc == "" {
		desc = "Quantized model"
	}

	ram := estimateRAM(f.Size)
	return &GGUFQuantization{
		Name:             quantType,
		File:             f,
		Quality:          quality,
		QualityStars:     qualityToStars(quality),
		EstimatedRAM:     ram,
		EstimatedRAMHuman: humanSize(ram),
		Description:      desc,
	}
}

// qualityToStars converts a 1-5 quality rating to star representation.
func qualityToStars(quality int) string {
	filled := quality
	empty := 5 - quality
	return strings.Repeat("★", filled) + strings.Repeat("☆", empty)
}

// estimateRAM estimates RAM usage for a GGUF file.
// Formula: file_size * 1.1 + 500MB overhead
func estimateRAM(fileSize int64) int64 {
	const overhead = 500 * 1024 * 1024 // 500 MiB
	return int64(float64(fileSize)*1.1) + overhead
}

// RecommendGGUF recommends quantizations based on available RAM.
func RecommendGGUF(info *GGUFInfo, availableRAM int64) []GGUFQuantization {
	var recommended []GGUFQuantization

	for _, q := range info.Quantizations {
		if q.EstimatedRAM <= availableRAM {
			q.Recommended = true
			recommended = append(recommended, q)
		}
	}

	// Sort by quality (best that fits in RAM first)
	sort.Slice(recommended, func(i, j int) bool {
		if recommended[i].Quality != recommended[j].Quality {
			return recommended[i].Quality > recommended[j].Quality
		}
		return recommended[i].File.Size < recommended[j].File.Size
	})

	return recommended
}

// preferredMMProj picks the preferred multimodal projector file from a set.
// Preference order by precision: F16 > BF16 > F32 > first file. Returns the
// chosen FileInfo and a narrow filter string (lowercased basename without
// .gguf extension) that matches only that file under the downloader's
// case-insensitive substring filter semantics.
func preferredMMProj(files []FileInfo) (FileInfo, string) {
	precedence := []string{"-f16", "-bf16", "-f32"}
	for _, p := range precedence {
		for i := range files {
			name := strings.ToLower(filepath.Base(files[i].Name))
			if strings.Contains(name, p) {
				return files[i], strings.TrimSuffix(name, ".gguf")
			}
		}
	}
	// Fallback: pick the first file and build a filter from its full basename
	// so the filter still matches exactly one file.
	name := strings.ToLower(filepath.Base(files[0].Name))
	return files[0], strings.TrimSuffix(name, ".gguf")
}

// GGUFToSelectableItems converts GGUF quantizations to SelectableItems.
// This provides a unified interface for the web UI and CLI.
//
// When the repo contains mmproj vision-encoder files (multimodal models),
// an additional SelectableItem with Category="vision_encoder" is emitted
// for the preferred mmproj file. It is Recommended=true by default so that
// the recommended download command auto-bundles the vision encoder
// alongside any selected LLM quant.
func GGUFToSelectableItems(info *GGUFInfo) []SelectableItem {
	if info == nil {
		return nil
	}
	if len(info.Quantizations) == 0 && len(info.MMProjFiles) == 0 {
		return nil
	}

	items := make([]SelectableItem, 0, len(info.Quantizations)+1)

	// Track if we have a Q4_K_M (common recommended default)
	hasQ4KM := false
	for _, q := range info.Quantizations {
		if q.Name == "Q4_K_M" {
			hasQ4KM = true
			break
		}
	}

	for _, q := range info.Quantizations {
		// Determine if this should be recommended
		// Q4_K_M is a good default, otherwise highest quality in 4-bit range
		recommended := false
		if hasQ4KM && q.Name == "Q4_K_M" {
			recommended = true
		} else if !hasQ4KM && q.Quality >= 4 && q.File.Size < 10*1024*1024*1024 { // < 10 GiB
			recommended = true
		}

		item := SelectableItem{
			ID:           strings.ToLower(q.Name),
			Label:        q.Name,
			Description:  q.Description,
			Size:         q.File.Size,
			SizeHuman:    q.File.SizeHuman,
			Quality:      q.Quality,
			QualityStars: q.QualityStars,
			Recommended:  recommended,
			Category:     "quantization",
			FilterValue:  strings.ToLower(q.Name),
			Files:        []string{q.File.Path},
			RAM:          q.EstimatedRAM,
			RAMHuman:     q.EstimatedRAMHuman,
		}
		items = append(items, item)
	}

	// Append a vision-encoder companion item when mmproj files are present.
	// This is what makes multimodal GGUF downloads actually work end-to-end
	// (github issue #76) — the user picks a quant and the mmproj tags along
	// via the comma-separated -F filter list.
	if len(info.MMProjFiles) > 0 {
		chosen, filter := preferredMMProj(info.MMProjFiles)
		items = append(items, SelectableItem{
			ID:          "mmproj",
			Label:       filepath.Base(chosen.Name),
			Description: "Multimodal projector; required alongside the LLM quant for vision/multimodal models",
			Size:        chosen.Size,
			SizeHuman:   chosen.SizeHuman,
			Recommended: true,
			Category:    "vision_encoder",
			FilterValue: filter,
			Files:       []string{chosen.Path},
		})
	}

	return items
}
