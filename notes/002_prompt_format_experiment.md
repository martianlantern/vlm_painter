---
title: prompt_format_experiment
description: Tested 5 different prompt formats for getting Qwen3.5-0.8B to generate brush stroke parameters. CSV with examples works best.
related notes: [[001_qwen3.5_vllm_server_setup.md]]
---

## Goal
Find the best prompt format for Qwen3.5-0.8B to output brush stroke parameters (x,y,w,h,rotation,R,G,B).

## Formats Tested
1. **csv_basic** - Detailed instructions with 2 examples, request 16 strokes
2. **csv_short** - Minimal instructions, 1 example
3. **json_strokes** - JSON array of objects
4. **tagged_csv** - XML-tagged prompt
5. **thinking_csv** - Description first, then STROKES: marker

## Results

| Format | Avg Valid Strokes | Max Valid | Runs w/ Valid (of 4) |
|--------|------------------|-----------|---------------------|
| csv_basic | 8.5 | 16 | 4/4 |
| csv_short | 9.75 | 37 | 3/4 |
| json_strokes | 6.25 | 19 | 2/4 |
| tagged_csv | 0.5 | 1 | 2/4 |
| thinking_csv | 1.25 | 2 | 4/4 |

## Key Observations
- **csv_basic is the winner**: reliable format, most consistently valid strokes
- Model tends to copy example values and increment monotonically (low diversity)
- Values often go out of range (x,y,w,h > 1.0) indicating model doesn't understand parameter semantics
- Format is correct (CSV, 8 values) ~50% of the time - sufficient baseline for RL
- JSON format causes values to exceed bounds more often
- Tagged/thinking formats cause model to output long comma sequences (degenerate)

## Decision
Use **csv_basic** format with these stroke params: `x,y,w,h,rotation,R,G,B` (all 0-1 range).
- 16 strokes per prediction
- Newline-separated CSV format
- Need SFT to teach correct value ranges and semantic meaning
- Then GRPO to teach actual painting

## Rendered Outputs
Saved in `results/prompt_experiment/` - most renderings show random blobs with no semantic connection to input images, confirming SFT is needed.
