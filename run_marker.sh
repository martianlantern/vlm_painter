#!/bin/bash
# This script is used to OCR pdfs into readable and llm ready markdown content for consumption
uv run --with marker-pdf marker_single paint_transformer.pdf --redo_inline_math --output_dir "papers" --output_format markdown
