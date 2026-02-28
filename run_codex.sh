#!/bin/bash
UUID=$(cat /proc/sys/kernel/random/uuid)
LOGFILE="codex_logs_$(date +'%d_%b_%Y_%H_%M_%S')_${UUID}.log"
mkdir -p agent_logs
codex exec --model gpt-5.2 \
    --yolo --sandbox danger-full-access \
    --config model_reasoning_effort "high" \
    "$(cat PROMPT.md)" &> "agent_logs/$LOGFILE"