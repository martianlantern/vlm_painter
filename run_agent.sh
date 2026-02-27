#!/bin/bash

LOGFILE="agent_logs/agent_${COMMIT}.log"
claude --dangerously-skip-permissions \
       -p "$(cat PROMPT.md)" \
       --output-format text \
       --model claude-opus-4-6-thinking &> "$LOGFILE"