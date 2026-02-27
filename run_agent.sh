#!/bin/bash

LOGFILE="agent_logs/agent_${COMMIT}.log"
claude --dangerously-skip-permissions \
       -p "$(cat PROMPT.md)" \
       --model claude-opus-X-Y &> "$LOGFILE"