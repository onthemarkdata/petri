#!/bin/bash
# Scratch: clean restart of Petri on pi into a PERSISTENT, inspectable dish.
# Default configs (no convergence tuning) + WebSearch/WebFetch corpus tools + Sonnet.
set -u
cd /home/user/petri
SP=/tmp/claude-0/-home-user-petri/767427e9-1a07-5e63-aad4-1e3d3de99d01/scratchpad
# Clear any prior colonies for a truly fresh, append-only-safe start.
rm -rf /home/user/petri/.petri
export PI_BIN=/opt/node22/bin/pi
export PI_PROVIDER=anthropic
export PI_MODEL=anthropic/claude-sonnet-4-6
export PI_TOOLS=WebSearch,WebFetch
export PI_MAX_CALLS=80
export PI_CONCURRENCY=1
uv run python "$SP/pi_e2e_run.py" \
    --claim "Regular expressions cannot parse arbitrary HTML." \
    --workdir /home/user/petri \
    > "$SP/e2e_persistent.log" 2>&1
echo "persistent-run exit=$?"
echo "dish at: /home/user/petri/.petri"
