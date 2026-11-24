#!/usr/bin/env bash
# Lightweight RAM watcher: appends one line per minute for the whole pipeline run.
# Started with nohup so it survives the session; harmless (no results/ files touched).
LOG="$(dirname "$0")/ram_watch.log"
while true; do
  printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(free -m | awk 'NR==2{printf "mem used=%sMB avail=%sMB", $3, $7} NR==3{printf " swap used=%sMB", $3}')" \
    >> "$LOG"
  sleep 60
done
