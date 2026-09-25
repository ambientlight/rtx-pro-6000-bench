#!/usr/bin/env bash
# Continuously retain exact HTTP traffic and SGLang native inference logs for dsv4.

set -euo pipefail
umask 077

dump_root=${DSV4_DUMP_ROOT:-/mnt/hot/dsv4_dumps}
container=${DSV4_DUMP_CONTAINER:-dsv4}
port=${DSV4_DUMP_PORT:-8000}
base_url=${DSV4_DUMP_BASE_URL:-http://127.0.0.1:8000}
rotate_bytes=${DSV4_DUMP_ROTATE_BYTES:-2147483648}
socket_buffer_bytes=${DSV4_DUMP_SOCKET_BUFFER_BYTES:-4194304}
capture_batch_bytes=${DSV4_DUMP_CAPTURE_BATCH_BYTES:-1048576}
duration_seconds=${DSV4_DUMP_DURATION_SECONDS:-315360000}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
wire_script=${script_dir}/dsv4_wire_capture.py
capture_args=(--interface "${DSV4_DUMP_INTERFACE:-eth0}")
if test -n "${DSV4_DUMP_API_KEY_FILE:-}"; then
  capture_args+=(--api-key-file "${DSV4_DUMP_API_KEY_FILE}")
fi

mkdir -p -- "${dump_root}"
chmod 0700 "${dump_root}"

# Avoid creating a stream of empty sessions while Docker or the model container
# is unavailable. systemd will keep this wrapper alive until dsv4 returns.
until test "$(docker inspect --format '{{.State.Running}}' "${container}" 2>/dev/null || true)" = true; do
  sleep 5
done

started_stamp=$(date -u +%Y%m%dT%H%M%SZ)
started_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
session_dir=${dump_root}/capture-${started_stamp}-$$
mkdir -m 0700 -- "${session_dir}"
ln -sfn -- "${session_dir}" "${dump_root}/current"

native_log_pid=0
wire_pid=0

stop_children() {
  trap - TERM INT EXIT
  if test "${wire_pid}" -gt 0 && kill -0 "${wire_pid}" 2>/dev/null; then
    kill -TERM "${wire_pid}" 2>/dev/null || true
  fi
  if test "${native_log_pid}" -gt 0 && kill -0 "${native_log_pid}" 2>/dev/null; then
    kill -TERM "${native_log_pid}" 2>/dev/null || true
  fi
  if test "${wire_pid}" -gt 0; then
    wait "${wire_pid}" 2>/dev/null || true
  fi
  if test "${native_log_pid}" -gt 0; then
    wait "${native_log_pid}" 2>/dev/null || true
  fi
}

trap stop_children TERM INT EXIT

# Native level-3 logging contains request.received/request.finished records with
# rendered inputs, raw generated output, token IDs, and scheduler metadata. The
# wire recorder independently retains the exact HTTP request and response bytes.
docker logs --follow --timestamps --since "${started_iso}" "${container}" \
  >"${session_dir}/dsv4-native.log" 2>&1 &
native_log_pid=$!

python3 "${wire_script}" \
  "${capture_args[@]}" \
  --output-dir "${session_dir}/wire" \
  --duration-seconds "${duration_seconds}" \
  --container "${container}" \
  --port "${port}" \
  --base-url "${base_url}" \
  --rotate-bytes "${rotate_bytes}" \
  --socket-buffer-bytes "${socket_buffer_bytes}" \
  --capture-batch-bytes "${capture_batch_bytes}" &
wire_pid=$!

wire_status=0
wait "${wire_pid}" || wire_status=$?
wire_pid=0

stop_children
exit "${wire_status}"
