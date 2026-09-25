#!/usr/bin/env bash
# CPU-only tests using the pinned image's exact dependencies. No inference load.
set -euo pipefail
umask 077
DSV41_TEST_IMAGE=${DSV41_TEST_IMAGE:-ambientlight/dsv41-sm120:sero-45f538a-baseline}
DSV41_OVERLAY_PATH=${DSV41_OVERLAY_PATH:-/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay}
test -d "$DSV41_OVERLAY_PATH/test/registered/unit"
DSV41_TEST_EVIDENCE=$(mktemp -d --tmpdir=/mnt/hot/dsv41_state api-tests-XXXXXXXX)
echo "Test evidence: $DSV41_TEST_EVIDENCE"
docker image inspect --format '{{.Id}}' "$DSV41_TEST_IMAGE" > "$DSV41_TEST_EVIDENCE/image-id.txt"
git -C "$DSV41_OVERLAY_PATH" rev-parse HEAD > "$DSV41_TEST_EVIDENCE/source-head.txt"
git -C "$DSV41_OVERLAY_PATH" diff --binary > "$DSV41_TEST_EVIDENCE/source.patch"
# Unit fixtures explicitly opt into recovery/signatures. Do not inherit a
# production image's external secret path or opt-ins; never mount the live key.
docker run --rm --runtime runc \
  --env NVIDIA_VISIBLE_DEVICES=void --env PYTHONDONTWRITEBYTECODE=1 \
  --env SGLANG_ANTHROPIC_THINKING_KEY_FILE= \
  --env SGLANG_ANTHROPIC_TOOL_RECOVERY=0 --env SGLANG_RESPONSES_TOOL_RECOVERY=0 \
  --env PYTHONPATH=/overlay/python:/opt/dsv41/adapter \
  --mount "type=bind,src=$DSV41_OVERLAY_PATH,dst=/overlay,readonly" \
  --mount "type=bind,src=$DSV41_TEST_EVIDENCE,dst=/evidence" \
  --workdir /overlay --entrypoint python3 "$DSV41_TEST_IMAGE" \
  -m pytest -q -p no:cacheprovider --tb=short --junitxml=/evidence/results.xml \
  test/registered/unit/function_call \
  test/registered/unit/parser/test_reasoning_parser.py \
  test/registered/unit/entrypoints/openai \
  test/registered/unit/entrypoints/anthropic \
  test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py \
  test/registered/unit/managers/test_scheduler_chunked_abort_race.py \
  "$@" 2>&1 | tee "$DSV41_TEST_EVIDENCE/pytest.log"
