"""Read-only configuration checks and stubbed launch-all routing tests."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProductionProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compose = json.loads(
            subprocess.check_output(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(ROOT / "docker/deepseek-v41/compose.api.yaml"),
                    "config",
                    "--format",
                    "json",
                ],
                text=True,
            )
        )

    def test_explicit_legacy_contract_and_restart_policy(self):
        service = self.compose["services"]["deepseek"]
        self.assertEqual(service["container_name"], "dsv41")
        self.assertEqual(service["restart"], "unless-stopped")
        env = service["environment"]
        for name, value in {
            "SERVER_PORT": "8000",
            "SERVED_MODEL_NAME": "deepseek-v4-flash",
            "REQUIRE_API_KEY": "0",
            "CONTEXT_LENGTH": "524288",
            "CHUNKED_PREFILL_SIZE": "2048",
            "PREFILL_DECODE_INTERVAL": "4",
            "MAX_RUNNING_REQUESTS": "8",
            "MEMORY_FRACTION": "0.87",
            "OFFLOAD_MODE": "ram",
            "DSV41_CACHE_GIB": "0",
            "DSV41_HICACHE_HOST_BUDGET_BYTES": "256000000000",
            "DSV41_HICACHE_SWA_FRACTION": "0.67",
        }.items():
            self.assertEqual(env[name], value)
        self.assertEqual(len(service["ports"]), 1)
        self.assertEqual(service["ports"][0]["target"], 8000)
        self.assertEqual(str(service["ports"][0]["published"]), "8000")

    def test_ram_offload_can_lock_memory_without_changing_gpu_profile(self):
        service = self.compose["services"]["deepseek"]
        self.assertIn("IPC_LOCK", service["cap_add"])
        self.assertEqual(service["ulimits"]["memlock"], {"soft": -1, "hard": -1})
        self.assertEqual(service["ipc"], "host")
        self.assertEqual(service["image"], "ambientlight/dsv41-sm120:cache-attribution-v1")
        self.assertEqual(service["environment"]["MAX_TOTAL_TOKENS"], "")

    def test_version_and_half_native_context_match_release_lock(self):
        service = self.compose["services"]["deepseek"]
        release = json.loads((ROOT / "docker/deepseek-v41/baseline.lock.json").read_text())["production_profile"]
        self.assertEqual(service["labels"]["dev.idle.deployment.version"], "dsv41-v2-log-forwarding-v1")
        self.assertEqual(service["environment"]["SGLANG_OPT_HICACHE_SWA_RETAIN_REQUEST_BOUNDARIES"], "1")
        self.assertEqual(service["environment"]["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"], "python")
        self.assertEqual(service["image"], release["image_tag"])
        self.assertEqual(service["labels"]["dev.idle.deployment.version"], release["deployment_version"])
        self.assertEqual(int(service["environment"]["CONTEXT_LENGTH"]), 1048576 // 2)
        self.assertEqual(release["context_length"], 1048576 // 2)
        self.assertEqual(release["mem_fraction_static"], 0.87)
        self.assertEqual(float(service["environment"]["MEMORY_FRACTION"]), release["mem_fraction_static"])
        self.assertEqual(release["offload_mode"], "ram")
        self.assertEqual(release["engram_cache_gib"], 0)
        self.assertEqual(release["hicache_swa_payload_fraction"], 0.67)

    def test_no_swap_policy_is_host_persistent_without_a_new_ram_cap(self):
        service = self.compose["services"]["deepseek"]
        release = json.loads((ROOT / "docker/deepseek-v41/baseline.lock.json").read_text())["production_profile"]
        self.assertEqual(service["cgroup_parent"], "dsv41.slice")
        self.assertEqual(release["cgroup_parent"], service["cgroup_parent"])
        self.assertEqual(release["effective_swap_limit_bytes"], 0)
        self.assertNotIn("mem_limit", service)
        self.assertNotIn("memswap_limit", service)
        unit = (ROOT / "systemd/dsv41.slice").read_text()
        self.assertIn("MemorySwapMax=0\n", unit)
        self.assertIn("WantedBy=slices.target\n", unit)
        self.assertNotIn("MemoryMax=", unit)

    def test_full_overlay_build_includes_host_cache_diagnostics(self):
        from dsv41_build_api_image import API_FILES
        name = "mem_cache/unified_cache/host_cache_diagnostics.py"
        self.assertIn(name, API_FILES)
        dockerfile = (ROOT / "docker/deepseek-v41/Dockerfile.api").read_text()
        self.assertIn("COPY python/sglang/srt/" + name + " ", dockerfile)

    def test_prefill_decode_interval_has_matching_boot_override(self):
        service = self.compose["services"]["deepseek"]
        release = json.loads((ROOT / "docker/deepseek-v41/baseline.lock.json").read_text())["production_profile"]
        self.assertEqual(int(service["environment"]["PREFILL_DECODE_INTERVAL"]), release["prefill_decode_interval"])
        mounts = [m for m in service["volumes"] if m["target"] == "/opt/dsv41/boot.py"]
        self.assertEqual(len(mounts), 1)
        self.assertEqual(mounts[0]["type"], "bind")
        self.assertEqual(mounts[0]["source"], release["boot_source"])
        self.assertTrue(mounts[0]["read_only"])
        self.assertFalse(mounts[0].get("bind", {}).get("create_host_path", False))
        self.assertTrue(Path(mounts[0]["source"]).is_file())

    def test_only_one_model_and_persistent_state(self):
        self.assertEqual(set(self.compose["services"]), {"deepseek"})
        mounts = {v["target"]: v["source"] for v in self.compose["services"]["deepseek"]["volumes"]}
        self.assertEqual(mounts["/state"], "/mnt/hot/dsv41_state/production")
        self.assertEqual(mounts["/run/dsv41/anthropic-thinking.key"], "/mnt/hot/dsv4_state/anthropic-thinking.key")
        self.assertEqual(mounts["/models/DeepSeek-V4.1-Flash"], "/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash")

    def test_host_cache_overrides_are_readonly_and_complete(self):
        service = self.compose['services']['deepseek']
        target = '/sgl-workspace/sglang/python/sglang/srt/mem_cache/hybrid_cache/'
        for filename in ('hybrid_pool_assembler.py', 'dsv41_host_budget.py'):
            mount, = [v for v in service['volumes'] if v['target'] == target + filename]
            self.assertTrue(mount['read_only'])
            self.assertFalse(mount.get('bind', {}).get('create_host_path', False))
            self.assertTrue(Path(mount['source']).is_file())

    def test_expandable_segments_are_disabled_and_match_release_lock(self):
        env = self.compose["services"]["deepseek"]["environment"]
        release = json.loads((ROOT / "docker/deepseek-v41/baseline.lock.json").read_text())["production_profile"]
        self.assertEqual(env["PYTORCH_ALLOC_CONF"], "expandable_segments:False")
        self.assertEqual(env["PYTORCH_ALLOC_CONF"], release["pytorch_alloc_conf"])
        self.assertEqual(env["MAX_TOTAL_TOKENS"], "")
        self.assertIsNone(release["max_total_tokens"])
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)
        self.assertNotIn("CUDA_EXPANDABLE_MEMORY_SEGMENTS", env)

    def test_capture_follows_production_endpoint_without_auth(self):
        unit = (ROOT / "systemd/dsv41-wire-dump.service").read_text()
        self.assertIn("Environment=DSV4_DUMP_CONTAINER=dsv41\n", unit)
        self.assertIn("Environment=DSV4_DUMP_PORT=8000\n", unit)
        self.assertIn("Environment=DSV4_DUMP_BASE_URL=http://127.0.0.1:8000\n", unit)
        self.assertNotIn("API_KEY_FILE", unit)

    def test_crash_diagnostics_are_persistent_and_bounded(self):
        service = self.compose["services"]["deepseek"]
        env = service["environment"]
        self.assertIn("SYS_PTRACE", service["cap_add"])
        self.assertNotIn("privileged", service)
        # Compose's JSON serializer omits zero-valued hard/soft fields.
        self.assertEqual(service["ulimits"]["core"].get("soft", 0), 0)
        self.assertEqual(service["ulimits"]["core"].get("hard", 0), 0)
        self.assertEqual(service["entrypoint"], ["python3", "-u", "/opt/dsv41/diagnostic_entrypoint.py"])
        self.assertEqual(service["command"], ["run"])
        self.assertEqual(env["DSV41_DIAGNOSTIC_ROOT"], "/mnt/hot/dsv41_dumps/diagnostics")
        for name in ("CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "CUDA_ENABLE_USER_TRIGGERED_COREDUMP",
                     "TORCH_NCCL_DUMP_ON_TIMEOUT", "TORCH_NCCL_ENABLE_MONITORING",
                     "TORCH_NCCL_TRACE_CPP_STACK", "TORCH_NCCL_EXTRA_DUMP_ON_EXEC"):
            self.assertEqual(env[name], "1")
        self.assertEqual(env["CUDA_COREDUMP_GENERATION_FLAGS"], "skip_global_memory,skip_abort")
        self.assertEqual(env["TORCH_NCCL_TRACE_BUFFER_SIZE"], "8192")
        self.assertEqual(env["TORCH_NCCL_ENABLE_TIMING"], "0")
        for key, value in {
            "SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS": "60",
            "SGLANG_PYSPY_DUMP_TIMEOUT_SECS": "5",
            "SGLANG_PYSPY_DUMP_TOTAL_TIMEOUT_SECS": "20",
            "SGLANG_PYSPY_DUMP_NATIVE": "1",
            "SGLANG_NCCL_DUMP_BEFORE_CRASH": "1",
            "SGLANG_NCCL_DUMP_BEFORE_CRASH_WAIT_SECS": "10",
            "SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS": "60",
        }.items():
            self.assertEqual(env[key], value)
        self.assertEqual(env["NCCL_DEBUG"], "INFO")
        self.assertNotIn("CUDA_LAUNCH_BLOCKING", env)
        wrapper = [m for m in service["volumes"] if m["target"] == "/opt/dsv41/diagnostic_entrypoint.py"]
        self.assertEqual(len(wrapper), 1)
        self.assertTrue(wrapper[0]["read_only"])

    def run_launcher(self, *args, canary_running=False):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            docker = folder / "docker"
            docker.write_text("""#!/bin/bash
printf '%s\\n' "$*" >> "$DSV41_TEST_TRACE"
if [ "$1" = inspect ]; then
  case "${@: -1}" in
    dsv41) echo true ;;
    dsv41-api) echo "$DSV41_TEST_CANARY" ;;
    *) echo false ;;
  esac
elif [ "$1" = compose ] && [[ "$*" = *"up -d"* ]]; then
  echo started-fixture
elif [ "$1" = ps ]; then
  echo 'NAMES STATUS PORTS'
  echo 'dsv41 healthy 8000'
fi
""")
            docker.chmod(0o700)
            for name in ("curl", "nvidia-smi"):
                script = folder / name
                script.write_text("#!/bin/bash\nexit 0\n")
                script.chmod(0o700)
            trace = folder / "trace"
            env = {
                **os.environ,
                "PATH": str(folder) + ":" + os.environ["PATH"],
                "DSV41_TEST_TRACE": str(trace),
                "DSV41_TEST_CANARY": str(canary_running).lower(),
            }
            process = subprocess.run(
                ["bash", str(ROOT / "launch_all.sh"), *args], env=env, capture_output=True, text=True, timeout=10
            )
            return process, trace.read_text() if trace.exists() else ""

    def test_normal_start_does_not_revive_embeddings_or_v12(self):
        process, trace = self.run_launcher()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertNotIn("compose", trace)
        self.assertNotIn("qwen3-embed", trace)
        self.assertNotIn("rm ", trace)

    def test_embedding_target_is_disabled(self):
        process, trace = self.run_launcher("qwen")
        self.assertEqual(process.returncode, 1)
        self.assertIn("disabled", process.stdout)
        self.assertEqual(trace, "")

    def test_legacy_target_launches_new_compose_not_v12(self):
        process, trace = self.run_launcher("dsv4")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("up -d --no-build --force-recreate deepseek", trace)
        self.assertNotIn("run -d", trace)
        self.assertNotIn("rm ", trace)

    def test_running_canary_blocks_second_gpu_model(self):
        process, trace = self.run_launcher("dsv41", canary_running=True)
        self.assertEqual(process.returncode, 1)
        self.assertIn("still running", process.stdout)
        self.assertNotIn("compose", trace)


if __name__ == "__main__":
    unittest.main()
