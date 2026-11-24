"""Start the existing LiteLLM CLI with private, content-free diagnostics."""
import datetime
import os
from pathlib import Path
import resource
import sys
import tempfile

from content_free_logging import discard_stdio, emit, install_or_exit

os.umask(0o077)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
root = Path("/var/log/litellm/diagnostics")
root.mkdir(parents=True, exist_ok=True, mode=0o700)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
os.environ["CONTENT_FREE_LOG_DIR"] = tempfile.mkdtemp(prefix=f"run-{stamp}-", dir=root)
os.environ["CONTENT_FREE_LOGGING"] = "1"
os.environ["LITELLM_MESSAGES_DUMP_DIR"] = ""
discard_stdio()
install_or_exit()
emit({"event": "proxy_starting"})
os.execv(sys.executable, [sys.executable, "-m", "litellm.proxy.proxy_cli", *sys.argv[1:]])
