"""Transport the synthetic-proof auth key to SGLang through private config."""
import json
import os
from pathlib import Path
import tempfile


def write_auth_config(state: Path, secret: str) -> str:
    target = state / "server-auth.yaml"
    fd, path = tempfile.mkstemp(prefix=".auth-", dir=state)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"api-key": secret}, handle)
            handle.write("\n")
        os.replace(path, target)
    finally:
        Path(path).unlink(missing_ok=True)
    return str(target)
