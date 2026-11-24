"""Narrow, checked policy edits on the installed SGLang runtime at image build."""
from pathlib import Path


def replace(path, old, new, count=1):
    text = path.read_text()
    if text.count(old) != count:
        raise RuntimeError(f"Unrecognized runtime source: {path.name}")
    path.write_text(text.replace(old, new))


root = Path('/sgl-workspace/sglang/python/sglang/srt')
if root.exists():
    # Unknown URLs and custom HTTP verbs are user input, not metric labels.
    replace(root / 'utils/common.py',
            '        path, is_handled_path = _get_fastapi_request_path(request)\n'
            '        method = request.method\n',
            '        path, is_handled_path = _get_fastapi_request_path(request)\n'
            '        if not is_handled_path:\n'
            '            path = "unmatched"\n'
            '        method = request.method if request.method in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"} else "OTHER"\n')
    replace(root / 'entrypoints/http_server.py',
            'app.router.route_class = ORJSONRoute',
            'from content_free_logging import PrivacyAdminMiddleware\n'
            'app.add_middleware(PrivacyAdminMiddleware)\n'
            'app.router.route_class = ORJSONRoute')
    replace(root / 'utils/cudacore_pyspy_dump_utils.py',
            'def _run_pyspy(cmd: List[str], timeout: float, deadline: float) -> None:\n',
            'def _run_pyspy(cmd: List[str], timeout: float, deadline: float) -> None:\n'
            '    if os.environ.get("CONTENT_FREE_LOGGING") == "1":\n'
            '        from content_free_logging import collect_pyspy\n'
            '        return collect_pyspy(cmd, timeout, deadline)\n')
    # Request logging remains off even if an internal caller changes settings.
    logger = root / 'utils/request_logger.py'
    replace(logger, 'import logging\n', 'import logging\nimport os\n')
    replace(logger, '        self.log_requests = log_requests\n',
            '        self.log_requests = False if os.environ.get("CONTENT_FREE_LOGGING") == "1" else log_requests\n', 2)
    replace(logger, '    def _setup_targets(self) -> List[logging.Logger]:\n',
            '    def _setup_targets(self) -> List[logging.Logger]:\n'
            '        if os.environ.get("CONTENT_FREE_LOGGING") == "1":\n'
            '            return []\n')
