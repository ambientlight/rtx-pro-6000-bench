"""Offline release-capture safety checks; never access the live container."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dsv41_snapshot_release as release


class SnapshotSafetyTest(unittest.TestCase):
    def test_deployment_documents_exist_and_keep_archive_names(self):
        self.assertEqual(
            {Path(relative).name for relative in release.DEPLOYMENT_FILES},
            {"baseline.lock.json", "NO-SWAP.md"},
        )
        for relative in release.DEPLOYMENT_FILES:
            with self.subTest(relative=relative):
                self.assertTrue((release.REPO / "docker/deepseek-v41" / relative).is_file())

    def test_exclusive_metadata_write_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            release.save(path, {"original": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                release.save(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), before)

    def test_exclusive_copy_and_streaming_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "private/key"
            source.write_bytes(b"synthetic fixture\x00\xff")
            release.private_copy(source, target)
            self.assertEqual(release.digest(target), hashlib.sha256(source.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                release.private_copy(source, target)

    def test_public_directory_rejected_before_docker(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(release, "run") as run:
            path = Path(tmp)
            path.chmod(0o755)
            with self.assertRaises(ValueError):
                release.snapshot(path, "dsv41")
            run.assert_not_called()

    def test_symlink_directory_rejected_before_docker(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(release, "run") as run:
            link = Path(tmp) / "link"
            link.symlink_to(tmp, target_is_directory=True)
            with self.assertRaises(ValueError):
                release.snapshot(link, "dsv41")
            run.assert_not_called()

    def test_existing_manifest_rejected_before_docker(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(release, "run") as run:
            path = Path(tmp)
            (path / "manifest.json").write_text("{}")
            with self.assertRaises(ValueError):
                release.snapshot(path, "dsv41")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
