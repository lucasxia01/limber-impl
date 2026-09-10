#!/usr/bin/env python3
"""Unit tests for scripts/artifact_fs.py (plan section 8 artifact-integrity row)."""
import os
import shutil
import tempfile
import unittest

from scripts import artifact_fs as af
from scripts import poseidon_common as pc

REQUIRED = ["a.json", "sub/b.txt"]


class ArtifactFsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.staging = os.path.join(self.tmp, "run-x.staging")
        os.makedirs(os.path.join(self.staging, "sub"))
        pc.write_canonical_json(os.path.join(self.staging, "a.json"), {"k": 1})
        pc.write_bytes(os.path.join(self.staging, "sub", "b.txt"), b"hello\n")

    def seal(self, root=None, failed=False):
        root = root or self.staging
        _, index_sha, entries = af.write_artifact_index(root)
        manifest = {"status": "failed" if failed else "complete",
                    "artifact_index_sha256": index_sha}
        return af.seal_manifest(root, manifest, failed=failed), entries

    def commit(self):
        final = os.path.join(self.tmp, "run-x")
        af.commit_staging(self.staging, final)
        return final

    def test_index_format_and_round_trip(self):
        digest, entries = self.seal()
        with open(os.path.join(self.staging, "artifacts.sha256"), "rb") as f:
            index = f.read()
        self.assertEqual(index, ("%s  8  a.json\n%s  6  sub/b.txt\n" % (
            pc.sha256_hex(b'{"k":1}\n'), pc.sha256_hex(b"hello\n"))).encode())
        self.assertEqual([e[0] for e in entries], REQUIRED)
        final = self.commit()
        self.assertFalse(os.path.exists(self.staging))
        manifest, sha, status = af.verify_artifact_dir(final, REQUIRED, [])
        self.assertEqual((status, sha), ("complete", digest))
        self.assertEqual(manifest["artifact_index_sha256"],
                         pc.sha256_file(os.path.join(final, "artifacts.sha256")))

    def test_modified_file_rejected(self):
        self.seal()
        final = self.commit()
        with open(os.path.join(final, "sub", "b.txt"), "ab") as f:
            f.write(b"!")
        with self.assertRaises(pc.RunnerError) as cm:
            af.verify_artifact_dir(final, REQUIRED, [])
        self.assertEqual(cm.exception.error_code, "IntegrityMismatch")

    def test_stale_detached_digest_rejected(self):
        self.seal()
        final = self.commit()
        pc.write_detached_digest(os.path.join(final, "manifest.sha256"), "0" * 64)
        with self.assertRaises(pc.RunnerError) as cm:
            af.verify_artifact_dir(final, REQUIRED, [])
        self.assertEqual(cm.exception.error_code, "IntegrityMismatch")

    def test_symlink_rejected(self):
        self.seal()
        final = self.commit()
        os.symlink("a.json", os.path.join(final, "link.json"))
        with self.assertRaises(pc.RunnerError) as cm:
            af.verify_artifact_dir(final, REQUIRED, ["link.json"])
        self.assertEqual(cm.exception.error_code, "SymlinkRejected")

    def test_unsafe_paths_rejected(self):
        for bad in ("a/../b", "./a", "a//b", "/abs", "a/", "sp ace", "café", ""):
            with self.assertRaises(pc.RunnerError) as cm:
                af.check_payload_path(bad)
            self.assertEqual(cm.exception.error_code, "UnsafePath")
        pc.write_bytes(os.path.join(self.staging, "bad name.txt"), b"x")
        with self.assertRaises(pc.RunnerError) as cm:
            af.write_artifact_index(self.staging)
        self.assertEqual(cm.exception.error_code, "UnsafePath")

    def test_wrong_file_set_rejected(self):
        self.seal()
        final = self.commit()
        with self.assertRaises(pc.RunnerError) as cm:
            af.verify_artifact_dir(final, REQUIRED + ["missing.json"], [])
        self.assertEqual(cm.exception.error_code, "FileSetMismatch")
        with self.assertRaises(pc.RunnerError) as cm:
            af.verify_artifact_dir(final, ["a.json"], [])  # sub/b.txt is then unexpected
        self.assertEqual(cm.exception.error_code, "FileSetMismatch")
        pc.write_bytes(os.path.join(final, "extra.txt"), b"x")  # unindexed payload
        with self.assertRaises(pc.RunnerError) as cm:
            af.verify_artifact_dir(final, REQUIRED, ["extra.txt"])
        self.assertEqual(cm.exception.error_code, "FileSetMismatch")

    def test_mixed_manifest_pairs_rejected(self):
        self.seal()
        final = self.commit()
        pc.write_bytes(os.path.join(final, "manifest.failed.json"), b"{}\n")
        pc.write_detached_digest(os.path.join(final, "manifest.failed.sha256"),
                                 pc.sha256_hex(b"{}\n"))
        with self.assertRaises(pc.RunnerError) as cm:
            af.verify_artifact_dir(final, REQUIRED, [])
        self.assertEqual(cm.exception.error_code, "ManifestInvalid")

    def test_staging_name_rejected(self):
        self.seal()
        with self.assertRaises(pc.RunnerError) as cm:
            af.verify_artifact_dir(self.staging, REQUIRED, [])
        self.assertEqual(cm.exception.error_code, "ManifestInvalid")

    def test_noreplace_rename_refuses_existing_destination(self):
        final = os.path.join(self.tmp, "run-x")
        os.mkdir(final)
        with self.assertRaises(pc.RunnerError) as cm:
            af.rename_dir_noreplace(self.staging, final)
        self.assertEqual(cm.exception.error_code, "RenameRefused")
        self.assertTrue(os.path.isdir(self.staging))
        self.assertEqual(os.listdir(final), [])
        # And through the full commit path: staging is retained.
        self.seal()
        with self.assertRaises(pc.RunnerError) as cm:
            af.commit_staging(self.staging, final)
        self.assertEqual(cm.exception.error_code, "RenameRefused")
        self.assertTrue(os.path.isfile(os.path.join(self.staging, "manifest.json")))

    def test_failed_pair(self):
        digest, _ = self.seal(failed=True)
        final = self.commit()
        manifest, sha, status = af.verify_artifact_dir(final, REQUIRED, [])
        self.assertEqual((status, sha, manifest["status"]), ("failed", digest, "failed"))

    def test_seal_manifest_requires_matching_index(self):
        af.write_artifact_index(self.staging)
        with self.assertRaises(pc.RunnerError):
            af.seal_manifest(self.staging, {"status": "complete",
                                            "artifact_index_sha256": "0" * 64})

    def test_index_parse_rejects_unsorted(self):
        line = "%s  1  %s\n"
        data = (line % ("a" * 64, "b.txt") + line % ("a" * 64, "a.txt")).encode()
        with self.assertRaises(pc.RunnerError):
            af.parse_artifact_index(data)
        with self.assertRaises(pc.RunnerError):
            af.parse_artifact_index(("%s 1  a.txt\n" % ("a" * 64)).encode())

    def test_create_staging_exclusive(self):
        with self.assertRaises(pc.RunnerError) as cm:
            af.create_staging_exclusive(self.staging)
        self.assertEqual(cm.exception.error_code, "StagingExists")

    def test_fsync_tree_rejects_symlink(self):
        os.symlink("a.json", os.path.join(self.staging, "l"))
        with self.assertRaises(pc.RunnerError):
            af.fsync_tree_bottom_up(self.staging)

    def test_verify_indexed_dir_and_reuse_existing(self):
        digest, _ = self.seal()
        final = self.commit()
        manifest, sha, status = af.verify_indexed_dir(final)
        self.assertEqual((sha, status), (digest, "complete"))
        manifest, sha, status = af.verify_and_reuse_existing(final, REQUIRED, [], digest)
        self.assertEqual(sha, digest)
        with self.assertRaises(pc.RunnerError) as cm:
            af.verify_and_reuse_existing(final, REQUIRED, [], "0" * 64)
        self.assertEqual(cm.exception.error_code, "IntegrityMismatch")
        with open(os.path.join(final, "a.json"), "ab") as f:
            f.write(b"!")
        with self.assertRaises(pc.RunnerError):
            af.verify_and_reuse_existing(final, REQUIRED, [], digest)


if __name__ == "__main__":
    unittest.main()
