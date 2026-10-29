"""Protocol invariants against the actual committed Group 26 manifest."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from tts_protocol import read_protocol, normalise_text

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_committed_protocol(self):
        groups, info = read_protocol(ROOT / "splits/split_manifest.csv",
                                     ROOT / "splits/split_config.json", ".", False)
        self.assertEqual(info["counts"], {"train":2400, "validation":300, "test":300})
        self.assertEqual(info["manifest_content_sha256"], "64a2db28bb18b5aedcc0be012d80d0e0569a774dad554b2e6238068574dcb885")
        self.assertEqual(len({r["utterance_id"] for rows in groups.values() for r in rows}), 3000)

    def reject_mutation(self, mutate):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with (ROOT / "splits/split_manifest.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                names = reader.fieldnames
                rows = list(reader)
            mutate(rows)
            path = folder / "manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=names)
                writer.writeheader()
                writer.writerows(rows)
            config = folder / "split_config.json"
            config.write_bytes((ROOT / "splits/split_config.json").read_bytes())
            with self.assertRaises(ValueError):
                read_protocol(path, config, ".", False)

    def test_duplicate_ids_rejected(self):
        self.reject_mutation(lambda rows: rows[1].update(utterance_id=rows[0]["utterance_id"]))

    def test_split_drift_rejected(self):
        self.reject_mutation(lambda rows: rows[0].update(split="test"))

    def test_text_drift_rejected(self):
        self.reject_mutation(lambda rows: rows[0].update(clean_text="different text"))

    def test_path_escape_rejected(self):
        self.reject_mutation(lambda rows: rows[0].update(audio_relative_path="../outside.wav"))

    def test_missing_audio_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                read_protocol(ROOT / "splits/split_manifest.csv", ROOT / "splits/split_config.json", folder, True)

    def test_line_endings_keep_same_content_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            content = (ROOT / "splits/split_manifest.csv").read_text(encoding="utf-8")
            hashes = []
            for newline in ("\n", "\r\n"):
                path.write_bytes(content.replace("\n", newline).encode("utf-8"))
                _, info = read_protocol(path, ROOT / "splits/split_config.json", ".", False)
                hashes.append(info["manifest_content_sha256"])
            self.assertEqual(hashes[0], hashes[1])

    def test_normalization(self):
        self.assertEqual(normalise_text("  ‘Müller’—à  "), "'Muller'-a")
        value = normalise_text("â ü è\t\n words")
        self.assertEqual(normalise_text(value), value)


if __name__ == "__main__":
    unittest.main()
