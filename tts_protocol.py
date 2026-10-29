"""Shared protocol checks. Text processing is copied from A's notebook."""
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata
from collections import Counter


def normalise_text(text):
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "à": "a", "â": "a", "è": "e", "ü": "u",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_protocol(manifest, split_config, data_root, check_files=True):
    """Read the committed manifest; never create or rewrite a split."""
    manifest, split_config, data_root = map(Path, (manifest, split_config, data_root))
    config = json.loads(split_config.read_text(encoding="utf-8-sig"))
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"utterance_id", "split", "audio_relative_path", "text", "clean_text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs columns {sorted(required)}")
        rows = list(reader)
    groups = {name: [] for name in ("train", "validation", "test")}
    seen = set()
    for row in rows:
        uid = row["utterance_id"]
        if not uid or uid in seen:
            raise ValueError(f"Empty or duplicate utterance ID: {uid!r}")
        seen.add(uid)
        if row["split"] not in groups:
            raise ValueError(f"Unknown split for {uid}: {row['split']}")
        relative = PurePosixPath(row["audio_relative_path"])
        if (relative.is_absolute() or relative.parts != ("wavs", f"{uid}.wav")
                or "\\" in row["audio_relative_path"] or ":" in row["audio_relative_path"]):
            raise ValueError(f"Unexpected audio path for {uid}: {relative}")
        if not row["clean_text"] or normalise_text(row["text"]) != row["clean_text"]:
            raise ValueError(f"clean_text differs from A's normalization for {uid}")
        item = dict(row)
        item["audio_path"] = data_root.joinpath(*relative.parts)
        if check_files and not item["audio_path"].is_file():
            raise FileNotFoundError(item["audio_path"])
        groups[row["split"]].append(item)
    counts = {name: len(items) for name, items in groups.items()}
    if counts != config["counts"] or sum(counts.values()) != config["subset_size"]:
        raise ValueError(f"Counts {counts} differ from split_config.json")
    if any(count == 0 for count in counts.values()):
        raise ValueError("An empty split was found")
    return groups, {
        "manifest_sha256": sha256_file(manifest),
        # Canonical row content is stable across Git's LF/CRLF conversion.
        # Row order is retained because it also affects seeded sampling.
        "manifest_content_sha256": hashlib.sha256(json.dumps(
            rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "split_config_sha256": sha256_file(split_config),
        "counts": counts, "seed": config["seed"],
    }
