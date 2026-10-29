"""B's shared SpeechT5 training entry point; runtime imports are lazy.

Sources: Hugging Face's SpeechT5 TTS guide, model cards for
microsoft/speecht5_tts and microsoft/speecht5_hifigan, and SpeechBrain's
speechbrain/spkrec-xvect-voxceleb model card. A's processing is in tts_protocol
and tts_audio. Experiment results are produced only by actual local runs.
"""
import argparse
from datetime import datetime, timezone
import gc
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import zipfile
from tts_protocol import normalise_text, read_protocol, sha256_file

ROOT = Path(__file__).resolve().parent
MODEL_ID = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan"
PUBLIC_SPEAKER_ID = "Matthijs/cmu-arctic-xvectors"


def cache_path():
    default = (r"E:\work\_cache\huggingface" if os.name == "nt" and Path("E:/work").is_dir()
               else str(Path.home() / ".cache/huggingface"))
    root = Path(os.environ.get("TTS_HF_CACHE", os.environ.get("HF_HOME", default)))
    os.environ["HF_HOME"] = str(root)
    (root / "hub").mkdir(parents=True, exist_ok=True)
    return root


def runtime():
    cache_path()  # before importing the Hub
    global np, sf, torch, transformers
    global SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def locked_base_model():
    """The first successful load pins the shared base revision for both runs."""
    lock_path = ROOT / "configs/base_model.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    if lock and (lock.get("model") != MODEL_ID or not lock.get("revision")):
        raise ValueError("Invalid shared base-model lock")
    kwargs = {"cache_dir": str(cache_path() / "hub")}
    if lock:
        kwargs["revision"] = lock["revision"]
    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID, **kwargs)
    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        raise RuntimeError("Could not identify the base-model revision for reproducibility")
    processor = SpeechT5Processor.from_pretrained(MODEL_ID, revision=revision,
                                                 cache_dir=str(cache_path() / "hub"))
    if not lock:
        write_json(lock_path, {"model": MODEL_ID, "revision": revision})
    return model, processor


def protocol(args, check_files=True):
    return read_protocol(ROOT / "splits/split_manifest.csv",
                         ROOT / "splits/split_config.json", args.data_root, check_files)


def check(args):
    groups, info = protocol(args, check_files=not args.manifest_only)
    print(json.dumps(info, indent=2))
    print("Audio paths checked:", not args.manifest_only)
    print("MANIFEST CHECK PASSED; committed split preserved.")


def prepare_speaker(args):
    """Create one fixed LJ voice vector using only training references."""
    runtime()
    from tts_audio import load_audio_for_speecht5
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    groups, info = protocol(args)
    if not 1 <= args.references <= len(groups["train"]):
        raise ValueError("Invalid training-reference count")
    refs = sorted(groups["train"], key=lambda item: item["utterance_id"])[:args.references]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=str(cache_path() / "speaker_encoder"),
        huggingface_cache_dir=str(cache_path() / "hub"),
        local_strategy=LocalStrategy.COPY, run_opts={"device": device},
    )
    vectors, reference_hashes = [], {}
    with torch.inference_mode():
        for index, row in enumerate(refs, 1):
            audio, _ = load_audio_for_speecht5(row["audio_path"])
            v = encoder.encode_batch(torch.from_numpy(audio).unsqueeze(0).to(device)).reshape(-1)
            if v.numel() != 512 or not torch.isfinite(v).all() or v.norm().item() == 0:
                raise ValueError(f"Invalid x-vector: {row['utterance_id']}")
            vectors.append(torch.nn.functional.normalize(v, dim=0).cpu())
            reference_hashes[row["utterance_id"]] = sha256_file(row["audio_path"])
            print(f"Training speaker reference {index}/{len(refs)}: {row['utterance_id']}", flush=True)
    vector = torch.stack(vectors).mean(0)
    vector = torch.nn.functional.normalize(vector, dim=0).numpy().astype(np.float32)
    out = Path(args.out).resolve()
    if out.suffix != ".npy":
        raise ValueError("--out must end in .npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(f"Speaker files already exist: {out}. Choose another --out to regenerate.")
    np.save(out, vector, allow_pickle=False)
    write_json(out.with_suffix(".json"), {
        "method": "L2-normalized mean of L2-normalized x-vectors from training references",
        "encoder": "speechbrain/spkrec-xvect-voxceleb", "reference_split": "train",
        "reference_ids": list(reference_hashes), "reference_wav_sha256": reference_hashes,
        "manifest_sha256": info["manifest_sha256"], "vector_sha256": sha256_file(out),
        "manifest_content_sha256": info["manifest_content_sha256"],
        "selection": "First N training IDs in sorted order", "references": len(refs),
    })
    print("Training-only speaker vector saved:", out)


def speaker_vector(args, groups, info, debug):
    if debug:
        from huggingface_hub import hf_hub_download
        archive = hf_hub_download(repo_id=PUBLIC_SPEAKER_ID, repo_type="dataset",
                                  filename="spkrec-xvect.zip", cache_dir=str(cache_path() / "hub"))
        with zipfile.ZipFile(archive) as handle:
            names = sorted(n for n in handle.namelist() if n.endswith(".npy"))
            preferred = [n for n in names if "cmu_us_slt_arctic" in n]
            if not names:
                raise ValueError("No public debug speaker vectors found")
            member = (preferred or names)[0]
            vector = np.load(io.BytesIO(handle.read(member)), allow_pickle=False)
        metadata = {"debug_only": True, "source": PUBLIC_SPEAKER_ID, "member": member}
    else:
        path = Path(args.speaker)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8-sig"))
        vector = np.load(path, allow_pickle=False)
        references = metadata.get("reference_ids", [])
        train_ids = {row["utterance_id"] for row in groups["train"]}
        if (metadata.get("reference_split") != "train" or not references
                or not set(references).issubset(train_ids)
                or metadata.get("manifest_content_sha256") != info["manifest_content_sha256"]
                or metadata.get("vector_sha256") != sha256_file(path)):
            raise ValueError("Speaker provenance does not match this training split")
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape != (512,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("Invalid speaker vector")
    vector = vector / np.linalg.norm(vector)
    return vector, metadata


def features(rows, processor, factor):
    from tts_audio import load_audio_for_speecht5
    result = []
    for index, row in enumerate(rows, 1):
        audio, rate = load_audio_for_speecht5(row["audio_path"])
        item = processor(text=row["clean_text"], audio_target=audio,
                         sampling_rate=rate, return_attention_mask=False)
        ids = item["input_ids"]
        if len(ids) > 450 or processor.tokenizer.unk_token_id in ids:
            raise ValueError(f"Unsupported/overlong input for {row['utterance_id']}; no rows are silently dropped")
        labels = np.asarray(item["labels"][0], dtype=np.float32)
        length = len(labels) - len(labels) % factor
        if labels.ndim != 2 or labels.shape[1] != 80 or length < factor or not np.isfinite(labels).all():
            raise ValueError(f"Invalid labels for {row['utterance_id']}")
        result.append({"input_ids": ids, "labels": labels[:length]})
        if index % 100 == 0 or index == len(rows):
            print(f"Prepared {index}/{len(rows)} utterances", flush=True)
    return result


def batch(items, processor, vector, device):
    value = processor.pad(
        input_ids=[{"input_ids": item["input_ids"]} for item in items],
        labels=[{"input_values": item["labels"]} for item in items], return_tensors="pt",
    )
    value["labels"] = value["labels"].masked_fill(value["decoder_attention_mask"].unsqueeze(-1).ne(1), -100)
    del value["decoder_attention_mask"]
    value["speaker_embeddings"] = vector.expand(len(items), -1)
    return {key: tensor.to(device) for key, tensor in value.items()}


def validation(model, items, processor, vector, device, batch_size, seed):
    model.eval()
    total = 0.0
    # SpeechT5's decoder prenet uses dropout during evaluation. Fix and restore
    # RNG state so model selection is comparable and does not change training RNG.
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(seed + 100000)
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                part = items[start:start + batch_size]
                loss = float(model(**batch(part, processor, vector, device)).loss.cpu())
                if not math.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss")
                total += loss * len(part)
    model.train()
    return total / len(items)


def train(args):
    runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires the verified CUDA environment")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    groups, info = protocol(args)
    debug = bool(cfg.get("debug", False))
    for key in ("max_steps", "batch_size", "gradient_accumulation_steps", "eval_steps"):
        if not isinstance(cfg.get(key), int) or cfg[key] <= 0:
            raise ValueError(f"Invalid config value: {key}")
    if not math.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0:
        raise ValueError("Invalid learning rate")
    if cfg["seed"] != info["seed"]:
        raise ValueError("Training seed differs from the committed split protocol")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", torch.cuda.current_device())
    vector_np, speaker_info = speaker_vector(args, groups, info, debug)
    vector = torch.from_numpy(vector_np).unsqueeze(0).to(device)
    run = ROOT / "runs" / (cfg["name"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run.mkdir(parents=True, exist_ok=False)
    train_rows = groups["train"][:16] if debug else groups["train"]
    val_rows = groups["validation"][:4] if debug else groups["validation"]
    print("Run:", run, flush=True)
    print("DEBUG ONLY" if debug else "FORMAL TRAINING; TEST SET HELD OUT", flush=True)
    print("Counts used:", len(train_rows), "train /", len(val_rows), "validation", flush=True)
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        git_head, git_dirty = None, None
    model, processor = locked_base_model()
    model = model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    record = {
        "config": cfg, "protocol": info, "data_root": str(Path(args.data_root).resolve()),
        "train_ids_used": [row["utterance_id"] for row in train_rows],
        "validation_ids_used": [row["utterance_id"] for row in val_rows],
        "speaker": speaker_info, "seed": cfg["seed"], "precision": "float32",
        "optimizer": "AdamW", "weight_decay": 0.01, "lr_schedule": "constant",
        "gradient_checkpointing": {"use_reentrant": False},
        "checkpoint_rule": "lowest validation loss; first checkpoint wins ties",
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(device),
        "model": MODEL_ID, "model_revision": getattr(model.config, "_commit_hash", None),
        "git_head": git_head, "git_dirty": git_dirty,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run / "run_config.json", record)
    train_items = features(train_rows, processor, model.config.reduction_factor)
    val_items = features(val_rows, processor, model.config.reduction_factor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
    order, cursor = [], 0
    sampler = random.Random(cfg["seed"])
    best_loss, best_step = float("inf"), None
    model.train()
    for step in range(1, cfg["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(cfg["gradient_accumulation_steps"]):
            picked = []
            for _ in range(cfg["batch_size"]):
                if cursor == len(order):
                    order = list(range(len(train_items)))
                    sampler.shuffle(order)
                    cursor = 0
                picked.append(train_items[order[cursor]])
                cursor += 1
            loss = model(**batch(picked, processor, vector, device)).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss at optimizer step {step}")
            losses.append(float(loss.detach().cpu()))
            (loss / cfg["gradient_accumulation_steps"]).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm).item():
            raise RuntimeError(f"Non-finite gradients at step {step}")
        optimizer.step()
        entry = {"step": step, "train_loss": sum(losses) / len(losses)}
        if step % cfg["eval_steps"] == 0 or step == cfg["max_steps"]:
            score = validation(model, val_items, processor, vector, device, cfg["batch_size"], cfg["seed"])
            entry["validation_loss"] = score
            if score < best_loss:
                best_loss, best_step = score, step
                checkpoint = run / "best"
                model.save_pretrained(checkpoint, safe_serialization=True)
                processor.save_pretrained(checkpoint)
                np.save(checkpoint / "speaker.npy", vector_np, allow_pickle=False)
                write_json(checkpoint / "speaker.json", speaker_info)
                write_json(checkpoint / "training_record.json", record)
        with (run / "log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    summary = {"debug": debug, "best_validation_loss": best_loss, "best_step": best_step,
               "best_checkpoint": str(run / "best"), "completed_steps": cfg["max_steps"],
               "manifest_sha256": info["manifest_sha256"], "test_evaluation": "not performed"}
    summary["manifest_content_sha256"] = info["manifest_content_sha256"]
    write_json(run / "summary.json", summary)
    write_json(ROOT / "results" / f"latest_{cfg['name']}.json", summary)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, model, loss
    gc.collect()
    torch.cuda.empty_cache()
    if debug:
        demo = argparse.Namespace(checkpoint=run / "best", text="This is our shared training pipeline.",
                                  out=ROOT / "wav_examples" / f"{run.name}.wav")
        synth(demo)
        print("INTEGRATED TRAINING CHECK PASSED", flush=True)
    else:
        print("TRAINING COMPLETE. Send the best checkpoint and paired speaker files to C.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def synth(args):
    """Checkpoint reload helper. C can reuse it for uniform evaluation."""
    runtime()
    checkpoint = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SpeechT5Processor.from_pretrained(checkpoint)
    model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint).to(device).eval()
    model.config.use_cache = True
    vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID, cache_dir=str(cache_path() / "hub")).to(device).eval()
    vector = np.load(checkpoint / "speaker.npy", allow_pickle=False).astype(np.float32).reshape(1, -1)
    if vector.shape != (1, 512) or not np.isfinite(vector).all():
        raise ValueError("Invalid checkpoint speaker vector")
    text = normalise_text(args.text)
    if not text:
        raise ValueError("Text is empty")
    ids = processor(text=text, return_tensors="pt")["input_ids"]
    if ids.shape[1] > 450 or (ids == processor.tokenizer.unk_token_id).any().item():
        raise ValueError("Unsupported/overlong text. Spell out numbers and use supported English characters.")
    with torch.inference_mode():
        speech = model.generate_speech(ids.to(device), torch.from_numpy(vector).to(device), vocoder=vocoder)
    audio = speech.cpu().numpy().reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
        raise RuntimeError("Invalid or silent generated audio")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, int(vocoder.config.sampling_rate), subtype="PCM_16")
    print("Saved WAV:", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    default_root = os.environ.get("LJSPEECH_ROOT", "LJSpeech-1.1")
    p = sub.add_parser("check")
    p.add_argument("--data-root", default=default_root)
    p.add_argument("--manifest-only", action="store_true")
    p.set_defaults(func=check)
    p = sub.add_parser("speaker")
    p.add_argument("--data-root", default=default_root)
    p.add_argument("--references", type=int, default=16)
    p.add_argument("--out", default=str(ROOT / "speaker/lj_speaker.npy"))
    p.set_defaults(func=prepare_speaker)
    p = sub.add_parser("train")
    p.add_argument("--data-root", default=default_root)
    p.add_argument("--config", required=True)
    p.add_argument("--speaker", default=str(ROOT / "speaker/lj_speaker.npy"))
    p.set_defaults(func=train)
    p = sub.add_parser("synth")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=synth)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
