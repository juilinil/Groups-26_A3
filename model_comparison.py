"""
Compare the trained expA and expB models with the same prompts.
Run from the project root:
    python model_comparison.py --expA "../B_handoff_20261029_170253/runs/expA_20261029_124740_320251/best" --expB "../B_handoff_20261029_170253/runs/expB_20261029_133311_034592/best"

    python model_comparison.py --expA "../B_handoff_20261029_170253/runs/expA_20261029_124740_320251/best" --expB "../B_handoff_20261029_170253/runs/expB_20261029_133311_034592/best" --device cuda:0

"""

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import model_test
from model_test import trainedSpeechT5


def load_comparison_manifest(manifest_path: str | Path,)\
        -> tuple[list[tuple[str, str]], dict[tuple[str, str], str], int]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Comparison manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    prompts = []
    prompt_ids = set()
    for item in manifest["prompts"]:
        prompt_id = str(item["id"]).strip()
        text = str(item["text"]).strip()
        if not prompt_id or not text or prompt_id in prompt_ids:
            raise ValueError(f"Invalid or duplicate prompt: {prompt_id}")
        prompts.append((prompt_id, text))
        prompt_ids.add(prompt_id)

    wav_names = {}
    for item in manifest["samples"]:
        model_name = str(item["model"]).strip()
        prompt_id = str(item["prompt_id"]).strip()
        wav_name = str(item["wav"]).strip()
        key = (model_name, prompt_id)
        if model_name not in {"expA", "expB"} or prompt_id not in prompt_ids:
            raise ValueError(f"Invalid sample entry: {item}")
        if Path(wav_name).name != wav_name or Path(wav_name).suffix.lower() != ".wav":
            raise ValueError(f"Invalid WAV filename: {wav_name}")
        if key in wav_names:
            raise ValueError(f"Duplicate WAV entry: {key}")
        wav_names[key] = wav_name

    expected = {
        (model_name, prompt_id)
        for model_name in ("expA", "expB")
        for prompt_id in prompt_ids
    }
    if set(wav_names) != expected:
        raise ValueError("Manifest must define one WAV filename for every prompt and experiment")

    seed = int(manifest.get("seed", 42))
    if seed < 0:
        raise ValueError("Manifest seed cannot be negative")

    return prompts, wav_names, seed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    # Record comparison result into a CSV file.
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    name: str,
    checkpoint: str | Path,
    prompts: list[tuple[str, str]],
    wav_names: dict[tuple[str, str], str],
    device: str,
    repeat: int,
    warmup: int,
    seed: int,
    wav_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    # Load the same checkpoint.
    load_started = time.perf_counter()
    tts = trainedSpeechT5(checkpoint=checkpoint, device=device)
    load_time = time.perf_counter() - load_started

    for _ in range(warmup):
        tts.generate_waveform(prompts[0][1])

    run_rows = []
    prompt_rows = []
    wav_dir.mkdir(parents=True, exist_ok=True)

    for prompt_index, (prompt_id, text) in enumerate(prompts, start=1):
        prompt_results = []
        last_result = None

        for repeat_index in range(1, repeat + 1):
            # Use the same random seed for the expA and expB models.
            run_seed = seed + prompt_index * 1000 + repeat_index
            tts.torch.manual_seed(run_seed)
            if tts.device.type == "cuda":
                tts.torch.cuda.manual_seed_all(run_seed)

            result = tts.generate_waveform(text)
            last_result = result

            generation_time = float(result["generation_time_s"])
            audio_duration = float(result["audio_duration_s"])
            if generation_time < 0 or audio_duration <= 0:
                raise RuntimeError(f"Invalid timing result: {name}, {prompt_id}")

            row = {
                "experiment": name,
                "prompt_id": prompt_id,
                "repeat": repeat_index,
                "text": text,
                "generation_time_s": generation_time,
                "audio_duration_s": audio_duration,
                "rtf": generation_time / audio_duration,
            }
            run_rows.append(row)
            prompt_results.append(row)

        wav_name = wav_names[(name, prompt_id)]
        wav_path = wav_dir / wav_name
        tts.save_waveform(last_result, wav_path)

        prompt_rows.append({
            "experiment": name,
            "prompt_id": prompt_id,
            "text": text,
            "repeat": repeat,
            "mean_generation_time_s": statistics.mean(
                item["generation_time_s"] for item in prompt_results
            ),
            "mean_audio_duration_s": statistics.mean(
                item["audio_duration_s"] for item in prompt_results
            ),
            "mean_rtf": statistics.mean(item["rtf"] for item in prompt_results),
            "wav_file": wav_name,
        })

        print(
            f"{name} | {prompt_id} | "
            f"latency={prompt_rows[-1]['mean_generation_time_s']:.4f}s | "
            f"RTF={prompt_rows[-1]['mean_rtf']:.4f} | saved={wav_name}",
            flush=True,
        )

    times = [item["generation_time_s"] for item in run_rows]
    rtfs = [item["rtf"] for item in run_rows]
    durations = [item["audio_duration_s"] for item in run_rows]

    summary = {
        "experiment": name,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "device": str(tts.device),
        "model_load_time_s": load_time,
        "prompt_count": len(prompts),
        "repeat_per_prompt": repeat,
        "warmup_runs": warmup,
        "mean_latency_s": statistics.mean(times),
        "median_latency_s": statistics.median(times),
        "stdev_latency_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "mean_audio_duration_s": statistics.mean(durations),
        "mean_rtf": statistics.mean(rtfs),
        "median_rtf": statistics.median(rtfs),
        "stdev_rtf": statistics.stdev(rtfs) if len(rtfs) > 1 else 0.0,
        "aggregate_rtf": sum(times) / sum(durations),
        "sample_rate_hz": int(last_result["sample_rate"]),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expA", required=True, help="Path to expA/best checkpoint")
    parser.add_argument("--expB", required=True, help="Path to expB/best checkpoint")
    parser.add_argument("--manifest", default="comparison_manifest.json")
    parser.add_argument("--device", default="auto", help="Same device for both models, e.g. cuda:0")
    parser.add_argument("--repeat", type=int, default=3, help="Measured runs per prompt")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs per model")
    parser.add_argument("--wav-dir", default="wav_examples")
    parser.add_argument("--results-dir", default="results/model_comparison")
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    prompts, wav_names, seed = load_comparison_manifest(args.manifest)
    wav_dir = Path(args.wav_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    all_run_rows = []
    all_prompt_rows = []
    summaries = []

    for name, checkpoint in (("expA", args.expA), ("expB", args.expB)):
        run_rows, prompt_rows, summary = evaluate_model(
            name=name,
            checkpoint=checkpoint,
            prompts=prompts,
            wav_names=wav_names,
            device=args.device,
            repeat=args.repeat,
            warmup=args.warmup,
            seed=seed,
            wav_dir=wav_dir,
        )
        all_run_rows.extend(run_rows)
        all_prompt_rows.extend(prompt_rows)
        summaries.append(summary)

    write_csv(results_dir / "per_run_metrics.csv", all_run_rows)
    write_csv(results_dir / "per_prompt_metrics.csv", all_prompt_rows)
    write_csv(results_dir / "summary.csv", summaries)

    expA = summaries[0]
    expB = summaries[1]
    comparison = [{
        "expA_mean_latency_s": expA["mean_latency_s"],
        "expB_mean_latency_s": expB["mean_latency_s"],
        "latency_change_percent": (
            (expB["mean_latency_s"] - expA["mean_latency_s"])
            / expA["mean_latency_s"] * 100
        ),
        "expA_mean_rtf": expA["mean_rtf"],
        "expB_mean_rtf": expB["mean_rtf"],
        "rtf_change_percent": (
            (expB["mean_rtf"] - expA["mean_rtf"])
            / expA["mean_rtf"] * 100
        ),
        "expA_aggregate_rtf": expA["aggregate_rtf"],
        "expB_aggregate_rtf": expB["aggregate_rtf"],
    }]
    write_csv(results_dir / "comparison.csv", comparison)

    torch = model_test.tts_b.torch
    transformers = model_test.tts_b.transformers
    selected_device = summaries[0]["device"]
    gpu_name = (
        torch.cuda.get_device_name(torch.device(selected_device))
        if str(selected_device).startswith("cuda") else None
    )

    metadata = {
        "purpose": "Fixed-prompt comparison; not held-out test-set evaluation",
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "prompt_count": len(prompts),
        "repeat_per_prompt": args.repeat,
        "warmup_runs_per_model": args.warmup,
        "seed": seed,
        "device": args.device,
        "timing_includes": "text normalisation, tokenisation and SpeechT5/HiFi-GAN generation; "
                           "CUDA synchronisation is handled by model_test.py",
        "timing_excludes": "model loading, warm-up and WAV file writing",
        "rtf_definition": "generation_time_s / audio_duration_s",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "gpu": gpu_name,
        "summaries": summaries,
    }
    with (results_dir / "evaluation_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("\nModel comparison completed.")
    print(f"WAV examples: {wav_dir}")
    print(f"Evaluation results: {results_dir}")
    print("\nSummary:")
    for item in summaries:
        print(
            f"{item['experiment']}: mean latency={item['mean_latency_s']:.4f}s, "
            f"mean RTF={item['mean_rtf']:.4f}, "
            f"aggregate RTF={item['aggregate_rtf']:.4f}"
        )


if __name__ == "__main__":
    main()
