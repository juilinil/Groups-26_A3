"""
Run from the project root:
    python model_test.py --checkpoint "../B_handoff_20261029_170253/runs/expA_20261029_124740_320251/best"
    python model_test.py --checkpoint "../B_handoff_20261029_170253/runs/expB_20261029_133311_034592/best" --device cuda:0

After the model is loaded, enter sentences in the terminal. Type 'exit' to end.

"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import tts_b

VOCODER_ID = tts_b.VOCODER_ID


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        root = Path(cache_dir).expanduser().resolve()
        os.environ["TTS_HF_CACHE"] = str(root)
        os.environ["HF_HOME"] = str(root)
    return tts_b.cache_path()


class trainedSpeechT5:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        vocoder_id: str = VOCODER_ID,
        cache_dir: str | Path | None = None,
    ) -> None:
        get_cache_dir(cache_dir)

        # Load run evn.
        tts_b.runtime()
        self.torch = tts_b.torch
        self.np = tts_b.np
        self.sf = tts_b.sf

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {self.checkpoint}")

        required_files = ("config.json", "speaker.npy")
        missing = [name for name in required_files if not (self.checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Checkpoint is missing {missing}: {self.checkpoint}")

        weights_exist = (
            (self.checkpoint / "model.safetensors").is_file()
            or (self.checkpoint / "pytorch_model.bin").is_file()
            or bool(list(self.checkpoint.glob("model-*.safetensors")))
        )
        if not weights_exist:
            raise FileNotFoundError(f"No model weights found in {self.checkpoint}")

        if device == "auto":
            self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        else:
            self.device = self.torch.device(device)
            if self.device.type == "cuda" and not self.torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

        # Use the same model split and checkpoint format.
        self.processor = tts_b.SpeechT5Processor.from_pretrained(self.checkpoint)
        self.model = tts_b.SpeechT5ForTextToSpeech.from_pretrained(
            self.checkpoint
        ).to(self.device).eval()
        self.model.config.use_cache = True

        self.vocoder = tts_b.SpeechT5HifiGan.from_pretrained(
            vocoder_id, cache_dir=str(tts_b.cache_path() / "hub")
        ).to(self.device).eval()
        self.sample_rate = int(self.vocoder.config.sampling_rate)
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid vocoder sampling rate: {self.sample_rate}")

        speaker = self.np.load(
            self.checkpoint / "speaker.npy", allow_pickle=False
        ).astype(self.np.float32).reshape(1, -1)
        if speaker.shape != (1, 512) or not self.np.isfinite(speaker).all():
            raise ValueError("speaker.npy must contain one finite 512-dimensional vector")
        if float(self.np.linalg.norm(speaker)) == 0.0:
            raise ValueError("speaker.npy contains a zero vector")
        self.speaker_embeddings = self.torch.from_numpy(speaker).to(self.device)

    def _synchronise(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def generate_waveform(self, text: str) -> dict[str, Any]:
        # Generate audio and record time.
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        self._synchronise()
        started = time.perf_counter()

        clean_text = tts_b.normalise_text(text)
        if not clean_text:
            raise ValueError("Input text is empty after normalisation")
        encoded = self.processor(text=clean_text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected one sentence; token tensor shape is {tuple(input_ids.shape)}")
        if input_ids.shape[1] > 450:
            raise ValueError(f"Input is too long ({input_ids.shape[1]} tokens; maximum is 450)")
        unk_id = self.processor.tokenizer.unk_token_id
        if unk_id is not None and (input_ids == unk_id).any().item():
            raise ValueError("Input contains unknown tokens; use supported English characters")

        with self.torch.inference_mode():
            speech = self.model.generate_speech(
                input_ids.to(self.device),
                self.speaker_embeddings,
                vocoder=self.vocoder,
            )

        self._synchronise()
        generation_time = time.perf_counter() - started
        audio = speech.detach().float().cpu().numpy().reshape(-1)
        if (
            audio.size == 0
            or not self.np.isfinite(audio).all()
            or float(self.np.max(self.np.abs(audio))) == 0.0
        ):
            raise RuntimeError("SpeechT5 generated empty, non-finite or silent audio")

        duration = float(audio.size / self.sample_rate)
        return {
            "audio": audio,
            "sample_rate": self.sample_rate,
            "input_text": text,
            "normalised_text": clean_text,
            "generation_time_s": float(generation_time),
            "audio_duration_s": duration,
            "rtf": float(generation_time / duration),
        }

    def save_waveform(self, result: dict[str, Any], output_path: str | Path) -> Path:
        out = Path(output_path).expanduser().resolve()
        if out.suffix.lower() != ".wav":
            raise ValueError(f"Output path must end with .wav: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.sf.write(
            out,
            result["audio"],
            int(result["sample_rate"]),
            subtype="PCM_16",
        )
        return out

    def synthesize_to_file(
        self, text: str, output_path: str | Path
    ) -> dict[str, Any]:
        result = self.generate_waveform(text)
        out = self.save_waveform(result, output_path)
        return {
            key: value for key, value in result.items() if key != "audio"
        } | {"output_path": str(out)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the selected expA or expB best checkpoint",
    )
    parser.add_argument(
        "--out-dir", default="user_input_wav",
        help="Directory used to save generated WAV files",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--vocoder-id", default=VOCODER_ID)
    parser.add_argument("--cache-dir", default=None)
    return parser


def print_result(result: dict[str, Any], device: str) -> None:
    output = dict(result)
    output["device"] = device
    print(json.dumps(output, indent=2, ensure_ascii=False))


def run_interactive(tts: trainedSpeechT5, output_dir: str | Path) -> None:
    folder = Path(output_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    print("\nSpeechT5 interactive inference is ready.")
    print(f"Checkpoint: {tts.checkpoint}")
    print(f"Device: {tts.device}")
    print(f"WAV output directory: {folder}")
    print("Enter an English sentence. Type 'quit' or 'exit' to finish.\n")

    index = 1
    while True:
        try:
            text = input("Text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInteractive inference stopped.")
            break
        if text.lower() in {"quit", "exit"}:
            print("Interactive inference stopped.")
            break
        if not text:
            print("Please enter a sentence; empty input was skipped.")
            continue

        output_path = folder / f"user_input_{index:03d}.wav"
        while output_path.exists():
            index += 1
            output_path = folder / f"user_input_{index:03d}.wav"
        try:
            result = tts.synthesize_to_file(text, output_path)
            print_result(result, str(tts.device))
            index += 1
        except Exception as exc:
            print(f"Generation failed: {type(exc).__name__}: {exc}")
            print("Please try another sentence.\n")


def main() -> None:
    args = build_parser().parse_args()
    get_cache_dir(args.cache_dir)
    started = time.perf_counter()
    tts = trainedSpeechT5(
        checkpoint=args.checkpoint,
        device=args.device,
        vocoder_id=args.vocoder_id,
        cache_dir=args.cache_dir,
    )
    load_seconds = time.perf_counter() - started
    print(f"Model loaded in {load_seconds:.3f} seconds on {tts.device}.")

    run_interactive(tts, args.out_dir)


if __name__ == "__main__":
    main()
