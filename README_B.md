# B integration for Group 26

This package connects B's model/training implementation to A's actual notebook and
committed manifest. It is an integration patch, not the final assessed submission.
Formal experiments, objective evaluation, qualitative analysis, report and slides
still need to be completed. The expA/expB configs are proposals for team agreement.

## What changed

- The notebook retains A's code and saved outputs. A's data path now reads
  `LJSPEECH_ROOT`; a setup cell supplies the verified local path when available.
- A's text and audio processing are reused in shared modules.
- The trainer reads the original manifest without regenerating it, verifies counts,
  unique IDs, portable paths and cleaned text, and records the manifest SHA256.
- Non-reentrant gradient checkpointing retains the fix verified in the earlier
  standalone two-step debug run.
- Training writes config, environment, sample IDs, logs, validation selection and
  inference checkpoints. It does not calculate final test metrics.
- The first integrated model load writes `configs/base_model.lock.json` to pin the
  pretrained starting revision. Commit it and share it before both formal runs.
- A formal fixed speaker vector is built from training references only. Each best
  checkpoint contains its own paired `speaker.npy` and `speaker.json`.
- The notebook contains complete copies of the module source in `%%writefile` cells.
  If you edit a module later, update its matching notebook source cell too.

The ZIP patch intentionally omits `splits/`, raw data, environments and weights.
Keep the repository's original `splits/` directory in place.

## Apply on the existing b-training branch

Copy the patch contents into `E:\work\Groups-26_A3`. It replaces the notebook and
`.gitignore` and adds the shared modules/configs. Inspect `git diff` afterwards.
Continue using the already verified environment at
`E:\work\TTS_A3\.venv\Scripts\python.exe`.

In a PowerShell terminal at the team repository root:

```powershell
$TtsPython = "E:\work\TTS_A3\.venv\Scripts\python.exe"
$env:LJSPEECH_ROOT = "E:\work\datasets\LJSpeech-1.1"
$env:TTS_HF_CACHE = "E:\work\_cache\huggingface"
& $TtsPython tts_b.py check --data-root $env:LJSPEECH_ROOT
```

This checks all 3000 local audio paths and must report 2400/300/300. The committed
manifest SHA256 at integration time is:

`a9c52de146c23328250419b6666fb71fbdb71f05db7ed31114b8f61050fa1b0c`

The content fingerprint is
`64a2db28bb18b5aedcc0be012d80d0e0569a774dad554b2e6238068574dcb885`.
It preserves row order and all fields while ignoring CSV line-ending differences.
Speaker provenance uses this fingerprint so Windows/Linux Git newline conversion
does not invalidate the same protocol; raw byte hashes are also retained for audit.

Run the integrated two-step check using the existing demo dependencies:

```powershell
& $TtsPython tts_b.py train --config configs/smoke.json --data-root $env:LJSPEECH_ROOT
```

It uses the first 16 train and first 4 validation records from A's manifest, runs
two optimizer updates, saves/reloads a checkpoint and writes a debug WAV.
Success ends with `INTEGRATED TRAINING CHECK PASSED`.
The public CMU ARCTIC speaker vector and debug checkpoint are not formal results.
The new integrated GPU run has not been executed by the authoring environment.

## Prepare the formal LJ speaker vector

The extra dependency SpeechBrain is needed only for this step. Install the shared
requirements; existing compatible CUDA packages are reused by pip.

```powershell
& $TtsPython -m pip install -r requirements.txt --cache-dir "E:\work\_cache\pip"
& $TtsPython tts_b.py speaker --data-root $env:LJSPEECH_ROOT
```

The encoder is `speechbrain/spkrec-xvect-voxceleb`. The 16 reference utterances are
the first training IDs in sorted order. Each x-vector is L2-normalized, the vectors
are averaged, and the mean is normalized. Both experiments must use the same
`speaker/lj_speaker.npy` and adjacent JSON. The JSON records the train reference IDs,
audio hashes and manifest hash. COPY fetching avoids Windows symlink privileges.

This is a deliberately simple single-speaker conditioning strategy and needs
listening-based validation. It is not a claim of optimal voice similarity.

## Confirm experiment design before full runs

The example settings hold the data, pretrained revision, voice conditioning,
batch size, accumulation, optimizer, training budget and validation rule fixed.
They propose changing learning rate only: expA=1e-5, expB=2e-5. Each uses 500
optimizer updates with 4 microbatches of size 1 per update, and validation every
100 updates on all 300 validation records. Confirm feasibility and settings with
the team after the integrated check, then preserve the chosen configurations.

```powershell
& $TtsPython tts_b.py train --config configs/expA.example.json --data-root $env:LJSPEECH_ROOT
& $TtsPython tts_b.py train --config configs/expB.example.json --data-root $env:LJSPEECH_ROOT
```

Each command starts from the pinned pretrained model, not a smoke checkpoint.
Each creates a new `runs/<name>_<timestamp>/` directory, stores every train loss,
and selects `best/` by minimum validation loss. The first checkpoint wins a tie.
Validation is the sample-count-weighted average of batch losses; with the provided
batch size 1 it is the mean per-utterance loss. It is a model-selection signal,
not the final objective quality metric. Test data is not used for model selection.

Outputs are inference checkpoints; optimizer-state resume is not implemented.
Feature preparation retains all selected utterances and fails on invalid input
rather than silently dropping or truncating it. Float32 feature tensors are held
in host memory. Runtime and memory on the full subset must be established locally.

## Interface for C

`results/latest_expA.json` and `results/latest_expB.json` contain actual best paths
after completed runs. Share each `best/` folder including processor and speaker
files. Example, replacing the checkpoint path with an actual saved run:

```powershell
& $TtsPython tts_b.py synth --checkpoint "runs/ACTUAL_RUN/best" --text "Hello world." --out "wav_examples/demo.wav"
```

Inference uses the same text normalization as A. Numeric prompts need numbers
spelled out where the tokenizer does not support digits. C should use the same
device, sentences, timing boundaries and warm-up policy for both experiments.
No latency, RTF, test quality or perceptual-improvement result is fabricated here.

## Validation and submission

Checked offline: original split bytes, all 3000 cleaned texts, counts, IDs,
relative paths, source syntax, notebook validity/source consistency, and protocol
failure cases. The authoring environment has no PyTorch/CUDA runtime; the integrated
backward pass and SpeechBrain extraction must be run on the verified local machine.

Run protocol tests if desired:

```powershell
& $TtsPython -m unittest discover -s tests -v
```

Select the verified Python as the VS Code notebook kernel. Run the setup and A's
dataset cells before B's notebook cells. `RUN_INTEGRATION_SMOKE` defaults to False;
enable it explicitly for the smoke run. Formal training is explicit, not automatic
on Run All. Before final submission, rerun the notebook and add actual formal
results, all required WAV examples, references and an accurate AI-use statement.
Package the modules, configs, original splits and dependencies it references in
the final code ZIP, along with the notebook and `wav_examples/`. Exclude the raw
LJSpeech dataset and local environment. Retain an accessible checkpoint download
route if weights are not packaged and the course permits it.

Sources are acknowledged in the notebook and source files. B's code was AI-assisted
and must be reviewed and understood before its final AI-use statement is completed.
