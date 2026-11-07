""" Shared export for the HuggingFace corpora (Svarah, IndicTTS Telugu).

Streams a dataset, resamples to the model rate, keeps utterances within a
duration window and writes ming024's raw_data layout:
    raw_path/<speaker>/<basename>.wav + .lab
then randomly subsamples to `total_hours` (train + val) with a fixed seed.
"""

import io
import os
import random

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.io import wavfile
from tqdm import tqdm


def export(config, get_speaker, get_text, per_speaker_cap=None):
    out_dir = config["path"]["raw_path"]
    sr = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    sel = config["preprocessing"]["selection"]

    ds = load_dataset(config["path"]["corpus_path"], split=sel["split"], streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    print("Features:", ds.features)

    kept = []  # (speaker, basename, seconds)
    hours = {}
    for idx, item in enumerate(tqdm(ds)):
        speaker = get_speaker(item)
        if per_speaker_cap and hours.get(speaker, 0) >= per_speaker_cap:
            continue
        text = get_text(item)
        if not text:
            continue
        wav, orig_sr = sf.read(io.BytesIO(item["audio"]["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        sec = len(wav) / orig_sr
        if not sel["min_sec"] <= sec <= sel["max_sec"]:
            continue
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
        wav = wav / max(np.abs(wav).max(), 1e-8) * max_wav_value * 0.95

        basename = "{}_{:05d}".format(speaker, idx)
        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        wavfile.write(os.path.join(out_dir, speaker, basename + ".wav"), sr, wav.astype(np.int16))
        with open(os.path.join(out_dir, speaker, basename + ".lab"), "w", encoding="utf-8") as f:
            f.write(text)
        kept.append((speaker, basename, sec))
        hours[speaker] = hours.get(speaker, 0) + sec / 3600

    # Random subsample to the requested total (5h train + 1h val)
    random.Random(1234).shuffle(kept)
    total, selected = 0.0, []
    for k in kept:
        if total >= sel["total_hours"] * 3600:
            for ext in (".wav", ".lab"):
                os.remove(os.path.join(out_dir, k[0], k[1] + ext))
        else:
            selected.append(k)
            total += k[2]
    print("Kept {} utterances, {:.2f} h, {} speakers".format(
        len(selected), total / 3600, len({s for s, _, _ in selected})))
    return selected
