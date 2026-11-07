""" Entry points behind project/<Language>/{preprocess,train,inference}.py.

Each language folder keeps a single config.yaml with `preprocess`, `model`
and `train` sections (ming024's three yaml files) plus the MFA command and
the assignment's test sentences. Paths inside it are relative to this
FastSpeech2/ directory, which is the working directory for every stage.
"""

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from string import punctuation

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from scipy.io import wavfile

# Telugu glyphs in plot titles: fall back to Windows' Nirmala UI (no shaping, so conjuncts render as separate glyphs)
plt.rcParams["font.family"] = ["DejaVu Sans", "Nirmala UI"]

FS2_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg, (cfg["preprocess"], cfg["model"], cfg["train"])


# ---------------------------------------------------------------- preprocess
def preprocess(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "prepare", "align", "features"], default="all")
    args = parser.parse_args()
    cfg, (pre, _, _) = load_config(config_path)
    os.chdir(FS2_DIR)

    if args.stage in ("all", "prepare"):
        import prepare_align
        prepare_align.main(pre)
    if args.stage in ("all", "align"):
        conda = os.path.expandvars(cfg["mfa"]["conda"])
        for cmd in cfg["mfa"]["commands"]:
            cmd = [conda, "run", "-n", cfg["mfa"]["env"], "--no-capture-output", "mfa"] + [str(c) for c in cmd]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
    if args.stage in ("all", "features"):
        from preprocessor.preprocessor import Preprocessor
        Preprocessor(pre).build_from_path()


# --------------------------------------------------------------------- train
def train(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    args = parser.parse_args()
    _, configs = load_config(config_path)
    os.chdir(FS2_DIR)
    import train as fs2_train
    fs2_train.main(args, configs)


# ------------------------------------------------------------ text frontends
def english_phones(text):
    """ Whole-sentence g2p_en, so homographs ("conduct" n./v.) get POS context.

    ming024 looks every word up in a static lexicon first, which gives both
    E2 sentences the same pronunciation. Numbers/times/dates are expanded
    first; spelling and spacing are otherwise kept exactly as given.
    """
    import inflect
    from g2p_en import G2p
    from text.numbers import normalize_numbers

    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1 \2", text)                     # 10:45 -> 10 45
    text = re.sub(r"\bAM\b", "ay em", text)
    text = re.sub(r"\b(\d{1,2}) (%s)\b" % months,                                  # 12 November -> 12th November
                  lambda m: inflect.engine().ordinal(m.group(1)) + " " + m.group(2), text)
    text = normalize_numbers(text).replace("-", " ")

    phones = []
    for p in G2p()(text.rstrip(punctuation)):
        if p == " ":
            continue
        phones.append("sp" if p in punctuation else p)
    return phones, text


def telugu_phones(text):
    from text import telugu

    norm = telugu.normalize_telugu(text)
    phones = []
    for tok in re.findall(r"[ఀ-౿‌‍]+|[,.?!;:]", norm.rstrip(",.?!;:")):
        phones += ["sp"] if tok in ",.?!;:" else telugu.word_to_phones(tok)
    return phones, norm


# ----------------------------------------------------------------- inference
def _bar(labels, values, ylabel, title, color, path):
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.28), 4))
    ax.bar(range(len(labels)), values, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Phoneme")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _loss_plots(log_path, out_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    names = ["total_loss", "mel_loss", "mel_postnet_loss", "pitch_loss", "energy_loss", "duration_loss"]
    for split, fname in (("train", "training_loss.png"), ("val", "validation_loss.png")):
        acc = EventAccumulator(os.path.join(log_path, split), size_guidance={"scalars": 0})
        acc.Reload()
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for ax, name in zip(axes.flat, names):
            ev = acc.Scalars("Loss/" + name)
            ax.plot([e.step for e in ev], [e.value for e in ev], lw=1)
            ax.set_title(name)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        fig.suptitle("{} loss".format("Training" if split == "train" else "Validation"))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)


def inference(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--speaker", type=str, default=None, help="speaker name from speakers.json")
    args = parser.parse_args()
    cfg, configs = load_config(config_path)
    pre, model_cfg, train_cfg = configs
    os.chdir(FS2_DIR)

    from text import text_to_sequence
    from utils.model import get_model, get_vocoder, vocoder_infer

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_cfg, device)

    prep = pre["path"]["preprocessed_path"]
    with open(os.path.join(prep, "speakers.json")) as f:
        speakers = json.load(f)
    with open(os.path.join(prep, "stats.json")) as f:
        stats = json.load(f)
    p_min, p_max, p_mean, p_std = stats["pitch"]
    e_min, e_max, e_mean, e_std = stats["energy"]

    speaker = args.speaker or cfg["inference"]["speaker"]
    if speaker == "most_data":  # speaker with the most training utterances
        with open(os.path.join(prep, "train.txt"), encoding="utf-8") as f:
            speaker = Counter(line.split("|")[1] for line in f).most_common(1)[0][0]
    print("Speaker:", speaker)

    results = cfg["inference"]["results_dir"]
    audio_dir = os.path.join(results, "test_generated_audios")
    plot_dir = os.path.join(results, "variance_adaptor_plots")
    os.makedirs(audio_dir, exist_ok=True)
    frontend = english_phones if pre["preprocessing"]["text"]["language"] == "en" else telugu_phones
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    hop = pre["preprocessing"]["stft"]["hop_length"]

    log_lines = []
    for sid, sentence in cfg["inference"]["test_sentences"].items():
        phones, normalized = frontend(sentence)
        seq = np.array(text_to_sequence("{" + " ".join(phones) + "}", []))
        batch = (
            torch.tensor([speakers[speaker]], device=device),
            torch.from_numpy(seq).long().unsqueeze(0).to(device),
            torch.tensor([len(seq)], device=device),
            len(seq),
        )
        with torch.no_grad():
            out = model(*batch)  # pyright: ignore[reportCallIssue]  (get_model returns a tuple only when train=True)
        mel_len = out[9][0].item()
        wav = vocoder_infer(out[1][:, :mel_len].transpose(1, 2), vocoder, model_cfg, pre,
                            lengths=[mel_len * hop])[0]
        wavfile.write(os.path.join(audio_dir, sid + ".wav"), sr, wav)

        # Phoneme-level predictions; pitch/energy were z-normalized in preprocessing
        dur = out[5][0, : len(seq)].cpu().numpy()
        pitch = out[2][0, : len(seq)].cpu().numpy() * p_std + p_mean
        energy = out[3][0, : len(seq)].cpu().numpy() * e_std + e_mean
        title = "{}: {}".format(sid, sentence if len(sentence) < 90 else sentence[:87] + "...")
        _bar(phones, dur, "Duration (mel frames)", "Predicted duration - " + title, "steelblue",
             os.path.join(plot_dir, "duration", sid + ".png"))
        _bar(phones, pitch, "F0 (Hz)", "Predicted pitch (F0) - " + title, "darkorange",
             os.path.join(plot_dir, "pitch", sid + ".png"))
        _bar(phones, energy, "Energy (frame L2 norm)", "Predicted energy - " + title, "seagreen",
             os.path.join(plot_dir, "energy", sid + ".png"))

        log_lines.append("{}\n  input:      {}\n  normalized: {}\n  phones:     {}\n  durations:  {}\n  seconds:    {:.2f}\n".format(
            sid, sentence, normalized, " ".join(phones), " ".join(str(int(d)) for d in dur), mel_len * hop / sr))
        print(log_lines[-1])

    with open(os.path.join(results, "test_sentences_phonemes.txt"), "w", encoding="utf-8") as f:
        f.write("speaker: {}\ncheckpoint: {}\n\n".format(speaker, args.restore_step) + "\n".join(log_lines))
    _loss_plots(train_cfg["path"]["log_path"], results)
    print("Wrote", os.path.abspath(results))
