# FastSpeech 2 TTS: English (Svarah) and Telugu (IndicTTS)

This project trains two separate FastSpeech 2 models, one for English and one for Telugu. It is based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (commit `d4e79eb`) and follows the paper [FastSpeech 2 (Ren et al., 2021)](https://arxiv.org/abs/2006.04558).

```
project/
├── FastSpeech2/      ming024 code, modified (shared by both languages, see "Changes")
├── English/          preprocess.py, train.py, inference.py, config.yaml, checkpoints/
├── Telugu/           preprocess.py, train.py, inference.py, config.yaml, checkpoints/
├── Results/<Lang>/   loss plots, test audio, variance-adaptor plots
└── Report/           Technical_Report.pdf
```

Each language has one `config.yaml` that combines ming024's `preprocess`, `model` and `train` YAML files. It also holds the MFA alignment commands and the test sentences.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Montreal Forced Aligner, in its own conda env named "mfa" (Miniforge)
conda create -n mfa -c conda-forge montreal-forced-aligner

# Svarah is gated: accept the terms at https://huggingface.co/datasets/ai4bharat/Svarah, then
hf auth login
```

The HiFi-GAN vocoder ships with ming024. Unzip `FastSpeech2/hifigan/generator_universal.pth.tar.zip` in place.

## Run (same commands for `English/` and `Telugu/`)

```bash
python English/preprocess.py                 # download, select 6 h, MFA alignment, features
python English/train.py                      # resume with --restore_step 20000
python English/inference.py --restore_step 40000
tensorboard --logdir English/logs            # monitor training
```

`preprocess.py --stage prepare|align|features` reruns a single stage:

1. **prepare** streams the dataset from HuggingFace, keeps clips between `min_sec` and `max_sec`, and resamples them to 22,050 Hz. It writes `data/<Lang>/raw/<speaker>/*.wav` and a matching `.lab` transcript for each clip. It then randomly subsamples to 6 h (seed 1234). For Telugu, this stage also writes the G2P lexicon.
2. **align** runs MFA. English uses the pretrained `english_us_arpa` model. Telugu has no pretrained model available, so MFA trains an acoustic model on this corpus using the G2P lexicon.
3. **features** computes mel-spectrograms, pitch, energy and durations, writes `stats.json`, and splits the data into 1 h validation and the rest for training. The split is by utterance, so no clip appears in both sets.

`inference.py` produces:
- `Results/<Lang>/test_generated_audios/<ID>.wav`
- `variance_adaptor_plots/{duration,pitch,energy}/<ID>.png`
- `training_loss.png` and `validation_loss.png`, read from the TensorBoard logs
- `test_sentences_phonemes.txt`, the normalized text, phonemes and durations for every test sentence

## Changes made to ming024/FastSpeech2

| File | Change |
|---|---|
| `text/telugu.py` (new) | Rule-based Telugu G2P: letter-to-phone table, inherent /a/ unless a vowel sign or virama follows, homorganic anusvara. Number and time expansion through `num2words(lang="te")`. |
| `text/symbols.py` | Adds Telugu phones as `@`-prefixed symbols. |
| `preprocessor/hf_corpus.py`, `svarah.py`, `telugu.py` (new) | Loaders that stream the HuggingFace corpora, resample, select by duration and cap total hours. Telugu balances the two speakers and writes the MFA lexicon. |
| `prepare_align.py` | Registers the Svarah and Telugu loaders. |
| `preprocessor/preprocessor.py` | Train/val split by duration (`val_hours`) instead of a fixed count. Reads MFA ≥ 2 TextGrids, where silences are unlabeled. Skips utterances with no alignment; the original code reused stale pitch values for these. Reads text as UTF-8. |
| `lang_runner.py` (new) | Entry points for the per-language scripts. The English frontend runs g2p_en on the whole sentence, so homographs like "conduct" (E2) get POS context; ming024 looks each word up in a lexicon with no context. It also expands times, dates and AM/PM. Writes the plots and audio. |
| `audio/stft.py`, `audio/audio_processing.py`, `utils/model.py` | Compatibility with librosa ≥ 0.10 (keyword arguments) and torch ≥ 2.6 (`weights_only`). |
| `dataset.py` | Keeps loaded samples in RAM (about 0.5 GB per language). Each `np.load` took about 9 ms on the training disk, which left the GPU idle; training went from 0.8 to 2.3 steps/s. |
| `utils/tools.py` | Casts energy to float before TensorBoard logging, which fixes a dtype crash at validation. |
| `preprocessor/svarah.py` | Svarah has no speaker column, so the speaker ID is a hash of the metadata profile (115 profiles). Its audio column is `audio_filepath`. |
| `config.yaml` (MFA) | Each language uses its own MFA temp directory. Both corpus folders are named `raw`, so they would otherwise collide in MFA's workspace. |

## Data and results

| | English (Svarah) | Telugu (IndicTTS) |
|---|---|---|
| Train / val | 3,480 utts, 4.32 h / 790 utts, 1.00 h | 2,443 utts, 3.43 h / 674 utts, 1.00 h |
| Speakers | 115 profiles | female and male |
| Training | 40k steps, 5 h 16 min on an RTX 3050 Laptop GPU | 40k steps, 6 h 42 min |
| Final val loss | 2.58 (mel 0.77) | 1.98 (mel 0.60) |
| Checkpoint used in `Results/` | 40000 | 25000 |

Telugu is under 5 h because only 5.39 h of the corpus passed the 1–11 s clip filter. Trimming silence to the MFA boundaries reduced that to 4.43 h. The `Results/<Lang>/checkpoint_comparison/` folders contain the test sentences synthesized at 10k, 25k and 40k steps. The full analysis is in [Report/Technical_Report.pdf](Report/Technical_Report.pdf); the source is `Report/Technical_Report.html`, printed to PDF with Edge.

## Checkpoints

Saved to `English/checkpoints/` and `Telugu/checkpoints/` as `<step>.pth.tar`, from 5000 to 40000.