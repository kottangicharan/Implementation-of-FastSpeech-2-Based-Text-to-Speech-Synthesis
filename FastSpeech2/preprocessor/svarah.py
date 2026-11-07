import hashlib
import re

from text import _clean_text
from preprocessor import hf_corpus
from preprocessor.hf_corpus import export

# Svarah has no speaker column; filename ids are per recording (~2.9k).
# The speaker-metadata tuple gives 110 unique profiles (~117 speakers per the paper).
# ponytail: speakers with identical metadata merge into one id; fine for a speaker embedding.
_PROFILE = ("gender", "age-group", "primary_language", "native_place_state",
            "native_place_district", "highest_qualification", "job_category", "occupation_domain")


def prepare_align(config):
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    def get_speaker(item):
        prof = "|".join(str(item.get(k)) for k in _PROFILE)
        tag = re.sub(r"\W", "", str(item.get("gender", ""))[:1] + str(item.get("primary_language", "")))
        return "{}_{}".format(tag, hashlib.md5(prof.encode()).hexdigest()[:6])

    def get_text(item):
        text = item.get("text")
        return _clean_text(text.strip(), cleaners) if text and text.strip() else None

    # Svarah's audio column is "audio_filepath"; the shared exporter expects "audio".
    # Patched only for this (English) process so hf_corpus.py stays untouched.
    # (datasets' rename_column/cast chain re-decodes audio and needs torchcodec, so rename per item.)
    class _Renamed:
        def __init__(self, ds):
            self.ds = ds.cast_column("audio_filepath", hf_corpus.Audio(decode=False))
            self.features = self.ds.features

        def cast_column(self, *_):  # export() casts "audio" to decode=False: already done
            return self

        def __iter__(self):
            for item in self.ds:
                item["audio"] = item.pop("audio_filepath")
                yield item

    orig = hf_corpus.load_dataset
    hf_corpus.load_dataset = lambda *a, **k: _Renamed(orig(*a, **k))
    try:
        export(config, get_speaker, get_text)
    finally:
        hf_corpus.load_dataset = orig
