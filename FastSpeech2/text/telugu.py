""" Rule-based Telugu G2P + number normalization (added for Telugu support).

Telugu script is largely phonemic: every consonant letter carries an inherent
/a/ unless followed by a vowel sign (matra) or the virama (్). That rule plus
a letter->phone table is enough to produce a phone sequence that MFA can align
and FastSpeech2 can train on.
"""

import re

from num2words import num2words

_vowels = {
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ii", "ఉ": "u", "ఊ": "uu",
    "ఋ": "r u", "ౠ": "r uu", "ఎ": "e", "ఏ": "ee", "ఐ": "ai",
    "ఒ": "o", "ఓ": "oo", "ఔ": "au",
}
_matras = {
    "ా": "aa", "ి": "i", "ీ": "ii", "ు": "u", "ూ": "uu", "ృ": "r u",
    "ౄ": "r uu", "ె": "e", "ే": "ee", "ై": "ai", "ొ": "o", "ో": "oo", "ౌ": "au",
}
_consonants = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ల": "l", "వ": "v", "శ": "sh", "ష": "ss",
    "స": "s", "హ": "h", "ళ": "ll", "ఱ": "rr",
}
_virama = "్"
_anusvara = "ం"
_visarga = "ః"
_candrabindu = "ఁ"

# Anusvara takes the place of articulation of the following consonant.
_homorganic = {}
for _c in "కఖగఘ":
    _homorganic[_c] = "ng"
for _c in "చఛజఝ":
    _homorganic[_c] = "ny"
for _c in "టఠడఢ":
    _homorganic[_c] = "nn"
for _c in "తథదధ":
    _homorganic[_c] = "n"

phones = sorted(
    set(
        " ".join(
            list(_vowels.values())
            + list(_matras.values())
            + list(_consonants.values())
            + ["ng", "ny", "nn", "n", "m", "h"]
        ).split()
    )
)


def word_to_phones(word):
    out = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if ch in _consonants:
            out.append(_consonants[ch])
            if nxt in _matras:
                out += _matras[nxt].split()
                i += 1
            elif nxt == _virama:
                i += 1
            else:
                out.append("a")
        elif ch in _vowels:
            out += _vowels[ch].split()
        elif ch == _anusvara:
            out.append(_homorganic.get(nxt, "m"))
        elif ch == _visarga:
            out.append("h")
        # candrabindu, ZWJ/ZWNJ, nukta and anything non-Telugu are dropped
        i += 1
    return out


def _num(n):
    return num2words(int(n), lang="te")


def normalize_telugu(text):
    # 10:45 -> పది నలభై ఐదు
    text = re.sub(r"(\d+):(\d+)", lambda m: _num(m.group(1)) + " " + _num(m.group(2)), text)
    # 12వ (ordinal) -> పన్నెండవ: the number's final ు merges into the suffix
    text = re.sub(r"(\d+)వ", lambda m: " " + re.sub("ు$", "", _num(m.group(1))) + "వ", text)
    text = re.sub(r"\d+", lambda m: " " + _num(m.group(0)) + " ", text)
    text = text.replace("వేయిల", "వేల")  # num2words' bookish form -> spoken "రెండు వేల"
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    """Telugu words of a normalized sentence; punctuation is a word boundary."""
    return re.findall(r"[ఀ-౿‌‍]+", text)


if __name__ == "__main__":
    assert word_to_phones("పులి") == ["p", "u", "l", "i"]
    assert word_to_phones("అడవిలో") == ["a", "dd", "a", "v", "i", "l", "oo"]
    assert word_to_phones("పట్టు") == ["p", "a", "tt", "tt", "u"]
    assert word_to_phones("సంబంధం")[:4] == ["s", "a", "m", "b"]
    print(normalize_telugu("2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు"))
    print(phones)
