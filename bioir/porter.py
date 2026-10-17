"""Original Porter (1980) stemmer, with the author's BLI/LOGI corrections in step 2.

Spec: https://tartarus.org/martin/PorterStemmer/def.txt
Only ASCII alphabetic words are stemmed; anything else is returned unchanged.
"""
from functools import lru_cache

# Rules are pre-sorted longest-first: the longest matching suffix decides,
# even when its measure condition then fails.
_STEP2 = sorted({
    "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance", "izer": "ize",
    "bli": "ble", "alli": "al", "entli": "ent", "eli": "e", "ousli": "ous",
    "ization": "ize", "ation": "ate", "ator": "ate", "alism": "al", "iveness": "ive",
    "fulness": "ful", "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
    "logi": "log",
}.items(), key=lambda rule: -len(rule[0]))
_STEP3 = sorted({
    "icate": "ic", "ative": "", "alize": "al", "iciti": "ic", "ical": "ic", "ful": "", "ness": "",
}.items(), key=lambda rule: -len(rule[0]))
_STEP4 = sorted(["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
                 "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"], key=len, reverse=True)


def _is_consonant(word: str, i: int) -> bool:
    if word[i] in "aeiou":
        return False
    if word[i] == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def measure(stem: str) -> int:
    """m in [C](VC)^m[V]: the number of vowel→consonant transitions."""
    flags = [_is_consonant(stem, i) for i in range(len(stem))]
    return sum(not a and b for a, b in zip(flags, flags[1:]))


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def cvc(word: str) -> bool:
    """*o: ends consonant-vowel-consonant, the last not w, x or y."""
    n = len(word)
    return (n >= 3 and _is_consonant(word, n - 3) and not _is_consonant(word, n - 2)
            and _is_consonant(word, n - 1) and word[-1] not in "wxy")


def _replace(word: str, rules) -> str:
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            return stem + replacement if measure(stem) > 0 else word
    return word


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    if len(word) <= 2 or not (word.isascii() and word.isalpha()):
        return word
    word = word.lower()

    # Step 1a: plurals.
    for suffix, replacement in (("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")):
        if word.endswith(suffix):
            word = word[:-len(suffix)] + replacement
            break

    # Step 1b: -eed, -ed, -ing.
    if word.endswith("eed"):
        if measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        for suffix in ("ed", "ing"):
            if word.endswith(suffix) and _has_vowel(word[:-len(suffix)]):
                word = word[:-len(suffix)]
                if word.endswith(("at", "bl", "iz")):
                    word += "e"
                elif _double_consonant(word) and word[-1] not in "lsz":
                    word = word[:-1]
                elif measure(word) == 1 and cvc(word):
                    word += "e"
                break

    # Step 1c: y → i.
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    word = _replace(word, _STEP2)
    word = _replace(word, _STEP3)

    # Step 4: drop suffixes when m > 1; -ion additionally needs a preceding s or t.
    for suffix in _STEP4:
        if word.endswith(suffix):
            base = word[:-len(suffix)]
            if measure(base) > 1 and (suffix != "ion" or base.endswith(("s", "t"))):
                word = base
            break

    # Step 5: final -e and -ll.
    if word.endswith("e"):
        m = measure(word[:-1])
        if m > 1 or (m == 1 and not cvc(word[:-1])):
            word = word[:-1]
    if word.endswith("ll") and measure(word) > 1:
        word = word[:-1]
    return word
