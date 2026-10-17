"""bioir.porter checked word by word against an independent implementation (NLTK)
over the whole vocabulary of the bundled corpus.

Which NLTK mode is the reference matters:
- ORIGINAL_ALGORITHM follows the 1980 paper literally. It lacks the BLI→BLE and
  LOGI→LOG corrections Martin Porter published later, which porter.py includes,
  so it disagrees on words such as "biology" / "cardiology"; it also stems
  two-letter words ("as" → "a").
- NLTK_EXTENSIONS (the default) adds NLTK-only rules on top (irregular forms,
  "-ied", "-fulli", "-lessli", a stricter step 1c ...), so it disagrees on
  correct output such as "aged" or "airway".
- MARTIN_EXTENSIONS is the corrected algorithm Porter publishes at
  tartarus.org/martin/PorterStemmer, the same specification porter.py follows,
  so it is the reference used here.

Skipped when nltk is not installed or the corpus is not next to the project.
Only ASCII words are compared: porter.py leaves non-ASCII words unchanged by design.
"""
import importlib.util
import unittest

from bioir.__main__ import DEFAULT_CORPUS
from bioir.corpus import load
from bioir.porter import stem
from bioir.text import words


@unittest.skipUnless(importlib.util.find_spec("nltk"), "nltk is not installed")
@unittest.skipUnless(DEFAULT_CORPUS.exists(), f"corpus not found at {DEFAULT_CORPUS}")
class PorterReferenceTest(unittest.TestCase):
    def test_whole_corpus_vocabulary_matches_nltk(self):
        from nltk.stem.porter import PorterStemmer
        reference = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)
        vocabulary = {w for d in load(DEFAULT_CORPUS) for w in words(f"{d.title} {d.abstract}") if w.isascii()}
        self.assertGreater(len(vocabulary), 5000)
        differences = [(w, stem(w), reference.stem(w)) for w in sorted(vocabulary) if stem(w) != reference.stem(w)]
        self.assertEqual(differences, [])


if __name__ == "__main__":
    unittest.main()
