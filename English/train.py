"""English FastSpeech 2: train. See README.md for usage."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "FastSpeech2"))

import lang_runner

if __name__ == "__main__":
    lang_runner.train(os.path.join(HERE, "config.yaml"))
