import os
from pathlib import Path

HELICOID_DIR = Path(os.getenv("HELICOID_DIR"))
FIVES_DIR = Path(os.getenv("FIVES_DIR"))
FIVES_RANDOM_CROPS_DIR = Path(os.getenv("FIVES_RANDOM_CROPS_DIR"))
MODELS_DIR = Path(os.getenv("MODELS_DIR"))
HELICOID_WITH_LABELS_DIR = Path(os.getenv("HELICOID_WITH_LABELS_DIR"))
RING_LABELS_DIR = Path(os.getenv("RING_LABELS_DIR"))