from pathlib import Path

import numpy as np
import pandas as pd

# crawl_length_count.py расположен в teleopit/train_mimic/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED_ROOT = PROJECT_ROOT / "data" / "seed"
CSV_ROOT = SEED_ROOT / "g1" / "csv"
METADATA = SEED_ROOT / "seed_crawl_metadata_3h.csv"

RAW_FPS = 120
TARGET_FPS = 30

print(f"Корень Teleopit: {PROJECT_ROOT}")
print(f"Папка движений:  {CSV_ROOT}")
print(f"Metadata:         {METADATA}")

if not METADATA.is_file():
    raise FileNotFoundError(f"Metadata не найден: {METADATA}")

if not CSV_ROOT.is_dir():
    raise FileNotFoundError(
        f"Папка с движениями не найдена: {CSV_ROOT}\n"
        "Проверьте фактическую структуру папки data/seed."
    )