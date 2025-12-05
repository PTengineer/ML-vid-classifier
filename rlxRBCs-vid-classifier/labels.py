
import pandas as pd
from pathlib import Path

def load_class_labels(csv_path: str | Path):
    """Extracts sorted unique class names from a CSV with a 'label' column."""
    df = pd.read_csv(csv_path)
    return sorted(df["label"].unique())
