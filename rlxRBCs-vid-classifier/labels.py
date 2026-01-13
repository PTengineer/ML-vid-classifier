
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List


def _read_and_validate_csv(csv_path: Union[str, Path], *, label_col: str, filename_col: str):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in (label_col, filename_col) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain columns: '{label_col}' and '{filename_col}'. Missing: {missing}")

    return df


def load_label_mapping(csv_path: Union[str, Path], label_col: str = "label", filename_col: str = "filename") -> Dict[str, int]:
    """
    Returns a deterministic mapping from label name -> class index (0-indexed).

    Args:
        csv_path: Path to CSV file.
        label_col: Column name for label strings (default: 'label').
        filename_col: Column name for filenames (default: 'filename').

    Returns:
        Dict[str, int]: Mapping from label string to integer index.
    """
    df = _read_and_validate_csv(csv_path, label_col=label_col, filename_col=filename_col)
    unique_labels: List[str] = sorted(df[label_col].unique())
    label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_idx


def load_class_labels(csv_path: Union[str, Path], label_col: str = "label", filename_col: str = "filename") -> Dict[int, str]:
    """
    Extracts unique class names from a CSV and creates a 0-indexed ID-to-Label map.

    Args:
        csv_path: Path to the training CSV file containing a label column.
        label_col: Column name for label strings (default: 'label').
        filename_col: Column name for filenames (default: 'filename').

    Returns:
        Dict[int, str]: A dictionary mapping integer class IDs (0, 1, ...) to labels.
    """
    df = _read_and_validate_csv(csv_path, label_col=label_col, filename_col=filename_col)
    unique_labels: List[str] = sorted(df[label_col].unique())
    id_to_label: Dict[int, str] = {i: label for i, label in enumerate(unique_labels)}
    return id_to_label