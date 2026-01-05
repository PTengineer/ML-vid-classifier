
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List

# Type annotation update: Path is usually imported from pathlib
def load_label_mapping(csv_path: Union[str, Path]) -> Dict[str, int]:
    """
    Returns a deterministic mapping from label name -> class index.
    Example: {"normal": 0, "mild": 1, ...}
    """
    df = pd.read_csv(csv_path)
    unique_labels: List[str] = sorted(df["label"].unique())

    label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(unique_labels)}

    return label_to_idx

def load_class_labels(csv_path: Union[str, Path]) -> Dict[int, str]:
    """
    Extracts unique class names from a CSV and creates a 0-indexed ID-to-Label map.

    Args:
        csv_path: Path to the training CSV file containing a 'label' column.

    Returns:
        Dict[int, str]: A dictionary mapping integer class IDs (0, 1, ...) 
                        to their corresponding human-readable string labels.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Get unique labels and ensure a consistent, sorted order
    # The sorted order guarantees that the class-to-index mapping is deterministic 
    # regardless of the order they appeared in the CSV file.
    unique_labels: List[str] = sorted(df["label"].unique())
    
    # 2. Create the ID-to-Label map (id2label)
    # This is the map required by the inference script to interpret the model's output index.
    id_to_label: Dict[int, str] = {i: label for i, label in enumerate(unique_labels)}
    
    return id_to_label