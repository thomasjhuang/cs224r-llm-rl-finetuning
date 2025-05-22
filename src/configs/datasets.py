"""Configuration for datasets used in the project."""
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DatasetConfig:
    """Base configuration for datasets."""
    name: str
    split: str = "train"
    max_length: int = 1024
    batch_size: int = 8
    subset: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "split": self.split,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "subset": self.subset,
        }

SMOLTALK_CONFIG = DatasetConfig(
    name="HuggingFaceTB/smol-smoltalk",
    max_length=1024,
    batch_size=8,
)

ULTRAFEEDBACK_CONFIG = DatasetConfig(
    name="HuggingFaceH4/ultrafeedback_binarized",
    max_length=1024,
    batch_size=4,
)

COUNTDOWN_CONFIG = DatasetConfig(
    name="Jiayi-Pan/Countdown-Tasks-3to4",
    max_length=512,
    batch_size=16,
)

DATASET_CONFIGS = {
    "smoltalk": SMOLTALK_CONFIG,
    "ultrafeedback": ULTRAFEEDBACK_CONFIG,
    "countdown": COUNTDOWN_CONFIG,
}

def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]
