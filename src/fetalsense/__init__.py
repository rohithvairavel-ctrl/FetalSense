"""FetalSense: quality-aware Transformer for fetal ECG extraction & fQRS detection."""

__version__ = "0.1.0"

from fetalsense.models.fetalsense import FetalSense, build_model

__all__ = ["FetalSense", "build_model", "__version__"]
