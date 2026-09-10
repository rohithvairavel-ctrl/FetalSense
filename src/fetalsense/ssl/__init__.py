"""Self-supervised learning objectives."""

from fetalsense.ssl.contrastive import NTXentLoss, make_contrastive_views

__all__ = ["NTXentLoss", "make_contrastive_views"]
