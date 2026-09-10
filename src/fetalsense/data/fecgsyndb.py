"""FECGSYNDB synthetic fetal ECG loader stub.

Expected: PhysioNet FECGSYNDB with component-wise GT (fetal, maternal, noise).
Ideal for extraction supervision and SSL pretraining pools.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fetalsense.data.base import FetalECGWindowDataset

logger = logging.getLogger(__name__)


class FECGSYNDBDataset(FetalECGWindowDataset):
    """Stub — empty until downloads are present under ``paths.fecgsyndb``."""

    def __init__(self, root: str | Path, **kwargs) -> None:  # noqa: ANN003
        root = Path(root)
        if not root.exists():
            logger.warning("FECGSYNDB path missing: %s", root)
        else:
            logger.info(
                "FECGSYNDBDataset stub at %s — implement WFDB component reading "
                "(abdominal mixture + fetal source GT).",
                root,
            )
        super().__init__([])
