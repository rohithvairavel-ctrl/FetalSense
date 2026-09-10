"""Matonia / NIFECGDB-style loader stub.

Document expected path: ``paths.matonia`` containing WFDB or EDF records.
Populate ``FetalECGWindowDataset.samples`` once data is available.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fetalsense.data.base import FetalECGWindowDataset

logger = logging.getLogger(__name__)


class MatoniaDataset(FetalECGWindowDataset):
    """Stub — returns empty dataset until raw files are present."""

    def __init__(self, root: str | Path, **kwargs) -> None:  # noqa: ANN003
        root = Path(root)
        if not root.exists():
            logger.warning("Matonia path missing: %s", root)
        else:
            logger.info(
                "MatoniaDataset stub: found %s but parser not implemented; "
                "expected multi-channel abdominal ECG + optional annotations.",
                root,
            )
        super().__init__([])
