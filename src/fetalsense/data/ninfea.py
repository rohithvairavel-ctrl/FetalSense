"""NInFEA non-invasive fetal ECG loader stub.

Note: NInFEA provides V-peaks (not always classic fQRS). Flag protocol in eval.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fetalsense.data.base import FetalECGWindowDataset

logger = logging.getLogger(__name__)


class NInFEADataset(FetalECGWindowDataset):
    """Stub — empty until downloads are present under ``paths.ninfea``."""

    def __init__(self, root: str | Path, **kwargs) -> None:  # noqa: ANN003
        root = Path(root)
        if not root.exists():
            logger.warning("NInFEA path missing: %s", root)
        else:
            logger.info(
                "NInFEADataset stub at %s — V-peak protocol must be stated in papers; "
                "do not claim fQRS without that caveat.",
                root,
            )
        super().__init__([])
