"""Stylometry Drift Engine — Hermes Plugin.

Registers the ``stylometry_drift`` tool that rewrites text to reduce
identifiable stylistic patterns while preserving meaning exactly.

All transformations are deterministic (seeded from the input text hash),
fully local, and require no external AI calls.
"""

from __future__ import annotations

import logging
from typing import Any

from . import schemas, tools

logger = logging.getLogger(__name__)


def register(ctx: Any) -> None:
    """Register the stylometry drift tool with the Hermes plugin system."""
    ctx.register_tool(
        name="stylometry_drift",
        toolset="default",
        schema=schemas.STYLOMETRY_DRIFT,
        handler=tools.stylometry_drift,
    )
    logger.info(
        "stylometry-drift-engine: registered stylometry_drift tool"
    )
