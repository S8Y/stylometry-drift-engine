"""Tool handlers for the Stylometry Drift Engine plugin."""

from __future__ import annotations

import json
import logging

from . import engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool handler: stylometry_drift
# ---------------------------------------------------------------------------


def stylometry_drift(args: dict, **kwargs) -> str:
    """Rewrite text to reduce identifiable stylistic patterns.

    Parameters are passed by the LLM via the tool call:
        text (str): The text to rewrite.
        intensity (float, optional): 0.0–1.0, default 0.5.
        history (list of str, optional): Prior texts whose patterns to avoid.
    """
    text = args.get("text", "")
    if not text or not isinstance(text, str) or not text.strip():
        return json.dumps({
            "error": "No text provided. Supply a non-empty 'text' string.",
        })

    intensity_raw = args.get("intensity", 0.5)
    try:
        intensity = float(intensity_raw)
    except (TypeError, ValueError):
        intensity = 0.5
    intensity = max(0.0, min(1.0, intensity))

    history = args.get("history")
    if history is not None and not isinstance(history, list):
        history = None

    try:
        rewritten = engine.rewrite(
            text=text,
            intensity=intensity,
            history=history,
        )
        return json.dumps({
            "original": text,
            "rewritten": rewritten,
            "intensity": intensity,
            "transformations_applied": text != rewritten,
        })
    except Exception as exc:
        logger.exception("stylometry_drift failed")
        return json.dumps({
            "original": text,
            "error": f"Rewriting failed: {exc}",
        })
