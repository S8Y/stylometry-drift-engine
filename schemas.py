"""Tool schemas for the Stylometry Drift Engine plugin."""

STYLOMETRY_DRIFT = {
    "name": "stylometry_drift",
    "description": (
        "Rewrite text to reduce identifiable stylistic writing patterns while "
        "preserving the original meaning exactly. Alters sentence length, "
        "clause order, punctuation, and word choice using deterministic local "
        "transformations. Uses a synonym bank and structural rules — no "
        "external AI calls. The intensity parameter (0.0–1.0) controls how "
        "aggressively the style is transformed. Higher values produce more "
        "dramatic rewrites. Supply prior text via 'history' to avoid "
        "repeating patterns the author has already used."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to rewrite. Must preserve meaning exactly.",
            },
            "intensity": {
                "type": "number",
                "description": (
                    "How aggressively to transform the style. "
                    "0.0 = minimal changes (punctuation only). "
                    "0.3 = moderate sentence variation and some synonym swaps. "
                    "0.6 = aggressive restructuring and frequent synonym substitution. "
                    "1.0 = maximum transformation (all applicable techniques). "
                    "Default: 0.5."
                ),
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "history": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Prior text strings the author has written. "
                    "The engine avoids n-gram patterns found in these texts "
                    "to prevent repeating the author's stylistic fingerprints."
                ),
            },
        },
        "required": ["text"],
    },
}
