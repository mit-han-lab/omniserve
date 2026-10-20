"""Read RoPE settings from a HF config across transformers versions.

transformers >= 5 no longer sets ``config.rope_theta`` / ``config.rope_scaling``: both moved into
``config.rope_parameters`` (``rope_scaling`` is kept as an alias of it and always contains
``rope_type="default"``). ``getattr(config, "rope_theta", 10000)`` then silently returns 10000
(e.g. instead of Llama-3's 500000), and the non-None ``rope_scaling`` has no ``factor`` key.
"""


def get_rope_theta(config, default=10000):
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        theta = params.get("rope_theta", default)
    return theta


def get_rope_scaling(config):
    """Return ``{"type": ..., "factor": ...}``, or None if RoPE is not scaled."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        rope_scaling = getattr(config, "rope_parameters", None)
    if not rope_scaling:
        return None
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type in (None, "default") and "factor" not in rope_scaling:
        return None
    return {"type": rope_type, "factor": rope_scaling["factor"]}
